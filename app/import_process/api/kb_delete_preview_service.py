import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set
from urllib.parse import unquote, urlparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.clients.milvus_utils import get_milvus_client
from app.clients.minio_utils import get_minio_client
from app.conf.milvus_config import milvus_config
from app.conf.minio_config import minio_config
from app.core.logger import logger
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.path_util import PROJECT_ROOT


DeleteMode = Literal["soft", "full"]

CHUNK_IMAGE_FIELDS = ["content", "figures", "file_title", "item_name"]
ITEM_COLLECTION_FIELDS = ["paper_title", "item_name"]
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "output"


class DeleteKnowledgeRequest(BaseModel):
    item_name: str = Field(..., description="要删除的知识项名称")
    mode: DeleteMode = Field(
        "soft",
        description="soft 仅删 Milvus；full 额外清理 MinIO 和本地 output 产物",
    )
    dry_run: bool = Field(
        True,
        description="True 只预览删除计划；False 执行真实删除",
    )


class KnowledgeDeletePreviewService:
    """
    独立预览版知识库删除服务。

    说明：
    1. 不接入现有主服务，只提供一个可单独查看/运行的实现样例。
    2. `soft` 模式复用当前“同名覆盖”的核心思路，只清理 Milvus。
    3. `full` 模式在 `soft` 基础上，尽力清理 MinIO 图片/原文件和本地 output 产物。
    4. 由于当前项目没有为“原始 PDF/任务目录”建立稳定索引，`full` 中的 MinIO/本地删除属于 best-effort。
    """

    def delete_by_item_name(
        self, item_name: str, mode: DeleteMode = "soft", dry_run: bool = True
    ) -> Dict[str, Any]:
        clean_name = (item_name or "").strip()
        if not clean_name:
            raise ValueError("item_name 不能为空")

        logger.info(
            f"知识库删除预览开始: item_name={clean_name}, mode={mode}, dry_run={dry_run}"
        )

        preview: Dict[str, Any] = {
            "item_name": clean_name,
            "mode": mode,
            "dry_run": dry_run,
            "milvus": {},
            "minio": {"image_objects": [], "raw_file_objects": []},
            "local_output": {"paths": []},
            "notes": [],
        }

        client = get_milvus_client()
        if not client:
            raise ValueError("Milvus 客户端不可用，无法执行删除预览")

        chunk_rows = self._query_chunk_rows(client, clean_name)
        preview["milvus"]["chunks_collection"] = self._delete_from_collection(
            client=client,
            collection_name=milvus_config.chunks_collection,
            candidate_fields=["item_name"],
            target_value=clean_name,
            dry_run=dry_run,
            expected_rows=chunk_rows,
        )

        item_collection_result = self._delete_from_collection(
            client=client,
            collection_name=milvus_config.item_name_collection,
            candidate_fields=ITEM_COLLECTION_FIELDS,
            target_value=clean_name,
            dry_run=dry_run,
        )
        preview["milvus"]["item_name_collection"] = item_collection_result

        if mode == "full":
            image_objects = sorted(self._collect_image_object_names(chunk_rows))
            raw_file_objects = sorted(self._collect_raw_file_objects(clean_name, chunk_rows))
            local_paths = sorted(str(p) for p in self._collect_local_output_paths(chunk_rows))

            preview["minio"]["image_objects"] = image_objects
            preview["minio"]["raw_file_objects"] = raw_file_objects
            preview["local_output"]["paths"] = local_paths

            if dry_run:
                preview["notes"].append("dry_run=True，MinIO 和本地文件仅预览，不执行删除。")
            else:
                preview["minio"]["deleted_image_objects"] = self._delete_minio_objects(
                    image_objects
                )
                preview["minio"]["deleted_raw_file_objects"] = self._delete_minio_objects(
                    raw_file_objects
                )
                preview["local_output"]["deleted_paths"] = self._delete_local_paths(
                    [Path(p) for p in local_paths]
                )

            preview["notes"].append(
                "full 模式的附件/本地产物清理是 best-effort；如果历史数据缺少稳定映射，可能需要人工补删。"
            )

        return preview

    def _query_chunk_rows(self, client, item_name: str) -> List[Dict[str, Any]]:
        collection_name = milvus_config.chunks_collection
        if not collection_name:
            logger.warning("CHUNKS_COLLECTION 未配置，跳过 chunk 查询")
            return []

        filter_expr = self._build_filter("item_name", item_name)
        try:
            rows = client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=CHUNK_IMAGE_FIELDS,
                limit=16384,
            )
            return rows or []
        except Exception as exc:
            logger.warning(f"查询 chunks collection 失败: {exc}")
            return []

    def _delete_from_collection(
        self,
        client,
        collection_name: Optional[str],
        candidate_fields: List[str],
        target_value: str,
        dry_run: bool,
        expected_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "collection": collection_name or "",
            "attempts": [],
        }
        if not collection_name:
            result["skipped"] = "collection 未配置"
            return result

        try:
            if not client.has_collection(collection_name=collection_name):
                result["skipped"] = f"collection 不存在: {collection_name}"
                return result
        except Exception as exc:
            result["error"] = f"检查 collection 失败: {exc}"
            return result

        for field_name in candidate_fields:
            filter_expr = self._build_filter(field_name, target_value)
            if expected_rows is not None and field_name == "item_name":
                matched_rows = expected_rows
            else:
                matched_rows = self._safe_query_rows(
                    client,
                    collection_name=collection_name,
                    filter_expr=filter_expr,
                    field_name=field_name,
                )

            attempt: Dict[str, Any] = {
                "field": field_name,
                "filter": filter_expr,
                "matched_count_preview": len(matched_rows),
            }
            result["attempts"].append(attempt)

            if dry_run:
                continue

            if not matched_rows:
                attempt["deleted"] = False
                attempt["reason"] = "未查询到匹配行"
                continue

            try:
                delete_result = client.delete(
                    collection_name=collection_name,
                    filter=filter_expr,
                )
                attempt["deleted"] = True
                attempt["delete_result"] = delete_result
                if hasattr(client, "flush"):
                    try:
                        client.flush(collection_name=collection_name)
                    except Exception as exc:
                        attempt["flush_warning"] = str(exc)
            except Exception as exc:
                attempt["deleted"] = False
                attempt["error"] = str(exc)

        return result

    def _safe_query_rows(
        self,
        client,
        collection_name: str,
        filter_expr: str,
        field_name: str,
    ) -> List[Dict[str, Any]]:
        try:
            rows = client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[field_name],
                limit=16384,
            )
            return rows or []
        except Exception as exc:
            logger.debug(
                f"预查询失败，collection={collection_name}, field={field_name}, error={exc}"
            )
            return []

    def _build_filter(self, field_name: str, target_value: str) -> str:
        safe_value = escape_milvus_string(target_value)
        return f'{field_name} == "{safe_value}"'

    def _collect_image_object_names(self, chunk_rows: List[Dict[str, Any]]) -> Set[str]:
        object_names: Set[str] = set()
        for row in chunk_rows:
            for url in self._extract_urls_from_chunk(row):
                object_name = self._minio_object_from_url(url)
                if object_name:
                    object_names.add(object_name)
        return object_names

    def _collect_raw_file_objects(
        self, item_name: str, chunk_rows: List[Dict[str, Any]]
    ) -> Set[str]:
        object_names: Set[str] = set()
        minio_client = get_minio_client()
        pdf_prefix = (os.getenv("MINIO_PDF_DIR", "pdf_files") or "pdf_files").strip("/")
        if not minio_client or not minio_config.bucket_name:
            return object_names

        file_titles = {
            str(row.get("file_title", "")).strip()
            for row in chunk_rows
            if str(row.get("file_title", "")).strip()
        }
        file_titles.add(item_name)

        try:
            for obj in minio_client.list_objects(
                bucket_name=minio_config.bucket_name,
                prefix=f"{pdf_prefix}/",
                recursive=True,
            ):
                basename = Path(obj.object_name).name
                stem = Path(basename).stem
                if stem in file_titles:
                    object_names.add(obj.object_name)
        except Exception as exc:
            logger.warning(f"扫描 MinIO 原始文件失败: {exc}")

        return object_names

    def _collect_local_output_paths(self, chunk_rows: List[Dict[str, Any]]) -> Set[Path]:
        candidates: Set[Path] = set()
        if not LOCAL_OUTPUT_ROOT.exists():
            return candidates

        file_titles = {
            str(row.get("file_title", "")).strip()
            for row in chunk_rows
            if str(row.get("file_title", "")).strip()
        }
        if not file_titles:
            return candidates

        for path in LOCAL_OUTPUT_ROOT.rglob("*"):
            if not self._is_within_output_root(path):
                continue

            if path.is_dir() and path.name in file_titles:
                candidates.add(path)
                continue

            if path.is_file():
                if path.stem in file_titles:
                    candidates.add(path)
                    # 当前任务目录通常是 output/YYYYMMDD/<task_id>，命中原始文件时顺带删除整目录
                    task_dir = path.parent
                    if (
                        task_dir.is_dir()
                        and task_dir.parent.parent == LOCAL_OUTPUT_ROOT
                        and (task_dir / "chunks.json").exists()
                    ):
                        candidates.add(task_dir)
                elif any(path.name == f"{title}_result.zip" for title in file_titles):
                    candidates.add(path)

        return self._collapse_child_paths(candidates)

    def _extract_urls_from_chunk(self, row: Dict[str, Any]) -> Set[str]:
        urls: Set[str] = set()

        content = str(row.get("content", "") or "")
        for part in content.split("]("):
            if ")" not in part:
                continue
            candidate = part.split(")", 1)[0].strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                urls.add(candidate)

        figures = row.get("figures")
        if isinstance(figures, str) and figures.strip():
            try:
                figures = json.loads(figures)
            except Exception:
                figures = []
        if isinstance(figures, list):
            for fig in figures:
                if isinstance(fig, dict):
                    image_url = str(fig.get("image_url", "")).strip()
                    if image_url.startswith("http://") or image_url.startswith("https://"):
                        urls.add(image_url)

        return urls

    def _minio_object_from_url(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
        except Exception:
            return None

        path = unquote(parsed.path or "").lstrip("/")
        if not path:
            return None

        bucket_name = (minio_config.bucket_name or "").strip("/")
        if bucket_name and path.startswith(bucket_name + "/"):
            return path[len(bucket_name) + 1 :]
        return path

    def _delete_minio_objects(self, object_names: Iterable[str]) -> List[str]:
        minio_client = get_minio_client()
        if not minio_client or not minio_config.bucket_name:
            logger.warning("MinIO 客户端或 bucket 未配置，跳过对象删除")
            return []

        deleted: List[str] = []
        for object_name in object_names:
            try:
                minio_client.remove_object(minio_config.bucket_name, object_name)
                deleted.append(object_name)
            except Exception as exc:
                logger.warning(f"删除 MinIO 对象失败: object={object_name}, error={exc}")
        return deleted

    def _delete_local_paths(self, paths: List[Path]) -> List[str]:
        deleted: List[str] = []
        for path in self._collapse_child_paths(set(paths)):
            if not self._is_within_output_root(path):
                logger.warning(f"跳过 output 根目录之外的路径删除: {path}")
                continue
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=False)
                else:
                    path.unlink()
                deleted.append(str(path))
            except Exception as exc:
                logger.warning(f"删除本地路径失败: path={path}, error={exc}")
        return deleted

    def _collapse_child_paths(self, paths: Set[Path]) -> Set[Path]:
        collapsed: Set[Path] = set()
        for path in sorted(paths, key=lambda p: len(p.parts)):
            if any(parent in collapsed for parent in path.parents):
                continue
            collapsed.add(path)
        return collapsed

    def _is_within_output_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(LOCAL_OUTPUT_ROOT.resolve())
            return True
        except Exception:
            return False


delete_service = KnowledgeDeletePreviewService()

app = FastAPI(
    title="Knowledge Delete Preview Service",
    description="独立的知识库删除实现预览，不接入主项目路由。",
)


@app.get("/health")
async def health():
    return {"ok": True, "service": "kb_delete_preview"}


@app.post("/kb/delete-preview")
async def delete_preview(payload: DeleteKnowledgeRequest):
    try:
        return delete_service.delete_by_item_name(
            item_name=payload.item_name,
            mode=payload.mode,
            dry_run=payload.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(f"知识库删除预览服务异常: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
