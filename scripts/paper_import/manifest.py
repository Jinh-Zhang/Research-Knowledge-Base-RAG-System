"""Resume manifest: persistent batch state for download/import staging."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logger import logger

from .pdf import build_paper_paths, ensure_pdf_suffix, validate_pdf_file
from .text_utils import normalize_space


def utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_resume_run_key(args: argparse.Namespace) -> str:
    payload = {
        "source": args.source,
        "conference": args.conference,
        "year": args.year,
        "query": args.query,
        "max_results": args.max_results,
        "start": args.start,
        "sort_by": args.sort_by,
        "sort_order": args.sort_order,
        "url_file": args.url_file,
        "openreview_venueid": args.openreview_venueid,
        "acl_event": args.acl_event,
        "cvf_event": args.cvf_event,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_resume_paths(root_dir: Path, run_key: str) -> Dict[str, Path]:
    resume_dir = root_dir / ".batch_state"
    return {
        "resume_dir": resume_dir,
        "manifest_path": resume_dir / f"{run_key}.manifest.json",
        "failed_url_file": resume_dir / f"{run_key}.failed_urls.txt",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_resume_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load resume manifest, ignoring old state: {manifest_path} | {exc}")
        return {}


def save_resume_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def make_manifest_item(index: int, paper: Dict[str, Any], root_dir: Path) -> Dict[str, Any]:
    paths = build_paper_paths(index, paper, root_dir)
    pdf_url = ensure_pdf_suffix(paper["pdf_url"])
    return {
        "index": index,
        "pdf_url": pdf_url,
        "title": paper.get("title") or "",
        "source": paper.get("source") or "",
        "venue": paper.get("venue") or "",
        "year": paper.get("year") or "",
        "paper_dir": str(paths["paper_dir"]),
        "pdf_path": str(paths["pdf_path"]),
        "metadata_md_path": str(paths["metadata_md_path"]),
        "chunks_path": str(paths["chunks_path"]),
        "download_status": "pending",
        "import_status": "pending",
        "task_id": "",
        "paper_title": "",
        "error": "",
        "updated_at": "",
        "paper": {**paper, "pdf_url": pdf_url},
    }


def merge_manifest_items(
    existing_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    root_dir: Path,
) -> List[Dict[str, Any]]:
    existing_by_url = {
        normalize_space(str(item.get("pdf_url", ""))).lower(): item
        for item in (existing_items or [])
        if normalize_space(str(item.get("pdf_url", "")))
    }
    merged: List[Dict[str, Any]] = []
    for index, paper in enumerate(papers, start=1):
        item = make_manifest_item(index, paper, root_dir)
        existing = existing_by_url.get(normalize_space(item["pdf_url"]).lower())
        if existing:
            for key in (
                "download_status",
                "import_status",
                "task_id",
                "paper_title",
                "error",
                "updated_at",
            ):
                item[key] = existing.get(key, item[key])
        merged.append(item)
    return merged


def build_resume_manifest(
    args: argparse.Namespace,
    papers: List[Dict[str, Any]],
    root_dir: Path,
    existing_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = dict(existing_manifest or {})
    manifest["run_key"] = build_resume_run_key(args)
    manifest["created_at"] = manifest.get("created_at") or utc_now_text()
    manifest["updated_at"] = utc_now_text()
    manifest["args"] = {
        "source": args.source,
        "conference": args.conference,
        "year": args.year,
        "query": args.query,
        "max_results": args.max_results,
        "start": args.start,
    }
    manifest["items"] = merge_manifest_items(manifest.get("items", []), papers, root_dir)
    return manifest


def write_failed_url_file(failed_url_file: Path, items: List[Dict[str, Any]]) -> None:
    failed_lines = []
    for item in items:
        if item.get("import_status") == "failed" or item.get("download_status") == "failed":
            title = normalize_space(str(item.get("title", "")))
            pdf_url = normalize_space(str(item.get("pdf_url", "")))
            if pdf_url:
                failed_lines.append(f"{title}\t{pdf_url}" if title else pdf_url)
    failed_url_file.parent.mkdir(parents=True, exist_ok=True)
    failed_url_file.write_text("\n".join(failed_lines) + ("\n" if failed_lines else ""), encoding="utf-8")


def sync_manifest_item_from_disk(item: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = Path(item.get("pdf_path", ""))
    chunks_path = Path(item.get("chunks_path", ""))

    if pdf_path.exists() and pdf_path.stat().st_size > 0 and not validate_pdf_file(pdf_path):
        item["download_status"] = "completed"

    if chunks_path.exists():
        item["import_status"] = "completed"
        item["error"] = ""

    return item


def summarize_manifest(manifest: Dict[str, Any]) -> Dict[str, int]:
    items = manifest.get("items", [])
    return {
        "total": len(items),
        "downloaded": sum(1 for item in items if item.get("download_status") == "completed"),
        "imported": sum(1 for item in items if item.get("import_status") == "completed"),
        "failed": sum(
            1
            for item in items
            if item.get("import_status") == "failed" or item.get("download_status") == "failed"
        ),
        "pending_import": sum(
            1
            for item in items
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        ),
    }
