"""Import pipeline: run the import graph, stage downloads, loop import batches."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.task_utils import clear_task, update_task_status

from .config import DEFAULT_IMPORT_RETRIES
from .import_errors import (
    is_retryable_import_error,
    should_fallback_to_metadata_import,
)
from .manifest import (
    save_resume_manifest,
    sync_manifest_item_from_disk,
    utc_now_text,
    write_failed_url_file,
)
from .metadata_writer import dump_metadata, write_metadata_fallback_md
from .pdf import download_pdf


def run_import_graph(task_id: str, local_dir: Path, local_file_path: Path) -> Dict[str, Any]:
    update_task_status(task_id, "processing")
    state = get_default_state()
    state["task_id"] = task_id
    state["local_dir"] = str(local_dir)
    state["local_file_path"] = str(local_file_path)

    final_state: Optional[Dict[str, Any]] = None
    try:
        for event in kb_import_app.stream(state):
            for node_name, node_state in event.items():
                logger.info(f"[{task_id}] import node completed: {node_name}")
                final_state = node_state
        update_task_status(task_id, "completed")
        return final_state or state
    except Exception:
        update_task_status(task_id, "failed")
        raise
    finally:
        clear_task(task_id)


def execute_import_pipeline(
    paper: Dict[str, Any],
    paper_dir: Path,
    pdf_path: Path,
    *,
    sleep_seconds: float = 0.0,
    max_retries: int = DEFAULT_IMPORT_RETRIES,
) -> Dict[str, Any]:
    final_state: Optional[Dict[str, Any]] = None
    last_task_id = ""
    for attempt in range(1, max_retries + 2):
        last_task_id = f"batch_import_{uuid.uuid4().hex[:12]}"
        try:
            logger.info(
                f"[{last_task_id}] import attempt {attempt}/{max_retries + 1}: "
                f"{paper.get('title')}"
            )
            final_state = run_import_graph(last_task_id, paper_dir, pdf_path)
            break
        except Exception as exc:
            if should_fallback_to_metadata_import(exc):
                fallback_md_path = write_metadata_fallback_md(paper_dir, paper, pdf_path)
                fallback_task_id = f"batch_import_{uuid.uuid4().hex[:12]}"
                logger.warning(
                    f"[{fallback_task_id}] switching to metadata-only import fallback: "
                    f"{paper.get('title')}"
                )
                final_state = run_import_graph(fallback_task_id, paper_dir, fallback_md_path)
                last_task_id = fallback_task_id
                break
            if attempt > max_retries or not is_retryable_import_error(exc):
                raise
            logger.warning(
                f"[{last_task_id}] retryable import failure on attempt {attempt}/"
                f"{max_retries + 1}: {exc}. Retrying..."
            )
            time.sleep(3)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return {
        "task_id": last_task_id,
        "title": paper.get("title") or pdf_path.stem,
        "pdf_url": paper.get("pdf_url") or "",
        "paper_dir": str(paper_dir),
        "pdf_path": str(pdf_path),
        "paper_title": (final_state or {}).get("paper_title", ""),
        "status": "completed",
    }


def stage_downloads(
    session: requests.Session,
    manifest: Dict[str, Any],
    *,
    overwrite_pdf: bool,
    manifest_path: Path,
    failed_url_file: Path,
) -> None:
    for item in manifest.get("items", []):
        sync_manifest_item_from_disk(item)
        paper = item["paper"]
        paper_dir = Path(item["paper_dir"])
        pdf_path = Path(item["pdf_path"])
        try:
            dump_metadata(paper_dir, paper)
            download_pdf(session, item["pdf_url"], pdf_path, overwrite=overwrite_pdf)
            item["download_status"] = "completed"
            item["error"] = ""
        except Exception as exc:
            item["download_status"] = "failed"
            item["error"] = str(exc)
            item["updated_at"] = utc_now_text()
            manifest["updated_at"] = utc_now_text()
            save_resume_manifest(manifest_path, manifest)
            write_failed_url_file(failed_url_file, manifest.get("items", []))
            logger.error(f"Download failed: {paper.get('title')} | {exc}", exc_info=True)
            continue

        item["updated_at"] = utc_now_text()
        manifest["updated_at"] = utc_now_text()
        save_resume_manifest(manifest_path, manifest)

    write_failed_url_file(failed_url_file, manifest.get("items", []))


def collect_import_candidates(
    manifest: Dict[str, Any],
    *,
    force_reimport: bool,
) -> Dict[str, Any]:
    sync_count = 0
    candidates: List[Dict[str, Any]] = []

    for item in manifest.get("items", []):
        sync_manifest_item_from_disk(item)
        sync_count += 1
        if item.get("download_status") != "completed":
            continue
        if item.get("import_status") == "completed" and not force_reimport:
            continue
        candidates.append(item)

    return {
        "synced": sync_count,
        "candidates": candidates,
    }


def format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    completed = max(0, min(completed, total))
    filled = int(width * completed / total)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def log_import_progress(
    *,
    completed: int,
    success_count: int,
    failed_count: int,
    total: int,
    title: str,
) -> None:
    progress_bar = format_progress_bar(completed, total)
    logger.info(
        f"Import progress {progress_bar} {completed}/{total} "
        f"success={success_count} failed={failed_count} current={title}"
    )


def import_staged_papers(
    manifest: Dict[str, Any],
    *,
    import_batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    force_reimport: bool,
    manifest_path: Path,
    failed_url_file: Path,
) -> Dict[str, int]:
    imported_count = 0
    failed_count = 0
    candidate_result = collect_import_candidates(
        manifest,
        force_reimport=force_reimport,
    )
    sync_count = int(candidate_result["synced"])
    candidates = list(candidate_result["candidates"])
    total_candidates = len(candidates)
    total_completed = 0
    if total_candidates == 0:
        return {
            "synced": sync_count,
            "imported": 0,
            "failed": 0,
            "remaining": 0,
            "batches": 0,
        }

    effective_batch_size = import_batch_size or total_candidates
    batch_count = 0

    for batch_start in range(0, total_candidates, effective_batch_size):
        batch_count += 1
        batch_items = candidates[batch_start : batch_start + effective_batch_size]
        logger.info(
            f"Import batch {batch_count}: size={len(batch_items)} "
            f"processed={batch_start}/{total_candidates}"
        )

        for item in batch_items:
            paper = item["paper"]
            paper_dir = Path(item["paper_dir"])
            pdf_path = Path(item["pdf_path"])
            try:
                result = execute_import_pipeline(
                    paper,
                    paper_dir,
                    pdf_path,
                    sleep_seconds=sleep_seconds,
                    max_retries=max_retries,
                )
                item["import_status"] = "completed"
                item["task_id"] = result["task_id"]
                item["paper_title"] = result.get("paper_title", "")
                item["error"] = ""
                imported_count += 1
            except Exception as exc:
                item["import_status"] = "failed"
                item["error"] = str(exc)
                failed_count += 1
                logger.error(f"Import failed: {paper.get('title')} | {exc}", exc_info=True)

            total_completed += 1
            log_import_progress(
                completed=total_completed,
                success_count=imported_count,
                failed_count=failed_count,
                total=total_candidates,
                title=paper.get("title") or pdf_path.stem,
            )
            item["updated_at"] = utc_now_text()
            manifest["updated_at"] = utc_now_text()
            save_resume_manifest(manifest_path, manifest)
            write_failed_url_file(failed_url_file, manifest.get("items", []))

        remaining_after_batch = sum(
            1
            for item in manifest.get("items", [])
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        )
        logger.info(
            f"Import batch {batch_count} finished: imported={imported_count} "
            f"failed={failed_count} remaining={remaining_after_batch}"
        )

    return {
        "synced": sync_count,
        "imported": imported_count,
        "failed": failed_count,
        "remaining": sum(
            1
            for item in manifest.get("items", [])
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        ),
        "batches": batch_count,
    }


def import_one_paper(
    session: requests.Session,
    paper: Dict[str, Any],
    index: int,
    root_dir: Path,
    overwrite_pdf: bool = False,
    sleep_seconds: float = 0.0,
    max_retries: int = DEFAULT_IMPORT_RETRIES,
) -> Dict[str, Any]:
    from .pdf import build_paper_paths, ensure_pdf_suffix

    pdf_url = ensure_pdf_suffix(paper["pdf_url"])
    paper = {**paper, "pdf_url": pdf_url}

    paths = build_paper_paths(index, paper, root_dir)
    paper_dir = paths["paper_dir"]
    pdf_path = paths["pdf_path"]
    logger.info(f"start import: {paper.get('title')}")
    logger.info(f"pdf_url: {pdf_url}")

    dump_metadata(paper_dir, paper)
    download_pdf(session, pdf_url, pdf_path, overwrite=overwrite_pdf)
    return execute_import_pipeline(
        paper,
        paper_dir,
        pdf_path,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )
