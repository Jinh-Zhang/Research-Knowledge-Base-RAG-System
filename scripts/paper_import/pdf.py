"""PDF download, validation, and URL/path helpers."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import fitz
import requests

from app.core.logger import logger

from .config import PDF_DOWNLOAD_VALIDATION_RETRIES, REQUEST_TIMEOUT
from .metadata import normalize_venue_token
from .text_utils import (
    extract_year_text,
    normalize_space,
    sanitize_file_component,
    sanitize_filename,
)


def guess_title_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name or "paper.pdf"
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("_", " ").replace("-", " ").strip() or "paper"


def ensure_pdf_suffix(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        return url
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"
    return url


def compute_url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def build_paper_dir_name(index: int, paper: Dict[str, Any]) -> str:
    source = sanitize_filename((paper.get("source") or "paper"), max_length=16)
    short_id = compute_url_hash(paper["pdf_url"])
    return f"{index:03d}_{source}_{short_id}"


def build_import_file_stem(paper: Dict[str, Any]) -> str:
    venue = normalize_venue_token(str(paper.get("venue", "")))
    year = extract_year_text(str(paper.get("year", "")) or str(paper.get("published", "")))
    title = normalize_space(str(paper.get("title", "")))

    parts = []
    if venue:
        parts.append(sanitize_file_component(venue, max_length=16))
    if year:
        parts.append(year)
    if title:
        parts.append(sanitize_file_component(title, max_length=28))

    if not parts:
        return "paper"
    return "_".join(parts)


def build_paper_paths(index: int, paper: Dict[str, Any], root_dir: Path) -> Dict[str, Path]:
    paper_dir = root_dir / build_paper_dir_name(index, paper)
    pdf_path = paper_dir / f"{build_import_file_stem(paper)}.pdf"
    metadata_md_path = paper_dir / f"{pdf_path.stem}.md"
    return {
        "paper_dir": paper_dir,
        "pdf_path": pdf_path,
        "metadata_md_path": metadata_md_path,
        "paper_metadata_path": paper_dir / "paper_metadata.txt",
        "chunks_path": paper_dir / "chunks.json",
    }


def validate_pdf_file(path: Path) -> Optional[str]:
    if not path.exists() or path.stat().st_size == 0:
        return "file is missing or empty"

    try:
        with fitz.open(path) as doc:
            if doc.page_count <= 0:
                return "page_count=0"
    except Exception as exc:
        return str(exc)

    return None


def download_pdf(
    session: requests.Session,
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
        cached_error = validate_pdf_file(output_path)
        if not cached_error:
            logger.info(f"Skip download, file already exists: {output_path}")
            return output_path
        logger.warning(
            f"Cached PDF is invalid and will be re-downloaded: path={output_path} "
            f"reason={cached_error}"
        )
        output_path.unlink(missing_ok=True)

    total_attempts = PDF_DOWNLOAD_VALIDATION_RETRIES + 1
    last_error = ""
    for attempt in range(1, total_attempts + 1):
        logger.info(
            f"[crawl] PDF download start: url={url} -> {output_path} "
            f"attempt={attempt}/{total_attempts}"
        )
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(
                    f"Download target does not look like PDF: content_type={content_type} url={url}"
                )

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

        if output_path.stat().st_size == 0:
            last_error = "downloaded file is empty"
        else:
            last_error = validate_pdf_file(output_path) or ""
            if not last_error:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"[crawl] PDF download finished: path={output_path} size_mb={file_size_mb:.2f}"
                )
                return output_path

        logger.warning(
            f"Downloaded PDF validation failed: path={output_path} reason={last_error}"
        )
        output_path.unlink(missing_ok=True)

    raise ValueError(f"Downloaded invalid PDF after {total_attempts} attempts: {last_error}")
