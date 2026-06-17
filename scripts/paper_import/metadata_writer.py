"""Write paper metadata to disk: sidecar txt and fallback markdown."""

from pathlib import Path
from typing import Any, Dict

from app.core.logger import logger

from .text_utils import extract_year_text, normalize_space
from .metadata import normalize_venue_token


def dump_metadata(local_dir: Path, paper: Dict[str, Any]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = local_dir / "paper_metadata.txt"
    lines = [
        f"title: {paper.get('title', '')}",
        f"source: {paper.get('source', '')}",
        f"venue: {paper.get('venue', '')}",
        f"id: {paper.get('id', '')}",
        f"published: {paper.get('published', '')}",
        f"authors: {', '.join(paper.get('authors') or [])}",
        f"pdf_url: {paper.get('pdf_url', '')}",
        f"detail_url: {paper.get('detail_url', '')}",
        "",
        paper.get("summary", "") or "",
    ]
    metadata_path.write_text("\n".join(lines), encoding="utf-8")


def build_metadata_markdown(paper: Dict[str, Any], pdf_path: Path) -> str:
    authors = paper.get("authors") or []
    authors_text = ", ".join(str(author) for author in authors if str(author).strip()) or "Unknown"
    title = normalize_space(str(paper.get("title", ""))) or pdf_path.stem
    venue = normalize_venue_token(str(paper.get("venue", "")))
    year = extract_year_text(str(paper.get("year", "")) or str(paper.get("published", "")))
    published = normalize_space(str(paper.get("published", "")))
    summary = normalize_space(str(paper.get("summary", "")))
    detail_url = str(paper.get("detail_url", "")).strip()
    pdf_url = str(paper.get("pdf_url", "")).strip()
    source = str(paper.get("source", "")).strip()

    lines = [f"# {title}", ""]
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Title: {title}")
    lines.append(f"- Authors: {authors_text}")
    if venue:
        lines.append(f"- Venue: {venue}")
    if year:
        lines.append(f"- Year: {year}")
    if published:
        lines.append(f"- Published: {published}")
    if source:
        lines.append(f"- Source: {source}")
    if detail_url:
        lines.append(f"- Detail URL: {detail_url}")
    if pdf_url:
        lines.append(f"- PDF URL: {pdf_url}")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(summary or "Abstract not available from the source listing.")
    lines.append("")
    lines.append("## Import Note")
    lines.append("")
    lines.append(
        "This markdown was generated from crawler metadata because PDF parsing did not produce usable text."
    )
    lines.append("")
    return "\n".join(lines)


def write_metadata_fallback_md(paper_dir: Path, paper: Dict[str, Any], pdf_path: Path) -> Path:
    md_path = paper_dir / f"{pdf_path.stem}.md"
    md_content = build_metadata_markdown(paper, pdf_path)
    md_path.write_text(md_content, encoding="utf-8")
    logger.warning(f"[fallback] metadata markdown created: {md_path}")
    return md_path
