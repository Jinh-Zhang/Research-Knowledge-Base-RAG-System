"""Paper-record construction and metadata enrichment."""

import re
from typing import Any, Dict, List, Optional

from .text_utils import extract_year_text, normalize_space


def extract_openreview_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def normalize_venue_token(value: str) -> str:
    text = normalize_space(value)
    if not text:
        return ""

    normalized = re.sub(r"[_\-]+", " ", text).lower()
    aliases = {
        "arxiv": "arXiv",
        "iclr": "ICLR",
        "icml": "ICML",
        "neurips": "NeurIPS",
        "nips": "NeurIPS",
        "cvpr": "CVPR",
        "iccv": "ICCV",
        "eccv": "ECCV",
        "wacv": "WACV",
        "acl": "ACL",
        "emnlp": "EMNLP",
        "naacl": "NAACL",
        "aaai": "AAAI",
        "ijcai": "IJCAI",
        "colm": "COLM",
    }
    for raw, canonical in aliases.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(raw)}(?![A-Za-z])", normalized, re.I):
            return canonical
    return text


def enrich_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(paper)
    venue = normalize_venue_token(str(enriched.get("venue", "")))
    year = extract_year_text(str(enriched.get("year", "")))
    if not year:
        year = extract_year_text(str(enriched.get("published", "")))
    if not year:
        year = extract_year_text(str(enriched.get("detail_url", "")))
    enriched["venue"] = venue
    enriched["year"] = year
    return enriched


def build_openreview_urls(note_id: str) -> Dict[str, str]:
    normalized_id = normalize_space(note_id)
    return {
        "pdf_url": f"https://openreview.net/pdf?id={normalized_id}",
        "detail_url": f"https://openreview.net/forum?id={normalized_id}",
    }


def build_paper_record(
    source: str,
    paper_id: str,
    title: str,
    *,
    summary: str = "",
    published: str = "",
    authors: Optional[List[str]] = None,
    pdf_url: str = "",
    detail_url: str = "",
    venue: str = "",
    year: str = "",
) -> Dict[str, Any]:
    return enrich_paper_metadata(
        {
            "source": source,
            "id": paper_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors or [],
            "pdf_url": pdf_url,
            "detail_url": detail_url,
            "venue": venue,
            "year": year,
        }
    )


def dedupe_papers(papers) -> List[Dict[str, Any]]:
    seen = set()
    unique_items = []
    for paper in papers:
        key = normalize_space(paper.get("pdf_url") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(paper)
    return unique_items
