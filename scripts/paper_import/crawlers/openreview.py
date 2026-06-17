"""OpenReview search via the api2 notes endpoint."""

import re
from typing import Any, Dict, List, Optional

import requests

from app.core.logger import logger

from ..config import OPENREVIEW_NOTES_URL, REQUEST_TIMEOUT
from ..metadata import (
    build_openreview_urls,
    build_paper_record,
    extract_openreview_value,
)
from ..text_utils import normalize_space, text_matches_query
from . import register


@register("openreview")
def search_openreview(
    session: requests.Session,
    venueid: str,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    batch_size = min(max(max_results * 10, 100), 1000)
    target_count = start + max_results
    offset = 0
    total_notes = 0
    total_available: Optional[int] = None
    papers: List[Dict[str, Any]] = []
    logger.info(
        f"[crawl] OpenReview search start: venueid={venueid} query={query!r} "
        f"start={start} max_results={max_results} batch_size={batch_size}"
    )

    while True:
        params = {
            "content.venueid": venueid,
            "limit": batch_size,
            "offset": offset,
            "count": "true",
        }
        response = session.get(OPENREVIEW_NOTES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        notes = payload.get("notes", [])
        total_available = payload.get("count", total_available)
        total_notes += len(notes)

        if not notes:
            break

        for note in notes:
            content = note.get("content") or {}
            title = normalize_space(extract_openreview_value(content.get("title")) or "")
            abstract = normalize_space(
                extract_openreview_value(content.get("abstract")) or ""
            )
            if not title:
                continue

            searchable_text = f"{title} {abstract}".strip()
            if not text_matches_query(searchable_text, query):
                continue

            authors = extract_openreview_value(content.get("authors")) or []
            if not isinstance(authors, list):
                authors = [str(authors)]

            note_id = note.get("id") or note.get("forum") or ""
            published = str(note.get("tcdate") or "")
            year_match = re.search(r"/(\d{4})/", venueid)
            year = year_match.group(1) if year_match else ""
            papers.append(
                build_paper_record(
                    "openreview",
                    note_id,
                    title,
                    summary=abstract,
                    published=year or published,
                    authors=[
                        normalize_space(str(x)) for x in authors if str(x).strip()
                    ],
                    venue=venueid,
                    year=year,
                    **build_openreview_urls(note_id),
                )
            )

        if len(papers) >= target_count:
            break
        if len(notes) < batch_size:
            break
        if total_available is not None and offset + len(notes) >= total_available:
            break

        offset += len(notes)

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] OpenReview search finished: fetched_notes={total_notes} "
        f"total_available={total_available} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
