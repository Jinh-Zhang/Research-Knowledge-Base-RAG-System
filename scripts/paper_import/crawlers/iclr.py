"""ICLR search via the virtual conference site."""

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import ICLR_BASE_URL, REQUEST_TIMEOUT
from ..metadata import enrich_paper_metadata
from ..text_utils import normalize_space, strip_html, title_matches_query
from . import register


@register("iclr_virtual")
def search_iclr_virtual(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    papers_url = f"{ICLR_BASE_URL}/virtual/{year}/papers.html"
    logger.info(
        f"[crawl] ICLR virtual search start: year={year} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(papers_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    list_pattern = re.compile(
        r'<li><a href="(?P<detail>/virtual/\d+/poster/\d+)">(?P<title>.*?)</a></li>',
        re.I | re.S,
    )

    candidates: List[Dict[str, str]] = []
    for match in list_pattern.finditer(html):
        title = strip_html(match.group("title"))
        if not title or not title_matches_query(title, query):
            continue
        candidates.append(
            {
                "title": title,
                "detail_url": urljoin(ICLR_BASE_URL, normalize_space(match.group("detail"))),
            }
        )

    logger.info(
        f"[crawl] ICLR virtual list parsed: year={year} "
        f"matched_candidates={len(candidates)}"
    )

    target_candidates = candidates[start : start + max_results]
    papers: List[Dict[str, Any]] = []

    for candidate in target_candidates:
        detail_url = candidate["detail_url"]
        detail_response = session.get(detail_url, timeout=REQUEST_TIMEOUT)
        detail_response.raise_for_status()
        detail_html = detail_response.text

        title_match = re.search(
            r'<h1 class="event-title">(?P<title>.*?)</h1>',
            detail_html,
            re.I | re.S,
        )
        authors_match = re.search(
            r'<div class="event-organizers">\s*(?P<authors>.*?)\s*</div>',
            detail_html,
            re.I | re.S,
        )
        abstract_match = re.search(
            r'<div class="abstract-text-inner">\s*(?P<abstract>.*?)\s*</div>',
            detail_html,
            re.I | re.S,
        )
        openreview_match = re.search(
            r'href="https://openreview\.net/forum\?id=(?P<id>[^"&]+)"',
            detail_html,
            re.I,
        )

        title = strip_html(title_match.group("title")) if title_match else candidate["title"]
        authors_text = strip_html(authors_match.group("authors")) if authors_match else ""
        abstract = strip_html(abstract_match.group("abstract")) if abstract_match else ""
        openreview_id = normalize_space(openreview_match.group("id")) if openreview_match else ""

        papers.append(
            enrich_paper_metadata(
                {
                    "source": "iclr_virtual",
                    "id": openreview_id or detail_url.rsplit("/", 1)[-1],
                    "title": title,
                    "summary": abstract,
                    "published": str(year),
                    "authors": [
                        normalize_space(author)
                        for author in re.split(r"[;,]| ?· ?", authors_text)
                        if normalize_space(author)
                    ],
                    "pdf_url": (
                        f"https://openreview.net/pdf?id={openreview_id}"
                        if openreview_id
                        else detail_url
                    ),
                    "detail_url": detail_url,
                    "venue": "ICLR",
                    "year": str(year),
                }
            )
        )

    logger.info(
        f"[crawl] ICLR virtual search finished: year={year} "
        f"matched_candidates={len(candidates)} returned={len(papers)}"
    )
    return papers
