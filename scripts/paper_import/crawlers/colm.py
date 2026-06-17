"""COLM search via the official accepted-papers page."""

import re
from typing import Any, Dict, List

import requests

from app.core.logger import logger

from ..config import COLM_BASE_URL, REQUEST_TIMEOUT
from ..metadata import enrich_paper_metadata
from ..text_utils import normalize_space, strip_html, title_matches_query
from . import register


@register("colm_official")
def search_colm_official(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    accepted_url = f"{COLM_BASE_URL}/{year}/AcceptedPapers.html"
    logger.info(
        f"[crawl] COLM official search start: year={year} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(accepted_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    pattern = re.compile(
        r'<p>\s*(?:<span class="badge[^"]*">.*?</span>\s*)?'
        r'<a href="https://openreview\.net/forum\?id=(?P<id>[^"&]+)">(?P<title>.*?)</a>\s*<br/>\s*'
        r'<em>(?P<authors>.*?)</em>\s*</p>',
        re.I | re.S,
    )

    papers: List[Dict[str, Any]] = []
    for match in pattern.finditer(html):
        title = strip_html(match.group("title"))
        authors_text = strip_html(match.group("authors"))
        if not title or not title_matches_query(title, query):
            continue

        openreview_id = normalize_space(match.group("id"))
        papers.append(
            enrich_paper_metadata(
                {
                    "source": "colm_official",
                    "id": openreview_id,
                    "title": title,
                    "summary": "",
                    "published": str(year),
                    "authors": [
                        normalize_space(author)
                        for author in authors_text.split(",")
                        if normalize_space(author)
                    ],
                    "pdf_url": f"https://openreview.net/pdf?id={openreview_id}",
                    "detail_url": f"https://openreview.net/forum?id={openreview_id}",
                    "venue": "COLM",
                    "year": str(year),
                }
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] COLM official search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
