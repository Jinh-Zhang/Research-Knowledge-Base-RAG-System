"""CVF Open Access proceedings search (CVPR / ICCV / ECCV / WACV)."""

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import CVF_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, strip_html, title_matches_query
from . import register


@register("cvf")
def search_cvf_event(
    session: requests.Session,
    event: str,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    event_url = f"{CVF_BASE_URL}/{event}?day=all"
    logger.info(
        f"[crawl] CVF search start: event={event} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(event_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    block_pattern = re.compile(
        r'<dt class="ptitle">.*?<a href="(?P<detail>/content/[^"]+\.html)">(?P<title>.*?)</a>.*?</dt>.*?'
        r'<dd>.*?\[<a href="(?P<pdf>/content/[^"]+\.pdf)">pdf</a>\]',
        re.I | re.S,
    )

    papers: List[Dict[str, Any]] = []
    seen = set()
    for match in block_pattern.finditer(html):
        detail_href = normalize_space(match.group("detail"))
        pdf_href = normalize_space(match.group("pdf"))
        title = strip_html(match.group("title"))
        if not title or not title_matches_query(title, query):
            continue
        if pdf_href in seen:
            continue
        seen.add(pdf_href)
        year_match = re.search(r"(\d{4})", event)
        papers.append(
            build_paper_record(
                "cvf",
                pdf_href.rsplit("/", 1)[-1],
                title,
                published=year_match.group(1) if year_match else "",
                pdf_url=urljoin(CVF_BASE_URL, pdf_href),
                detail_url=urljoin(CVF_BASE_URL, detail_href),
                venue=event,
                year=year_match.group(1) if year_match else "",
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] CVF search finished: event={event} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
