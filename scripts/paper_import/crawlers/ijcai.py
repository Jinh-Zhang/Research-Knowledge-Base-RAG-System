"""IJCAI proceedings search."""

import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import IJCAI_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, split_authors_text, strip_html, title_matches_query
from . import register


@register("ijcai")
def search_ijcai_year(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    proceedings_url = f"{IJCAI_BASE_URL}/proceedings/{year}/"
    logger.info(
        f"[crawl] IJCAI search start: year={year} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(proceedings_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    pattern = re.compile(
        r'<div id="paper\d+" class="paper_wrapper">\s*'
        r'<div class="title">(?P<title>.*?)</div>\s*'
        r'<div class="authors">(?P<authors>.*?)</div>\s*'
        r'<div class="details">\s*\(<a href="(?P<pdf>[^"]+\.pdf)">PDF</a>\s*\|\s*<a href="(?P<detail>[^"]+)"',
        re.I | re.S,
    )

    papers: List[Dict[str, Any]] = []
    seen = set()
    for match in pattern.finditer(html):
        title = strip_html(match.group("title"))
        if not title or not title_matches_query(title, query):
            continue
        pdf_href = normalize_space(match.group("pdf"))
        if pdf_href in seen:
            continue
        seen.add(pdf_href)
        authors_text = strip_html(match.group("authors"))
        detail_href = normalize_space(match.group("detail"))

        papers.append(
            build_paper_record(
                "ijcai",
                Path(pdf_href).stem,
                title,
                published=str(year),
                authors=split_authors_text(authors_text),
                pdf_url=urljoin(proceedings_url, pdf_href),
                detail_url=urljoin(IJCAI_BASE_URL, detail_href),
                venue="IJCAI",
                year=str(year),
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] IJCAI search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
