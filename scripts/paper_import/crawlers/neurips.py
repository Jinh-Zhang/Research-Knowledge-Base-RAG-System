"""NeurIPS proceedings search."""

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import NEURIPS_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, split_authors_text, strip_html, title_matches_query
from . import register


def build_neurips_pdf_url(abstract_href: str) -> str:
    match = re.search(
        r"/paper_files/paper/(?P<year>\d{4})/hash/(?P<hash>[A-Fa-f0-9]+)-Abstract-Conference\.html",
        abstract_href,
    )
    if not match:
        return urljoin(NEURIPS_BASE_URL, abstract_href)
    year = match.group("year")
    hash_value = match.group("hash")
    return (
        f"{NEURIPS_BASE_URL}/paper_files/paper/{year}/file/"
        f"{hash_value}-Paper-Conference.pdf"
    )


@register("neurips")
def search_neurips_year(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    proceedings_url = f"{NEURIPS_BASE_URL}/paper_files/paper/{year}"
    logger.info(
        f"[crawl] NeurIPS search start: year={year} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(proceedings_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    pattern = re.compile(
        r'<a[^>]+title="paper title"[^>]+href="(?P<href>/paper_files/paper/\d{4}/hash/[^"]+-Abstract-Conference\.html)">(?P<title>.*?)</a>\s*'
        r'<span class="paper-authors">(?P<authors>.*?)</span>',
        re.I | re.S,
    )

    papers: List[Dict[str, Any]] = []
    seen = set()
    for match in pattern.finditer(html):
        abstract_href = normalize_space(match.group("href"))
        title = strip_html(match.group("title"))
        authors_text = strip_html(match.group("authors"))
        if not title or not title_matches_query(title, query):
            continue
        if abstract_href in seen:
            continue
        seen.add(abstract_href)
        papers.append(
            build_paper_record(
                "neurips",
                abstract_href.rsplit("/", 1)[-1],
                title,
                published=str(year),
                authors=split_authors_text(authors_text),
                pdf_url=build_neurips_pdf_url(abstract_href),
                detail_url=urljoin(NEURIPS_BASE_URL, abstract_href),
                venue="NeurIPS",
                year=str(year),
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] NeurIPS search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
