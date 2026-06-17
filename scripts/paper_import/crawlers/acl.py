"""ACL Anthology event search."""

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import ACL_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, strip_html, title_matches_query
from . import register


@register("acl")
def search_acl_event(
    session: requests.Session,
    event: str,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    event_url = f"{ACL_BASE_URL}/events/{event}/"
    logger.info(
        f"[crawl] ACL search start: event={event} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(event_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    event_tag = event.split("-", 1)[0].lower()
    seen = set()
    papers: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"<a[^>]+href=(?:\"(?P<quoted>/[^\"]+/)\"|(?P<bare>/[^ >]+))[^>]*>(?P<title>.*?)</a>",
        re.I | re.S,
    )

    for match in pattern.finditer(html):
        href = normalize_space(match.group("quoted") or match.group("bare") or "")
        raw_title = match.group("title")
        title = strip_html(raw_title)
        if not href.startswith("/") or not title:
            continue
        if not re.match(r"^/\d{4}\.[^/]+/$", href):
            continue
        if event_tag not in href.lower():
            continue
        if len(title) < 8:
            continue
        if not title_matches_query(title, query):
            continue
        if href in seen:
            continue
        seen.add(href)

        article_url = urljoin(ACL_BASE_URL, href)
        pdf_url = urljoin(ACL_BASE_URL, href.rstrip("/") + ".pdf")
        year_match = re.search(r"-(\d{4})$", event)
        papers.append(
            build_paper_record(
                "acl",
                href.strip("/"),
                title,
                published=year_match.group(1) if year_match else "",
                pdf_url=pdf_url,
                detail_url=article_url,
                venue=event,
                year=year_match.group(1) if year_match else "",
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] ACL search finished: event={event} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
