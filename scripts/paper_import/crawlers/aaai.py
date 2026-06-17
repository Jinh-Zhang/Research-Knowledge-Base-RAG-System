"""AAAI search via the OJS archive.

The flow is two-stage: resolve the issue page URLs for a given year from the
archive index, then parse each issue page for articles. Split into helpers so
each stage is independently testable.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests

from app.core.logger import logger

from ..config import AAAI_ARCHIVE_URL, AAAI_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, split_authors_text, strip_html, title_matches_query
from . import register

_ARTICLE_PATTERN = re.compile(
    r'<a[^>]+href="(?P<detail>[^"]*/article/view/\d+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)
_PDF_PATTERN = re.compile(
    r'<a[^>]+href="(?P<pdf>[^"]*/article/(?:view/\d+/pdf|download/\d+/[^"]+))"[^>]*>\s*(?:<span[^>]*>)?\s*PDF\b.*?</a>',
    re.I | re.S,
)
_AUTHORS_PATTERN = re.compile(
    r'<div[^>]+class="authors"[^>]*>(?P<authors>.*?)</div>',
    re.I | re.S,
)
_ISSUE_PATTERN = re.compile(
    r'href="(?P<href>[^"]*/issue/view/\d+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)


def fetch_aaai_issue_urls(session: requests.Session, year: int) -> List[str]:
    """Resolve the AAAI issue page URLs that belong to ``year``."""
    archive_response = session.get(AAAI_ARCHIVE_URL, timeout=REQUEST_TIMEOUT)
    archive_response.raise_for_status()
    archive_html = archive_response.text

    issue_links: List[str] = []
    seen_issue_links = set()
    year_tokens = {
        str(year),
        f"AAAI-{str(year)[-2:]}",
        f"AAAI {year}",
    }
    for match in _ISSUE_PATTERN.finditer(archive_html):
        href = normalize_space(match.group("href"))
        if not href:
            continue
        window = archive_html[match.start() : min(len(archive_html), match.start() + 2500)]
        title = strip_html(match.group("title"))
        haystack = f"{title}\n{strip_html(window)}"
        if not any(token.lower() in haystack.lower() for token in year_tokens):
            continue
        normalized_href = href if href.startswith("http") else urljoin(AAAI_BASE_URL, href)
        if normalized_href not in seen_issue_links:
            seen_issue_links.add(normalized_href)
            issue_links.append(normalized_href)

    if not issue_links:
        raise ValueError(f"Could not find AAAI issue page for year={year}")

    logger.info(
        f"[crawl] AAAI archive resolved: year={year} issue_candidates={len(issue_links)}"
    )
    return issue_links


def parse_aaai_issue_page(
    issue_html: str,
    query: str,
    year: int,
    seen: set,
) -> List[Dict[str, Any]]:
    """Parse one AAAI issue page into paper records, skipping ``seen`` details."""
    article_blocks = re.split(
        r'(?=<div[^>]+class="obj_article_summary"[^>]*>)',
        issue_html,
        flags=re.I,
    )
    if len(article_blocks) <= 1:
        article_blocks = re.split(
            r"(?=<h3\b)|(?=<h2\b)|(?=<article\b)",
            issue_html,
            flags=re.I,
        )

    papers: List[Dict[str, Any]] = []
    for block in article_blocks:
        article_match = _ARTICLE_PATTERN.search(block)
        pdf_match = _PDF_PATTERN.search(block)
        if not article_match or not pdf_match:
            continue

        title = strip_html(article_match.group("title"))
        if not title or not title_matches_query(title, query):
            continue

        detail_href = normalize_space(article_match.group("detail"))
        if detail_href in seen:
            continue
        seen.add(detail_href)

        authors_match = _AUTHORS_PATTERN.search(block)
        authors_text = strip_html(authors_match.group("authors") if authors_match else "")

        papers.append(
            build_paper_record(
                "aaai",
                detail_href.rsplit("/", 1)[-1],
                title,
                published=str(year),
                authors=split_authors_text(authors_text, pattern=r",|;|\n"),
                pdf_url=urljoin(AAAI_BASE_URL, normalize_space(pdf_match.group("pdf"))),
                detail_url=urljoin(AAAI_BASE_URL, detail_href),
                venue="AAAI",
                year=str(year),
            )
        )
    return papers


@register("aaai")
def search_aaai_year(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    logger.info(
        f"[crawl] AAAI search start: year={year} query={query!r} "
        f"start={start} max_results={max_results}"
    )

    issue_links = fetch_aaai_issue_urls(session, year)

    papers: List[Dict[str, Any]] = []
    seen: set = set()
    for issue_url in issue_links:
        issue_response = session.get(issue_url, timeout=REQUEST_TIMEOUT)
        issue_response.raise_for_status()
        papers.extend(parse_aaai_issue_page(issue_response.text, query, year, seen))

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] AAAI search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
