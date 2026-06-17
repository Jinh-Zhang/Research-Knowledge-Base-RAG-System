"""ICML search via PMLR proceedings, with dynamic volume discovery."""

import re
from typing import Any, Dict, List

import requests

from app.core.logger import logger

from ..config import ICML_YEAR_TO_VOLUME, PMLR_BASE_URL, REQUEST_TIMEOUT
from ..metadata import build_paper_record
from ..text_utils import normalize_space, split_authors_text, strip_html, title_matches_query
from . import register


def is_icml_volume_title(title: str, year: int) -> bool:
    normalized = normalize_space(title)
    if str(year) not in normalized:
        return False

    patterns = [
        r"\bProceedings of ICML\b",
        r"\bProceedings of the \d+(?:st|nd|rd|th) International Conference on Machine Learning\b",
        r"\bInternational Conference on Machine Learning\b",
        r"\bICML\b",
    ]
    return any(re.search(pattern, normalized, re.I) for pattern in patterns)


def discover_icml_volume(session: requests.Session, year: int) -> int:
    cached_volume = ICML_YEAR_TO_VOLUME.get(year)
    if cached_volume:
        logger.info(f"[crawl] ICML volume cache hit: year={year} volume={cached_volume}")
        return cached_volume

    logger.info(f"[crawl] ICML volume discovery start: year={year}")
    response = session.get(PMLR_BASE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    volume_pattern = re.compile(
        r'<a[^>]+href="(?P<href>/v(?P<volume>\d+)/?)"[^>]*>\s*Volume\s+(?P=volume)\s*</a>\s*(?P<title>.*?)</li>',
        re.I | re.S,
    )

    matches: List[int] = []
    for match in volume_pattern.finditer(html):
        title = strip_html(match.group("title"))
        if not is_icml_volume_title(title, year):
            continue
        matches.append(int(match.group("volume")))

    if not matches:
        raise ValueError(
            f"Could not dynamically discover ICML volume for year={year} from {PMLR_BASE_URL}"
        )

    volume = max(matches)
    ICML_YEAR_TO_VOLUME[year] = volume
    logger.info(f"[crawl] ICML volume discovery finished: year={year} volume={volume}")
    return volume


@register("icml")
def search_icml_year(
    session: requests.Session,
    year: int,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    volume = discover_icml_volume(session, year)

    proceedings_url = f"{PMLR_BASE_URL}/v{volume}/"
    logger.info(
        f"[crawl] ICML search start: year={year} volume={volume} query={query!r} "
        f"start={start} max_results={max_results}"
    )
    response = session.get(proceedings_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    html = response.text

    block_pattern = re.compile(r'<div class="paper">(.*?)</div>', re.I | re.S)
    title_pattern = re.compile(r'<p class="title">(.*?)</p>', re.I | re.S)
    authors_pattern = re.compile(
        r'<span class="authors">(.*?)</span>', re.I | re.S
    )
    pdf_pattern = re.compile(
        r'<a href="(?P<pdf>[^"]+)"[^>]*>\s*Download PDF\s*</a>', re.I | re.S
    )
    abs_pattern = re.compile(r'<a href="(?P<abs>[^"]+)"[^>]*>\s*abs\s*</a>', re.I | re.S)

    papers: List[Dict[str, Any]] = []
    seen = set()
    for block_match in block_pattern.finditer(html):
        block = block_match.group(1)
        title_match = title_pattern.search(block)
        pdf_match = pdf_pattern.search(block)
        if not title_match or not pdf_match:
            continue

        title = strip_html(title_match.group(1))
        if not title or not title_matches_query(title, query):
            continue

        pdf_url = normalize_space(pdf_match.group("pdf"))
        if pdf_url in seen:
            continue
        seen.add(pdf_url)

        authors_match = authors_pattern.search(block)
        authors_text = strip_html(authors_match.group(1) if authors_match else "")
        abs_match = abs_pattern.search(block)

        papers.append(
            build_paper_record(
                "icml",
                pdf_url.rsplit("/", 1)[-1],
                title,
                published=str(year),
                authors=split_authors_text(authors_text.replace("\xa0", " ")),
                pdf_url=pdf_url,
                detail_url=normalize_space(abs_match.group("abs")) if abs_match else proceedings_url,
                venue="ICML",
                year=str(year),
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] ICML search finished: year={year} volume={volume} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers
