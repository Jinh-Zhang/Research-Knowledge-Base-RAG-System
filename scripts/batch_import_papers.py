import argparse
import hashlib
from http.client import RemoteDisconnected
import json
import re
import sys
import time
import uuid
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import fitz
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.task_utils import clear_task, update_task_status


ARXIV_API_URL = "http://export.arxiv.org/api/query"
OPENREVIEW_NOTES_URL = "https://api2.openreview.net/notes"
ACL_BASE_URL = "https://aclanthology.org"
CVF_BASE_URL = "https://openaccess.thecvf.com"
NEURIPS_BASE_URL = "https://proceedings.neurips.cc"
PMLR_BASE_URL = "https://proceedings.mlr.press"
AAAI_BASE_URL = "https://ojs.aaai.org"
AAAI_ARCHIVE_URL = f"{AAAI_BASE_URL}/index.php/AAAI/issue/archive"
IJCAI_BASE_URL = "https://www.ijcai.org"
ICLR_BASE_URL = "https://iclr.cc"
COLM_BASE_URL = "https://colmweb.org"
DEFAULT_USER_AGENT = "knowledge-base-paper-importer/1.0"
DEFAULT_IMPORT_ROOT = PROJECT_ROOT / "output" / "batch_imports"
REQUEST_TIMEOUT = 60
REQUEST_RETRY_TOTAL = 3
REQUEST_RETRY_BACKOFF = 1.0
SOURCE_SEARCH_RETRIES = 2
SOURCE_SEARCH_RETRY_SLEEP = 2.0
PDF_DOWNLOAD_VALIDATION_RETRIES = 1
DEFAULT_IMPORT_RETRIES = 2
DEFAULT_IMPORT_BATCH_SIZE = 5
CLI_SOURCE_CHOICES = [
    "all",
    "arxiv",
    "url_file",
    "openreview",
    "acl",
    "cvf",
    "neurips",
    "icml",
    "aaai",
    "ijcai",
]
DEFAULT_SOURCE_YEARS = {
    "neurips": 2024,
    "icml": 2024,
    "aaai": 2025,
    "ijcai": 2024,
}
DEFAULT_MULTI_SOURCE_ORDER = ["arxiv", "neurips", "icml", "aaai", "ijcai"]
YEAR_BASED_SOURCES = {"neurips", "icml", "aaai", "ijcai", "iclr_virtual", "colm_official"}
CONFERENCE_SOURCE_CHOICES = [
    "neurips",
    "icml",
    "aaai",
    "ijcai",
    "cvpr",
    "iccv",
    "eccv",
    "wacv",
    "acl",
    "emnlp",
    "naacl",
    "iclr",
    "colm",
]
ICML_YEAR_TO_VOLUME = {
    2024: 235,
    2025: 267,
}
DIRECT_SOURCE_REQUIRED_FIELDS = {
    "url_file": "url_file",
    "openreview": "openreview_venueid",
    "acl": "acl_event",
    "cvf": "cvf_event",
}
CONFERENCE_TARGET_RULES = {
    "NeurIPS": {"source": "neurips", "uses_year": True},
    "ICML": {"source": "icml", "uses_year": True},
    "AAAI": {"source": "aaai", "uses_year": True},
    "IJCAI": {"source": "ijcai", "uses_year": True},
    "CVPR": {"source": "cvf", "params": {"event": "CVPR{year}"}},
    "ICCV": {"source": "cvf", "params": {"event": "ICCV{year}"}},
    "ECCV": {"source": "cvf", "params": {"event": "ECCV{year}"}},
    "WACV": {"source": "cvf", "params": {"event": "WACV{year}"}},
    "ACL": {"source": "acl", "params": {"event": "acl-{year}"}},
    "EMNLP": {"source": "acl", "params": {"event": "emnlp-{year}"}},
    "NAACL": {"source": "acl", "params": {"event": "naacl-{year}"}},
    "ICLR": {"source": "iclr_virtual", "uses_year": True},
    "COLM": {"source": "colm_official", "uses_year": True},
}

def sanitize_filename(name: str, max_length: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = "paper"
    return cleaned[:max_length]


def sanitize_file_component(name: str, max_length: int = 32) -> str:
    cleaned = sanitize_filename(name, max_length=max_length)
    cleaned = cleaned.replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "paper"


def build_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    session.trust_env = False
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
        status=REQUEST_RETRY_TOTAL,
        backoff_factor=REQUEST_RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text or "", flags=re.S)
    return normalize_space(unescape(text))


def title_matches_query(title: str, query: str) -> bool:
    return text_matches_query(title, query)


def text_matches_query(text: str, query: str) -> bool:
    query = normalize_space(query).lower()
    text = normalize_space(text).lower()
    if not query:
        return True

    lowered = query
    for prefix in ("all:", "ti:", "abs:", "cat:", "au:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break

    lowered = lowered.strip().strip('"').strip("'")
    lowered = normalize_space(lowered)
    if not lowered:
        return True

    tokens = [token for token in re.split(r"\s+", lowered) if token]
    if not tokens:
        return True

    return lowered in text or all(token in text for token in tokens)


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


def extract_openreview_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def extract_year_text(text: str) -> str:
    match = re.search(r"(?<!\d)(20[0-4]\d)(?!\d)", text or "")
    return match.group(1) if match else ""


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


def split_authors_text(authors_text: str, pattern: str = r",") -> List[str]:
    return [
        normalize_space(author)
        for author in re.split(pattern, authors_text or "")
        if normalize_space(author)
    ]


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


def parse_arxiv_feed(xml_text: str) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entries: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ns):
        entry_id = normalize_space(
            entry.findtext("atom:id", default="", namespaces=ns) or ""
        )
        title = normalize_space(
            entry.findtext("atom:title", default="", namespaces=ns) or ""
        )
        summary = normalize_space(
            entry.findtext("atom:summary", default="", namespaces=ns) or ""
        )
        published = normalize_space(
            entry.findtext("atom:published", default="", namespaces=ns) or ""
        )

        authors = []
        for author in entry.findall("atom:author", ns):
            name = normalize_space(
                author.findtext("atom:name", default="", namespaces=ns) or ""
            )
            if name:
                authors.append(name)

        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            title_attr = normalize_space(link.attrib.get("title") or "").lower()
            type_attr = normalize_space(link.attrib.get("type") or "").lower()
            href = normalize_space(link.attrib.get("href") or "")
            if title_attr == "pdf" or type_attr == "application/pdf":
                pdf_url = href
                break

        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

        entries.append(
            enrich_paper_metadata(
                {
                "source": "arxiv",
                "id": entry_id,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors,
                "pdf_url": pdf_url,
                "detail_url": entry_id,
                "venue": "arXiv",
                }
            )
        )

    return entries


def search_arxiv(
    session: requests.Session,
    query: str,
    max_results: int,
    start: int = 0,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    logger.info(
        f"[crawl] arXiv search start: query={query!r} start={start} "
        f"max_results={max_results} sort_by={sort_by} sort_order={sort_order}"
    )
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    response = session.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    entries = parse_arxiv_feed(response.text)
    logger.info(f"[crawl] arXiv search finished: matched={len(entries)}")
    return entries


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

    archive_response = session.get(AAAI_ARCHIVE_URL, timeout=REQUEST_TIMEOUT)
    archive_response.raise_for_status()
    archive_html = archive_response.text

    issue_links = []
    seen_issue_links = set()
    issue_pattern = re.compile(
        r'href="(?P<href>[^"]*/issue/view/\d+)"[^>]*>(?P<title>.*?)</a>',
        re.I | re.S,
    )
    year_tokens = {
        str(year),
        f"AAAI-{str(year)[-2:]}",
        f"AAAI {year}",
    }
    for match in issue_pattern.finditer(archive_html):
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

    article_pattern = re.compile(
        r'<a[^>]+href="(?P<detail>[^"]*/article/view/\d+)"[^>]*>(?P<title>.*?)</a>',
        re.I | re.S,
    )
    pdf_pattern = re.compile(
        r'<a[^>]+href="(?P<pdf>[^"]*/article/(?:view/\d+/pdf|download/\d+/[^"]+))"[^>]*>\s*(?:<span[^>]*>)?\s*PDF\b.*?</a>',
        re.I | re.S,
    )
    authors_pattern = re.compile(
        r'<div[^>]+class="authors"[^>]*>(?P<authors>.*?)</div>',
        re.I | re.S,
    )

    papers: List[Dict[str, Any]] = []
    seen = set()
    for issue_url in issue_links:
        issue_response = session.get(issue_url, timeout=REQUEST_TIMEOUT)
        issue_response.raise_for_status()
        issue_html = issue_response.text

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
        for block in article_blocks:
            article_match = article_pattern.search(block)
            pdf_match = pdf_pattern.search(block)
            if not article_match or not pdf_match:
                continue

            title = strip_html(article_match.group("title"))
            if not title or not title_matches_query(title, query):
                continue

            detail_href = normalize_space(article_match.group("detail"))
            if detail_href in seen:
                continue
            seen.add(detail_href)

            authors_match = authors_pattern.search(block)
            authors_text = strip_html(authors_match.group("authors") if authors_match else "")

            papers.append(
                build_paper_record(
                    "aaai",
                    detail_href.rsplit("/", 1)[-1],
                    title,
                    published=str(year),
                    authors=split_authors_text(authors_text, pattern=r",|;|\n"),
                    pdf_url=urljoin(
                        AAAI_BASE_URL, normalize_space(pdf_match.group("pdf"))
                    ),
                    detail_url=urljoin(AAAI_BASE_URL, detail_href),
                    venue="AAAI",
                    year=str(year),
                )
            )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] AAAI search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers


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
                        for author in re.split(r"[;,]| ?\u00b7 ?", authors_text)
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


SOURCE_SEARCHERS = {
    "arxiv": search_arxiv,
    "openreview": search_openreview,
    "acl": search_acl_event,
    "cvf": search_cvf_event,
    "neurips": search_neurips_year,
    "icml": search_icml_year,
    "aaai": search_aaai_year,
    "ijcai": search_ijcai_year,
    "iclr_virtual": search_iclr_virtual,
    "colm_official": search_colm_official,
}

SOURCE_OPTION_FIELDS = {
    "openreview": ("venueid", "openreview_venueid"),
    "acl": ("event", "acl_event"),
    "cvf": ("event", "cvf_event"),
}


def run_named_search(
    session: requests.Session,
    source: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    search_fn = SOURCE_SEARCHERS.get(source)
    if not search_fn:
        raise ValueError(f"Unsupported searchable source: {source}")
    return dedupe_papers(search_fn(session=session, **kwargs))


def build_search_kwargs(
    source: str,
    args: argparse.Namespace,
    query: str,
    max_results: int,
    start: int,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "start": start,
    }

    if source == "arxiv":
        kwargs.update(sort_by=args.sort_by, sort_order=args.sort_order)
        return kwargs

    option_fields = SOURCE_OPTION_FIELDS.get(source)
    if option_fields:
        param_name, arg_name = option_fields
        kwargs[param_name] = str(getattr(args, arg_name, "") or "").strip()
        return kwargs

    if source in YEAR_BASED_SOURCES:
        kwargs["year"] = resolve_source_year(args, source)

    return kwargs


def run_source_search(
    session: requests.Session,
    source: str,
    args: argparse.Namespace,
    query: str,
    max_results: int,
    start: int = 0,
) -> List[Dict[str, Any]]:
    return run_named_search(
        session=session,
        source=source,
        **build_search_kwargs(
            source=source,
            args=args,
            query=query,
            max_results=max_results,
            start=start,
        ),
    )


def is_retryable_collection_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.RequestException, RemoteDisconnected)):
        return True

    message = normalize_space(str(exc)).lower()
    retryable_fragments = [
        "connection aborted",
        "connection reset",
        "remote end closed connection without response",
        "read timed out",
        "timed out",
        "temporarily unavailable",
    ]
    return any(fragment in message for fragment in retryable_fragments)


def run_source_search_with_retries(
    session: requests.Session,
    source: str,
    args: argparse.Namespace,
    query: str,
    max_results: int,
    start: int = 0,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    total_attempts = SOURCE_SEARCH_RETRIES + 1
    last_exc: Optional[Exception] = None

    for attempt in range(1, total_attempts + 1):
        try:
            return run_source_search(
                session=session,
                source=source,
                args=args,
                query=query,
                max_results=max_results,
                start=start,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= total_attempts or not is_retryable_collection_error(exc):
                break
            logger.warning(
                f"[crawl] source search retry: source={source} "
                f"attempt={attempt}/{total_attempts} error={exc}"
            )
            time.sleep(SOURCE_SEARCH_RETRY_SLEEP)

    if strict and last_exc:
        raise last_exc

    if last_exc:
        logger.warning(
            f"[crawl] source failed and will be skipped: source={source} error={last_exc}"
        )
    return []


def load_pdf_url_file(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        title = ""
        url = line
        if "\t" in line:
            maybe_title, maybe_url = line.split("\t", 1)
            title = maybe_title.strip()
            url = maybe_url.strip()

        items.append(
            enrich_paper_metadata(
                {
                "source": "url_file",
                "id": url,
                "title": title or guess_title_from_url(url),
                "summary": "",
                "published": "",
                "authors": [],
                "pdf_url": url,
                "detail_url": url,
                "venue": "",
                "year": "",
                }
            )
        )
    return items


def guess_title_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name or "paper.pdf"
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("_", " ").replace("-", " ").strip() or "paper"


def ensure_pdf_suffix(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        return url
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"
    return url


def compute_url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_paper_dir_name(index: int, paper: Dict[str, Any]) -> str:
    source = sanitize_filename((paper.get("source") or "paper"), max_length=16)
    short_id = compute_url_hash(paper["pdf_url"])
    return f"{index:03d}_{source}_{short_id}"


def build_import_file_stem(paper: Dict[str, Any]) -> str:
    venue = normalize_venue_token(str(paper.get("venue", "")))
    year = extract_year_text(str(paper.get("year", "")) or str(paper.get("published", "")))
    title = normalize_space(str(paper.get("title", "")))

    parts: List[str] = []
    if venue:
        parts.append(sanitize_file_component(venue, max_length=16))
    if year:
        parts.append(year)
    if title:
        parts.append(sanitize_file_component(title, max_length=28))

    if not parts:
        return "paper"
    return "_".join(parts)


def build_paper_paths(index: int, paper: Dict[str, Any], root_dir: Path) -> Dict[str, Path]:
    paper_dir = root_dir / build_paper_dir_name(index, paper)
    pdf_path = paper_dir / f"{build_import_file_stem(paper)}.pdf"
    metadata_md_path = paper_dir / f"{pdf_path.stem}.md"
    return {
        "paper_dir": paper_dir,
        "pdf_path": pdf_path,
        "metadata_md_path": metadata_md_path,
        "paper_metadata_path": paper_dir / "paper_metadata.txt",
        "chunks_path": paper_dir / "chunks.json",
    }


def build_resume_run_key(args: argparse.Namespace) -> str:
    payload = {
        "source": args.source,
        "conference": args.conference,
        "year": args.year,
        "query": args.query,
        "max_results": args.max_results,
        "start": args.start,
        "sort_by": args.sort_by,
        "sort_order": args.sort_order,
        "url_file": args.url_file,
        "openreview_venueid": args.openreview_venueid,
        "acl_event": args.acl_event,
        "cvf_event": args.cvf_event,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_resume_paths(root_dir: Path, run_key: str) -> Dict[str, Path]:
    resume_dir = root_dir / ".batch_state"
    return {
        "resume_dir": resume_dir,
        "manifest_path": resume_dir / f"{run_key}.manifest.json",
        "failed_url_file": resume_dir / f"{run_key}.failed_urls.txt",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_resume_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load resume manifest, ignoring old state: {manifest_path} | {exc}")
        return {}


def save_resume_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def make_manifest_item(index: int, paper: Dict[str, Any], root_dir: Path) -> Dict[str, Any]:
    paths = build_paper_paths(index, paper, root_dir)
    pdf_url = ensure_pdf_suffix(paper["pdf_url"])
    return {
        "index": index,
        "pdf_url": pdf_url,
        "title": paper.get("title") or "",
        "source": paper.get("source") or "",
        "venue": paper.get("venue") or "",
        "year": paper.get("year") or "",
        "paper_dir": str(paths["paper_dir"]),
        "pdf_path": str(paths["pdf_path"]),
        "metadata_md_path": str(paths["metadata_md_path"]),
        "chunks_path": str(paths["chunks_path"]),
        "download_status": "pending",
        "import_status": "pending",
        "task_id": "",
        "paper_title": "",
        "error": "",
        "updated_at": "",
        "paper": {**paper, "pdf_url": pdf_url},
    }


def merge_manifest_items(
    existing_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    root_dir: Path,
) -> List[Dict[str, Any]]:
    existing_by_url = {
        normalize_space(str(item.get("pdf_url", ""))).lower(): item
        for item in (existing_items or [])
        if normalize_space(str(item.get("pdf_url", "")))
    }
    merged: List[Dict[str, Any]] = []
    for index, paper in enumerate(papers, start=1):
        item = make_manifest_item(index, paper, root_dir)
        existing = existing_by_url.get(normalize_space(item["pdf_url"]).lower())
        if existing:
            for key in (
                "download_status",
                "import_status",
                "task_id",
                "paper_title",
                "error",
                "updated_at",
            ):
                item[key] = existing.get(key, item[key])
        merged.append(item)
    return merged


def build_resume_manifest(
    args: argparse.Namespace,
    papers: List[Dict[str, Any]],
    root_dir: Path,
    existing_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = dict(existing_manifest or {})
    manifest["run_key"] = build_resume_run_key(args)
    manifest["created_at"] = manifest.get("created_at") or utc_now_text()
    manifest["updated_at"] = utc_now_text()
    manifest["args"] = {
        "source": args.source,
        "conference": args.conference,
        "year": args.year,
        "query": args.query,
        "max_results": args.max_results,
        "start": args.start,
    }
    manifest["items"] = merge_manifest_items(manifest.get("items", []), papers, root_dir)
    return manifest


def write_failed_url_file(failed_url_file: Path, items: List[Dict[str, Any]]) -> None:
    failed_lines = []
    for item in items:
        if item.get("import_status") == "failed" or item.get("download_status") == "failed":
            title = normalize_space(str(item.get("title", "")))
            pdf_url = normalize_space(str(item.get("pdf_url", "")))
            if pdf_url:
                failed_lines.append(f"{title}\t{pdf_url}" if title else pdf_url)
    failed_url_file.parent.mkdir(parents=True, exist_ok=True)
    failed_url_file.write_text("\n".join(failed_lines) + ("\n" if failed_lines else ""), encoding="utf-8")


def validate_pdf_file(path: Path) -> Optional[str]:
    if not path.exists() or path.stat().st_size == 0:
        return "file is missing or empty"

    try:
        with fitz.open(path) as doc:
            if doc.page_count <= 0:
                return "page_count=0"
    except Exception as exc:
        return str(exc)

    return None


def download_pdf(
    session: requests.Session,
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
        cached_error = validate_pdf_file(output_path)
        if not cached_error:
            logger.info(f"Skip download, file already exists: {output_path}")
            return output_path
        logger.warning(
            f"Cached PDF is invalid and will be re-downloaded: path={output_path} "
            f"reason={cached_error}"
        )
        output_path.unlink(missing_ok=True)

    total_attempts = PDF_DOWNLOAD_VALIDATION_RETRIES + 1
    last_error = ""
    for attempt in range(1, total_attempts + 1):
        logger.info(
            f"[crawl] PDF download start: url={url} -> {output_path} "
            f"attempt={attempt}/{total_attempts}"
        )
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(
                    f"Download target does not look like PDF: content_type={content_type} url={url}"
                )

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

        if output_path.stat().st_size == 0:
            last_error = "downloaded file is empty"
        else:
            last_error = validate_pdf_file(output_path) or ""
            if not last_error:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"[crawl] PDF download finished: path={output_path} size_mb={file_size_mb:.2f}"
                )
                return output_path

        logger.warning(
            f"Downloaded PDF validation failed: path={output_path} reason={last_error}"
        )
        output_path.unlink(missing_ok=True)

    raise ValueError(f"Downloaded invalid PDF after {total_attempts} attempts: {last_error}")


def normalize_error_text(exc: Exception) -> str:
    message = normalize_space(str(exc))
    if not message:
        return ""

    normalized_parts = [message.lower()]
    try:
        repaired = message.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        repaired = ""
    repaired = normalize_space(repaired).lower()
    if repaired:
        normalized_parts.append(repaired)

    return " ".join(normalized_parts)


def error_has_any_marker(message: str, markers: Iterable[str]) -> bool:
    return any(marker in message for marker in markers)


def is_timeout_like_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    return error_has_any_marker(
        normalize_error_text(exc),
        [
            "timeout",
            "timed out",
            "超时",
        ],
    )


def is_parse_failure_error(exc: Exception) -> bool:
    return error_has_any_marker(
        normalize_error_text(exc),
        [
            "parsing failed",
            "retry limit reached",
            "解析任务失败",
            "解析失败",
            "任务轮询",
        ],
    )


def is_missing_text_error(exc: Exception) -> bool:
    return error_has_any_marker(
        normalize_error_text(exc),
        [
            "未能从文件中提取文本",
            "提取文本",
            "no text",
            "empty text",
        ],
    )


def is_pdf_content_error(exc: Exception) -> bool:
    return error_has_any_marker(
        normalize_error_text(exc),
        [
            "please replace the file",
            "pdf兜底解析失败",
            "unsupported pdf",
            "file is broken",
            "file is damaged",
        ],
    )


def is_retryable_import_error(exc: Exception) -> bool:
    return is_timeout_like_error(exc) or is_parse_failure_error(exc)


def should_fallback_to_metadata_import(exc: Exception) -> bool:
    return (
        is_timeout_like_error(exc)
        or is_parse_failure_error(exc)
        or is_missing_text_error(exc)
        or is_pdf_content_error(exc)
    )


def run_import_graph(task_id: str, local_dir: Path, local_file_path: Path) -> Dict[str, Any]:
    update_task_status(task_id, "processing")
    state = get_default_state()
    state["task_id"] = task_id
    state["local_dir"] = str(local_dir)
    state["local_file_path"] = str(local_file_path)

    final_state: Optional[Dict[str, Any]] = None
    try:
        for event in kb_import_app.stream(state):
            for node_name, node_state in event.items():
                logger.info(f"[{task_id}] import node completed: {node_name}")
                final_state = node_state
        update_task_status(task_id, "completed")
        return final_state or state
    except Exception:
        update_task_status(task_id, "failed")
        raise
    finally:
        clear_task(task_id)


def dump_metadata(local_dir: Path, paper: Dict[str, Any]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = local_dir / "paper_metadata.txt"
    lines = [
        f"title: {paper.get('title', '')}",
        f"source: {paper.get('source', '')}",
        f"venue: {paper.get('venue', '')}",
        f"id: {paper.get('id', '')}",
        f"published: {paper.get('published', '')}",
        f"authors: {', '.join(paper.get('authors') or [])}",
        f"pdf_url: {paper.get('pdf_url', '')}",
        f"detail_url: {paper.get('detail_url', '')}",
        "",
        paper.get("summary", "") or "",
    ]
    metadata_path.write_text("\n".join(lines), encoding="utf-8")


def build_metadata_markdown(paper: Dict[str, Any], pdf_path: Path) -> str:
    authors = paper.get("authors") or []
    authors_text = ", ".join(str(author) for author in authors if str(author).strip()) or "Unknown"
    title = normalize_space(str(paper.get("title", ""))) or pdf_path.stem
    venue = normalize_space(str(paper.get("venue", "")))
    year = extract_year_text(str(paper.get("year", "")) or str(paper.get("published", "")))
    published = normalize_space(str(paper.get("published", "")))
    summary = normalize_space(str(paper.get("summary", "")))
    detail_url = str(paper.get("detail_url", "")).strip()
    pdf_url = str(paper.get("pdf_url", "")).strip()
    source = str(paper.get("source", "")).strip()

    lines = [f"# {title}", ""]
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Title: {title}")
    lines.append(f"- Authors: {authors_text}")
    if venue:
        lines.append(f"- Venue: {venue}")
    if year:
        lines.append(f"- Year: {year}")
    if published:
        lines.append(f"- Published: {published}")
    if source:
        lines.append(f"- Source: {source}")
    if detail_url:
        lines.append(f"- Detail URL: {detail_url}")
    if pdf_url:
        lines.append(f"- PDF URL: {pdf_url}")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(summary or "Abstract not available from the source listing.")
    lines.append("")
    lines.append("## Import Note")
    lines.append("")
    lines.append(
        "This markdown was generated from crawler metadata because PDF parsing did not produce usable text."
    )
    lines.append("")
    return "\n".join(lines)


def write_metadata_fallback_md(paper_dir: Path, paper: Dict[str, Any], pdf_path: Path) -> Path:
    md_path = paper_dir / f"{pdf_path.stem}.md"
    md_content = build_metadata_markdown(paper, pdf_path)
    md_path.write_text(md_content, encoding="utf-8")
    logger.warning(f"[fallback] metadata markdown created: {md_path}")
    return md_path


def sync_manifest_item_from_disk(item: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = Path(item.get("pdf_path", ""))
    chunks_path = Path(item.get("chunks_path", ""))

    if pdf_path.exists() and pdf_path.stat().st_size > 0 and not validate_pdf_file(pdf_path):
        item["download_status"] = "completed"

    if chunks_path.exists():
        item["import_status"] = "completed"
        item["error"] = ""

    return item


def execute_import_pipeline(
    paper: Dict[str, Any],
    paper_dir: Path,
    pdf_path: Path,
    *,
    sleep_seconds: float = 0.0,
    max_retries: int = DEFAULT_IMPORT_RETRIES,
) -> Dict[str, Any]:
    final_state: Optional[Dict[str, Any]] = None
    last_task_id = ""
    for attempt in range(1, max_retries + 2):
        last_task_id = f"batch_import_{uuid.uuid4().hex[:12]}"
        try:
            logger.info(
                f"[{last_task_id}] import attempt {attempt}/{max_retries + 1}: "
                f"{paper.get('title')}"
            )
            final_state = run_import_graph(last_task_id, paper_dir, pdf_path)
            break
        except Exception as exc:
            if should_fallback_to_metadata_import(exc):
                fallback_md_path = write_metadata_fallback_md(paper_dir, paper, pdf_path)
                fallback_task_id = f"batch_import_{uuid.uuid4().hex[:12]}"
                logger.warning(
                    f"[{fallback_task_id}] switching to metadata-only import fallback: "
                    f"{paper.get('title')}"
                )
                final_state = run_import_graph(fallback_task_id, paper_dir, fallback_md_path)
                last_task_id = fallback_task_id
                break
            if attempt > max_retries or not is_retryable_import_error(exc):
                raise
            logger.warning(
                f"[{last_task_id}] retryable import failure on attempt {attempt}/"
                f"{max_retries + 1}: {exc}. Retrying..."
            )
            time.sleep(3)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return {
        "task_id": last_task_id,
        "title": paper.get("title") or pdf_path.stem,
        "pdf_url": paper.get("pdf_url") or "",
        "paper_dir": str(paper_dir),
        "pdf_path": str(pdf_path),
        "paper_title": (final_state or {}).get("paper_title", ""),
        "status": "completed",
    }


def stage_downloads(
    session: requests.Session,
    manifest: Dict[str, Any],
    *,
    overwrite_pdf: bool,
    manifest_path: Path,
    failed_url_file: Path,
) -> None:
    for item in manifest.get("items", []):
        sync_manifest_item_from_disk(item)
        paper = item["paper"]
        paper_dir = Path(item["paper_dir"])
        pdf_path = Path(item["pdf_path"])
        try:
            dump_metadata(paper_dir, paper)
            download_pdf(session, item["pdf_url"], pdf_path, overwrite=overwrite_pdf)
            item["download_status"] = "completed"
            item["error"] = ""
        except Exception as exc:
            item["download_status"] = "failed"
            item["error"] = str(exc)
            item["updated_at"] = utc_now_text()
            manifest["updated_at"] = utc_now_text()
            save_resume_manifest(manifest_path, manifest)
            write_failed_url_file(failed_url_file, manifest.get("items", []))
            logger.error(f"Download failed: {paper.get('title')} | {exc}", exc_info=True)
            continue

        item["updated_at"] = utc_now_text()
        manifest["updated_at"] = utc_now_text()
        save_resume_manifest(manifest_path, manifest)

    write_failed_url_file(failed_url_file, manifest.get("items", []))


def collect_import_candidates(
    manifest: Dict[str, Any],
    *,
    force_reimport: bool,
) -> Dict[str, Any]:
    sync_count = 0
    candidates: List[Dict[str, Any]] = []

    for item in manifest.get("items", []):
        sync_manifest_item_from_disk(item)
        sync_count += 1
        if item.get("download_status") != "completed":
            continue
        if item.get("import_status") == "completed" and not force_reimport:
            continue
        candidates.append(item)

    return {
        "synced": sync_count,
        "candidates": candidates,
    }


def format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    completed = max(0, min(completed, total))
    filled = int(width * completed / total)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def log_import_progress(
    *,
    completed: int,
    success_count: int,
    failed_count: int,
    total: int,
    title: str,
) -> None:
    progress_bar = format_progress_bar(completed, total)
    logger.info(
        f"Import progress {progress_bar} {completed}/{total} "
        f"success={success_count} failed={failed_count} current={title}"
    )


def import_staged_papers(
    manifest: Dict[str, Any],
    *,
    import_batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    force_reimport: bool,
    manifest_path: Path,
    failed_url_file: Path,
) -> Dict[str, int]:
    imported_count = 0
    failed_count = 0
    candidate_result = collect_import_candidates(
        manifest,
        force_reimport=force_reimport,
    )
    sync_count = int(candidate_result["synced"])
    candidates = list(candidate_result["candidates"])
    total_candidates = len(candidates)
    total_completed = 0
    if total_candidates == 0:
        return {
            "synced": sync_count,
            "imported": 0,
            "failed": 0,
            "remaining": 0,
            "batches": 0,
        }

    effective_batch_size = import_batch_size or total_candidates
    batch_count = 0

    for batch_start in range(0, total_candidates, effective_batch_size):
        batch_count += 1
        batch_items = candidates[batch_start : batch_start + effective_batch_size]
        logger.info(
            f"Import batch {batch_count}: size={len(batch_items)} "
            f"processed={batch_start}/{total_candidates}"
        )

        for item in batch_items:
            paper = item["paper"]
            paper_dir = Path(item["paper_dir"])
            pdf_path = Path(item["pdf_path"])
            try:
                result = execute_import_pipeline(
                    paper,
                    paper_dir,
                    pdf_path,
                    sleep_seconds=sleep_seconds,
                    max_retries=max_retries,
                )
                item["import_status"] = "completed"
                item["task_id"] = result["task_id"]
                item["paper_title"] = result.get("paper_title", "")
                item["error"] = ""
                imported_count += 1
            except Exception as exc:
                item["import_status"] = "failed"
                item["error"] = str(exc)
                failed_count += 1
                logger.error(f"Import failed: {paper.get('title')} | {exc}", exc_info=True)

            total_completed += 1
            log_import_progress(
                completed=total_completed,
                success_count=imported_count,
                failed_count=failed_count,
                total=total_candidates,
                title=paper.get("title") or pdf_path.stem,
            )
            item["updated_at"] = utc_now_text()
            manifest["updated_at"] = utc_now_text()
            save_resume_manifest(manifest_path, manifest)
            write_failed_url_file(failed_url_file, manifest.get("items", []))

        remaining_after_batch = sum(
            1
            for item in manifest.get("items", [])
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        )
        logger.info(
            f"Import batch {batch_count} finished: imported={imported_count} "
            f"failed={failed_count} remaining={remaining_after_batch}"
        )

    return {
        "synced": sync_count,
        "imported": imported_count,
        "failed": failed_count,
        "remaining": sum(
            1
            for item in manifest.get("items", [])
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        ),
        "batches": batch_count,
    }


def import_one_paper(
    session: requests.Session,
    paper: Dict[str, Any],
    index: int,
    root_dir: Path,
    overwrite_pdf: bool = False,
    sleep_seconds: float = 0.0,
    max_retries: int = DEFAULT_IMPORT_RETRIES,
) -> Dict[str, Any]:
    pdf_url = ensure_pdf_suffix(paper["pdf_url"])
    paper = {**paper, "pdf_url": pdf_url}

    paths = build_paper_paths(index, paper, root_dir)
    paper_dir = paths["paper_dir"]
    pdf_path = paths["pdf_path"]
    logger.info(f"start import: {paper.get('title')}")
    logger.info(f"pdf_url: {pdf_url}")

    dump_metadata(paper_dir, paper)
    download_pdf(session, pdf_url, pdf_path, overwrite=overwrite_pdf)
    return execute_import_pipeline(
        paper,
        paper_dir,
        pdf_path,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )


def dedupe_papers(papers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_items = []
    for paper in papers:
        key = normalize_space(paper.get("pdf_url") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(paper)
    return unique_items


def resolve_source_year(args: argparse.Namespace, source: str) -> int:
    unified_year = getattr(args, "year", 0) or 0
    if unified_year > 0:
        return unified_year

    legacy_attr = f"{source}_year"
    legacy_year = getattr(args, legacy_attr, 0) or 0
    if legacy_year > 0:
        return legacy_year

    return DEFAULT_SOURCE_YEARS.get(source, 0)


def resolve_conference_target(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    conference = normalize_venue_token(str(getattr(args, "conference", "") or ""))
    if not conference:
        return None

    year = getattr(args, "year", 0) or 0
    if year <= 0:
        raise ValueError("--conference requires a positive --year")

    rule = CONFERENCE_TARGET_RULES.get(conference)
    if rule:
        target = {"source": rule["source"], "year": str(year)}
        for key, template in (rule.get("params") or {}).items():
            target_key = {
                ("cvf", "event"): "cvf_event",
                ("acl", "event"): "acl_event",
                ("openreview", "venueid"): "openreview_venueid",
            }.get((rule["source"], key), key)
            target[target_key] = template.format(year=year)
        for key, value in rule.items():
            if key in {"source", "uses_year", "params"}:
                continue
            target[key] = value.format(year=year) if isinstance(value, str) else value
        return target

    raise ValueError(
        f"Unsupported conference={conference}. "
        f"Supported conferences: {', '.join(CONFERENCE_SOURCE_CHOICES)}"
    )


def interleave_paper_groups(groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                merged.append(group[idx])
    return merged


def collect_all_sources_papers(
    args: argparse.Namespace, session: requests.Session
) -> List[Dict[str, Any]]:
    query = args.query.strip()
    logger.info(
        f"[crawl] multi-source search start: query={query!r} max_results={args.max_results}"
    )

    paper_groups: List[List[Dict[str, Any]]] = []
    sources = list(DEFAULT_MULTI_SOURCE_ORDER)
    sources.extend(
        source
        for source in SOURCE_OPTION_FIELDS
        if str(getattr(args, SOURCE_OPTION_FIELDS[source][1], "") or "").strip()
    )
    for source in sources:
        papers = run_source_search_with_retries(
            session=session,
            source=source,
            args=args,
            query=query,
            max_results=args.max_results,
            start=args.start,
            strict=False,
        )
        if papers:
            paper_groups.append(papers)

    merged = interleave_paper_groups(paper_groups)
    deduped = dedupe_papers(merged)
    final_papers = deduped[: args.max_results]
    logger.info(
        f"[crawl] multi-source search finished: source_groups={len(paper_groups)} "
        f"merged={len(merged)} deduped={len(deduped)} returned={len(final_papers)}"
    )
    return final_papers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch download papers from multiple open sources and import them into the knowledge base."
    )
    parser.add_argument(
        "--source",
        choices=[
            "all",
            "arxiv",
            "url_file",
            "openreview",
            "acl",
            "cvf",
            "neurips",
            "icml",
            "aaai",
            "ijcai",
        ],
        default="all",
        help="Paper source. If omitted, search across multiple configured sources.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="For arXiv: raw API query. For other sources: local title keyword filter.",
    )
    parser.add_argument(
        "--conference",
        default="",
        help=(
            "Conference shorthand such as `cvpr`, `iccv`, `eccv`, `wacv`, "
            "`acl`, `emnlp`, `naacl`, `iclr`, `colm`, `neurips`, `icml`, "
            "`aaai`, or `ijcai`. When provided, the script will derive the "
            "underlying source-specific parameters automatically."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=0,
        help="Unified publication year for conference-based searches.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of papers to process.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset after filtering.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        default="submittedDate",
        help="arXiv sort field.",
    )
    parser.add_argument(
        "--sort-order",
        choices=["ascending", "descending"],
        default="descending",
        help="arXiv sort order.",
    )
    parser.add_argument(
        "--url-file",
        default="",
        help="Local text file with one PDF URL per line, or `title<TAB>url`.",
    )
    parser.add_argument(
        "--openreview-venueid",
        default="",
        help='OpenReview venue id, for example `ICLR.cc/2024/Conference`.',
    )
    parser.add_argument(
        "--acl-event",
        default="",
        help='ACL Anthology event slug, for example `acl-2024` or `emnlp-2024`.',
    )
    parser.add_argument(
        "--cvf-event",
        default="",
        help='CVF proceedings key, for example `CVPR2023` or `ICCV2023`.',
    )
    parser.add_argument("--neurips-year", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--icml-year", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--aaai-year", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--ijcai-year", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_IMPORT_ROOT),
        help="Batch output directory.",
    )
    parser.add_argument(
        "--overwrite-pdf",
        action="store_true",
        help="Re-download PDF even if it already exists locally.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between papers to reduce pressure on external services.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only collect paper metadata and download/stage PDFs, without running import batches.",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Reuse an existing staged batch and only run the import phase until all pending items finish.",
    )
    parser.add_argument(
        "--refresh-papers",
        action="store_true",
        help="Ignore saved batch manifest and collect the paper list again.",
    )
    parser.add_argument(
        "--import-batch-size",
        type=int,
        default=DEFAULT_IMPORT_BATCH_SIZE,
        help="How many staged papers to import per batch. The script will keep looping batches in one run. Use 0 to process all pending papers in a single batch.",
    )
    parser.add_argument(
        "--force-reimport",
        action="store_true",
        help="Re-run import even for items already marked as completed in the resume manifest.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matched papers without downloading or importing.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_results <= 0:
        raise ValueError("--max-results must be greater than 0")
    if args.download_only and args.import_only:
        raise ValueError("--download-only and --import-only cannot be used together")
    if args.import_batch_size < 0:
        raise ValueError("--import-batch-size must be >= 0")

    if args.conference.strip():
        resolve_conference_target(args)
        if not args.query.strip():
            raise ValueError("--conference mode requires --query")
        return

    if args.source in {"all", "arxiv"} and not args.query.strip():
        raise ValueError("source=arxiv requires --query")
    if args.source == "url_file" and not args.url_file.strip():
        raise ValueError("source=url_file requires --url-file")
    if args.source in DIRECT_SOURCE_REQUIRED_FIELDS:
        field_name = DIRECT_SOURCE_REQUIRED_FIELDS[args.source]
        if not str(getattr(args, field_name, "") or "").strip():
            raise ValueError(f"source={args.source} requires --{field_name.replace('_', '-')}")
    if args.source in {"neurips", "icml", "aaai", "ijcai"}:
        resolved_year = resolve_source_year(args, args.source)
        if resolved_year <= 0:
            raise ValueError(f"source={args.source} requires a positive --year")


def collect_papers(args: argparse.Namespace, session: requests.Session) -> List[Dict[str, Any]]:
    conference_target = resolve_conference_target(args)
    if conference_target:
        target = dict(conference_target)
        source = str(target.pop("source"))
        target_kwargs = {
            "query": args.query.strip(),
            "max_results": args.max_results,
            "start": args.start,
        }
        if source in YEAR_BASED_SOURCES and "year" in target:
            target["year"] = int(target["year"])
        elif "year" in target:
            target.pop("year", None)
        option_fields = SOURCE_OPTION_FIELDS.get(source)
        if option_fields:
            param_name, arg_name = option_fields
            if arg_name in target:
                target[param_name] = target.pop(arg_name)
        target_kwargs.update(target)
        return run_named_search(session=session, source=source, **target_kwargs)

    if args.source == "all":
        return collect_all_sources_papers(args, session)

    if args.source == "url_file":
        papers = load_pdf_url_file(Path(args.url_file))
        return dedupe_papers(papers[args.start : args.start + args.max_results])
    if args.source in CLI_SOURCE_CHOICES:
        return run_source_search_with_retries(
            session=session,
            source=args.source,
            args=args,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
            strict=True,
        )

    raise ValueError(f"Unsupported source: {args.source}")


def summarize_manifest(manifest: Dict[str, Any]) -> Dict[str, int]:
    items = manifest.get("items", [])
    return {
        "total": len(items),
        "downloaded": sum(1 for item in items if item.get("download_status") == "completed"),
        "imported": sum(1 for item in items if item.get("import_status") == "completed"),
        "failed": sum(
            1
            for item in items
            if item.get("import_status") == "failed" or item.get("download_status") == "failed"
        ),
        "pending_import": sum(
            1
            for item in items
            if item.get("download_status") == "completed"
            and item.get("import_status") != "completed"
        ),
    }


def main() -> int:
    load_dotenv(override=True)
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as e:
        logger.error(str(e))
        return 2

    root_dir = Path(args.output_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    run_key = build_resume_run_key(args)
    resume_paths = build_resume_paths(root_dir, run_key)
    manifest_path = resume_paths["manifest_path"]
    failed_url_file = resume_paths["failed_url_file"]
    manifest = load_resume_manifest(manifest_path)
    session = build_requests_session()

    papers: List[Dict[str, Any]] = []
    need_collect = not manifest or args.refresh_papers

    if args.import_only and not manifest:
        logger.error(
            f"No staged batch manifest found for the current arguments: {manifest_path}. "
            f"Run once without --import-only first."
        )
        return 1

    if need_collect:
        try:
            papers = collect_papers(args, session)
        except Exception as e:
            logger.error(f"Failed to collect papers: {e}", exc_info=True)
            return 1

        if not papers:
            logger.warning("No papers matched the current source and filter.")
            return 0

        manifest = build_resume_manifest(args, papers, root_dir, manifest if not args.refresh_papers else None)
        save_resume_manifest(manifest_path, manifest)
        logger.info(f"Saved batch manifest: {manifest_path}")
    else:
        papers = [item.get("paper", {}) for item in manifest.get("items", [])]
        logger.info(f"Resuming staged batch from manifest: {manifest_path}")

    logger.info(f"Prepared {len(papers)} papers for processing.")
    for idx, paper in enumerate(papers, start=1):
        logger.info(
            f"[{idx}] {paper.get('title')} | source={paper.get('source')} | pdf={paper.get('pdf_url')}"
        )

    if args.list_only:
        return 0

    if not args.import_only:
        logger.info("Stage 1/2: downloading and staging PDFs...")
        stage_downloads(
            session,
            manifest,
            overwrite_pdf=args.overwrite_pdf,
            manifest_path=manifest_path,
            failed_url_file=failed_url_file,
        )

    manifest = load_resume_manifest(manifest_path) or manifest
    if args.download_only:
        summary = summarize_manifest(manifest)
        logger.info(
            f"Download staging finished: total={summary['total']} downloaded={summary['downloaded']} "
            f"pending_import={summary['pending_import']} failed={summary['failed']}"
        )
        logger.info(f"Resume manifest: {manifest_path}")
        return 0

    logger.info(
        f"Stage 2/2: importing staged papers "
        f"(per_batch={'all' if args.import_batch_size == 0 else args.import_batch_size}, auto_loop=yes)..."
    )
    import_summary = import_staged_papers(
        manifest,
        import_batch_size=args.import_batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=DEFAULT_IMPORT_RETRIES,
        force_reimport=args.force_reimport,
        manifest_path=manifest_path,
        failed_url_file=failed_url_file,
    )
    manifest = load_resume_manifest(manifest_path) or manifest
    summary = summarize_manifest(manifest)

    logger.info(
        f"Batch import status: total={summary['total']} downloaded={summary['downloaded']} "
        f"imported={summary['imported']} failed={summary['failed']} pending_import={summary['pending_import']}"
    )
    logger.info(
        f"This run processed: batches={import_summary['batches']} imported={import_summary['imported']} "
        f"failed={import_summary['failed']} remaining={import_summary['remaining']}"
    )
    logger.info(f"Resume manifest: {manifest_path}")
    logger.info(f"Failed URL file: {failed_url_file}")

    return 0 if import_summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
