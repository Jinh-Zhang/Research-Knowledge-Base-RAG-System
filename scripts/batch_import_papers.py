import argparse
import hashlib
import re
import sys
import time
import uuid
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv

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
DEFAULT_USER_AGENT = "knowledge-base-paper-importer/1.0"
DEFAULT_IMPORT_ROOT = PROJECT_ROOT / "output" / "batch_imports"
REQUEST_TIMEOUT = 60
ICML_YEAR_TO_VOLUME = {
    2024: 235,
    2025: 267,
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
    return session


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text or "", flags=re.S)
    return normalize_space(unescape(text))


def title_matches_query(title: str, query: str) -> bool:
    query = normalize_space(query).lower()
    title = normalize_space(title).lower()
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

    return lowered in title or all(token in title for token in tokens)


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
        "acl": "ACL",
        "emnlp": "EMNLP",
        "naacl": "NAACL",
        "aaai": "AAAI",
        "ijcai": "IJCAI",
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
    request_limit = min(max(start + max_results, 50), 1000)
    logger.info(
        f"[crawl] OpenReview search start: venueid={venueid} query={query!r} "
        f"start={start} max_results={max_results} request_limit={request_limit}"
    )
    params = {
        "content.venueid": venueid,
        "limit": request_limit,
    }
    response = session.get(OPENREVIEW_NOTES_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    total_notes = len(payload.get("notes", []))

    papers: List[Dict[str, Any]] = []
    for note in payload.get("notes", []):
        content = note.get("content") or {}
        title = normalize_space(extract_openreview_value(content.get("title")) or "")
        if not title or not title_matches_query(title, query):
            continue

        authors = extract_openreview_value(content.get("authors")) or []
        if not isinstance(authors, list):
            authors = [str(authors)]

        note_id = note.get("id") or note.get("forum") or ""
        published = str(note.get("tcdate") or "")
        year_match = re.search(r"/(\d{4})/", venueid)
        year = year_match.group(1) if year_match else ""

        papers.append(
            enrich_paper_metadata(
                {
                "source": "openreview",
                "id": note_id,
                "title": title,
                "summary": normalize_space(
                    extract_openreview_value(content.get("abstract")) or ""
                ),
                "published": year or published,
                "authors": [normalize_space(str(x)) for x in authors if str(x).strip()],
                "pdf_url": f"https://openreview.net/pdf?id={note_id}",
                "detail_url": f"https://openreview.net/forum?id={note_id}",
                "venue": venueid,
                "year": year,
                }
            )
        )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] OpenReview search finished: fetched_notes={total_notes} "
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
            enrich_paper_metadata(
                {
                "source": "acl",
                "id": href.strip("/"),
                "title": title,
                "summary": "",
                "published": year_match.group(1) if year_match else "",
                "authors": [],
                "pdf_url": pdf_url,
                "detail_url": article_url,
                "venue": event,
                "year": year_match.group(1) if year_match else "",
                }
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
            enrich_paper_metadata(
                {
                "source": "cvf",
                "id": pdf_href.rsplit("/", 1)[-1],
                "title": title,
                "summary": "",
                "published": year_match.group(1) if year_match else "",
                "authors": [],
                "pdf_url": urljoin(CVF_BASE_URL, pdf_href),
                "detail_url": urljoin(CVF_BASE_URL, detail_href),
                "venue": event,
                "year": year_match.group(1) if year_match else "",
                }
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
            enrich_paper_metadata(
                {
                    "source": "neurips",
                    "id": abstract_href.rsplit("/", 1)[-1],
                    "title": title,
                    "summary": "",
                    "published": str(year),
                    "authors": [
                        normalize_space(author)
                        for author in authors_text.split(",")
                        if normalize_space(author)
                    ],
                    "pdf_url": build_neurips_pdf_url(abstract_href),
                    "detail_url": urljoin(NEURIPS_BASE_URL, abstract_href),
                    "venue": "NeurIPS",
                    "year": str(year),
                }
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
    volume = ICML_YEAR_TO_VOLUME.get(year)
    if not volume:
        raise ValueError(
            f"ICML year {year} is not configured yet. "
            f"Available years: {sorted(ICML_YEAR_TO_VOLUME)}"
        )

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
            enrich_paper_metadata(
                {
                    "source": "icml",
                    "id": pdf_url.rsplit("/", 1)[-1],
                    "title": title,
                    "summary": "",
                    "published": str(year),
                    "authors": [
                        normalize_space(author)
                        for author in authors_text.replace("\xa0", " ").split(",")
                        if normalize_space(author)
                    ],
                    "pdf_url": pdf_url,
                    "detail_url": normalize_space(abs_match.group("abs")) if abs_match else proceedings_url,
                    "venue": "ICML",
                    "year": str(year),
                }
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
            enrich_paper_metadata(
                {
                    "source": "ijcai",
                    "id": Path(pdf_href).stem,
                    "title": title,
                    "summary": "",
                    "published": str(year),
                    "authors": [
                        normalize_space(author)
                        for author in authors_text.split(",")
                        if normalize_space(author)
                    ],
                    "pdf_url": urljoin(proceedings_url, pdf_href),
                    "detail_url": urljoin(IJCAI_BASE_URL, detail_href),
                    "venue": "IJCAI",
                    "year": str(year),
                }
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
                enrich_paper_metadata(
                    {
                        "source": "aaai",
                        "id": detail_href.rsplit("/", 1)[-1],
                        "title": title,
                        "summary": "",
                        "published": str(year),
                        "authors": [
                            normalize_space(author)
                            for author in re.split(r",|;|\n", authors_text)
                            if normalize_space(author)
                        ],
                        "pdf_url": urljoin(
                            AAAI_BASE_URL, normalize_space(pdf_match.group("pdf"))
                        ),
                        "detail_url": urljoin(AAAI_BASE_URL, detail_href),
                        "venue": "AAAI",
                        "year": str(year),
                    }
                )
            )

    sliced_papers = papers[start : start + max_results]
    logger.info(
        f"[crawl] AAAI search finished: year={year} "
        f"filtered_matches={len(papers)} returned={len(sliced_papers)}"
    )
    return sliced_papers


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


def download_pdf(
    session: requests.Session,
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
        logger.info(f"Skip download, file already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[crawl] PDF download start: url={url} -> {output_path}")
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
        raise ValueError(f"Downloaded empty file: {output_path}")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"[crawl] PDF download finished: path={output_path} size_mb={file_size_mb:.2f}"
    )
    return output_path


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


def import_one_paper(
    session: requests.Session,
    paper: Dict[str, Any],
    index: int,
    root_dir: Path,
    overwrite_pdf: bool = False,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    pdf_url = ensure_pdf_suffix(paper["pdf_url"])
    paper = {**paper, "pdf_url": pdf_url}

    paper_dir = root_dir / build_paper_dir_name(index, paper)
    pdf_path = paper_dir / f"{build_import_file_stem(paper)}.pdf"
    task_id = f"batch_import_{uuid.uuid4().hex[:12]}"

    logger.info(f"[{task_id}] start import: {paper.get('title')}")
    logger.info(f"[{task_id}] pdf_url: {pdf_url}")

    dump_metadata(paper_dir, paper)
    download_pdf(session, pdf_url, pdf_path, overwrite=overwrite_pdf)
    final_state = run_import_graph(task_id, paper_dir, pdf_path)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return {
        "task_id": task_id,
        "title": paper.get("title") or pdf_path.stem,
        "pdf_url": pdf_url,
        "paper_dir": str(paper_dir),
        "pdf_path": str(pdf_path),
        "paper_title": (final_state or {}).get("paper_title", ""),
        "status": "completed",
    }


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

    arxiv_papers = search_arxiv(
        session=session,
        query=query,
        max_results=args.max_results,
        start=args.start,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
    )
    if arxiv_papers:
        paper_groups.append(arxiv_papers)

    neurips_papers = search_neurips_year(
        session=session,
        year=args.neurips_year,
        query=query,
        max_results=args.max_results,
        start=args.start,
    )
    if neurips_papers:
        paper_groups.append(neurips_papers)

    icml_papers = search_icml_year(
        session=session,
        year=args.icml_year,
        query=query,
        max_results=args.max_results,
        start=args.start,
    )
    if icml_papers:
        paper_groups.append(icml_papers)

    aaai_papers = search_aaai_year(
        session=session,
        year=args.aaai_year,
        query=query,
        max_results=args.max_results,
        start=args.start,
    )
    if aaai_papers:
        paper_groups.append(aaai_papers)

    ijcai_papers = search_ijcai_year(
        session=session,
        year=args.ijcai_year,
        query=query,
        max_results=args.max_results,
        start=args.start,
    )
    if ijcai_papers:
        paper_groups.append(ijcai_papers)

    if args.openreview_venueid.strip():
        openreview_papers = search_openreview(
            session=session,
            venueid=args.openreview_venueid.strip(),
            query=query,
            max_results=args.max_results,
            start=args.start,
        )
        if openreview_papers:
            paper_groups.append(openreview_papers)

    if args.acl_event.strip():
        acl_papers = search_acl_event(
            session=session,
            event=args.acl_event.strip(),
            query=query,
            max_results=args.max_results,
            start=args.start,
        )
        if acl_papers:
            paper_groups.append(acl_papers)

    if args.cvf_event.strip():
        cvf_papers = search_cvf_event(
            session=session,
            event=args.cvf_event.strip(),
            query=query,
            max_results=args.max_results,
            start=args.start,
        )
        if cvf_papers:
            paper_groups.append(cvf_papers)

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
    parser.add_argument(
        "--neurips-year",
        type=int,
        default=2024,
        help="NeurIPS proceedings year, for example `2024`.",
    )
    parser.add_argument(
        "--icml-year",
        type=int,
        default=2024,
        help="ICML proceedings year, for example `2024`.",
    )
    parser.add_argument(
        "--aaai-year",
        type=int,
        default=2025,
        help="AAAI proceedings year, for example `2025`.",
    )
    parser.add_argument(
        "--ijcai-year",
        type=int,
        default=2024,
        help="IJCAI proceedings year, for example `2024`.",
    )
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
        "--list-only",
        action="store_true",
        help="Only list matched papers without downloading or importing.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_results <= 0:
        raise ValueError("--max-results must be greater than 0")

    if args.source in {"all", "arxiv"} and not args.query.strip():
        raise ValueError("source=arxiv requires --query")
    if args.source == "url_file" and not args.url_file.strip():
        raise ValueError("source=url_file requires --url-file")
    if args.source == "openreview" and not args.openreview_venueid.strip():
        raise ValueError("source=openreview requires --openreview-venueid")
    if args.source == "acl" and not args.acl_event.strip():
        raise ValueError("source=acl requires --acl-event")
    if args.source == "cvf" and not args.cvf_event.strip():
        raise ValueError("source=cvf requires --cvf-event")
    if args.source == "neurips" and args.neurips_year <= 0:
        raise ValueError("source=neurips requires a positive --neurips-year")
    if args.source == "icml" and args.icml_year <= 0:
        raise ValueError("source=icml requires a positive --icml-year")
    if args.source == "aaai" and args.aaai_year <= 0:
        raise ValueError("source=aaai requires a positive --aaai-year")
    if args.source == "ijcai" and args.ijcai_year <= 0:
        raise ValueError("source=ijcai requires a positive --ijcai-year")


def collect_papers(args: argparse.Namespace, session: requests.Session) -> List[Dict[str, Any]]:
    if args.source == "all":
        return collect_all_sources_papers(args, session)

    if args.source == "arxiv":
        papers = search_arxiv(
            session=session,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
            sort_by=args.sort_by,
            sort_order=args.sort_order,
        )
        return dedupe_papers(papers)

    if args.source == "url_file":
        papers = load_pdf_url_file(Path(args.url_file))
        return dedupe_papers(papers[args.start : args.start + args.max_results])

    if args.source == "openreview":
        papers = search_openreview(
            session=session,
            venueid=args.openreview_venueid.strip(),
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "acl":
        papers = search_acl_event(
            session=session,
            event=args.acl_event.strip(),
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "cvf":
        papers = search_cvf_event(
            session=session,
            event=args.cvf_event.strip(),
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "neurips":
        papers = search_neurips_year(
            session=session,
            year=args.neurips_year,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "icml":
        papers = search_icml_year(
            session=session,
            year=args.icml_year,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "aaai":
        papers = search_aaai_year(
            session=session,
            year=args.aaai_year,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    if args.source == "ijcai":
        papers = search_ijcai_year(
            session=session,
            year=args.ijcai_year,
            query=args.query.strip(),
            max_results=args.max_results,
            start=args.start,
        )
        return dedupe_papers(papers)

    raise ValueError(f"Unsupported source: {args.source}")


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
    session = build_requests_session()

    try:
        papers = collect_papers(args, session)
    except Exception as e:
        logger.error(f"Failed to collect papers: {e}", exc_info=True)
        return 1

    if not papers:
        logger.warning("No papers matched the current source and filter.")
        return 0

    logger.info(f"Prepared {len(papers)} papers for processing.")
    for idx, paper in enumerate(papers, start=1):
        logger.info(
            f"[{idx}] {paper.get('title')} | source={paper.get('source')} | pdf={paper.get('pdf_url')}"
        )

    if args.list_only:
        return 0

    success_count = 0
    failed_items: List[Dict[str, str]] = []

    for idx, paper in enumerate(papers, start=1):
        try:
            result = import_one_paper(
                session=session,
                paper=paper,
                index=idx,
                root_dir=root_dir,
                overwrite_pdf=args.overwrite_pdf,
                sleep_seconds=args.sleep_seconds,
            )
            success_count += 1
            logger.info(
                f"Import completed: title={result['title']} | paper_title={result['paper_title']} | dir={result['paper_dir']}"
            )
        except Exception as e:
            logger.error(f"Import failed: {paper.get('title')} | {e}", exc_info=True)
            failed_items.append(
                {
                    "title": paper.get("title") or "",
                    "pdf_url": paper.get("pdf_url") or "",
                    "error": str(e),
                }
            )

    logger.info(
        f"Batch import finished: success={success_count}, failed={len(failed_items)}."
    )
    for item in failed_items:
        logger.warning(
            f"Failed item: title={item['title']} | url={item['pdf_url']} | error={item['error']}"
        )

    return 0 if not failed_items else 1


if __name__ == "__main__":
    raise SystemExit(main())
