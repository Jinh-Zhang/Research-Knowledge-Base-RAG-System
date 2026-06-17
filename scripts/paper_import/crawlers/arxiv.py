"""arXiv search via the public Atom API."""

from typing import Any, Dict, List

import requests

from app.core.logger import logger

from ..config import ARXIV_API_URL, REQUEST_TIMEOUT
from ..metadata import enrich_paper_metadata
from ..text_utils import normalize_space
from . import register


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


@register("arxiv")
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
