"""Paper collection orchestration: conference resolution and source dispatch."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.core.logger import logger

from .config import (
    CLI_SOURCE_CHOICES,
    CONFERENCE_SOURCE_CHOICES,
    CONFERENCE_TARGET_RULES,
    DEFAULT_MULTI_SOURCE_ORDER,
    SOURCE_OPTION_FIELDS,
    YEAR_BASED_SOURCES,
)
from .metadata import dedupe_papers, enrich_paper_metadata, normalize_venue_token
from .pdf import guess_title_from_url
from .search import (
    run_named_search,
    run_source_search_with_retries,
)


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
