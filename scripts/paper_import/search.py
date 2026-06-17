"""Search dispatch: build per-source kwargs, run searchers, retry on failure."""

import argparse
import time
from typing import Any, Dict, List

from app.core.logger import logger

from .config import (
    DEFAULT_SOURCE_YEARS,
    SOURCE_OPTION_FIELDS,
    SOURCE_SEARCH_RETRIES,
    SOURCE_SEARCH_RETRY_SLEEP,
    YEAR_BASED_SOURCES,
)
from .crawlers import SOURCE_SEARCHERS
from .http_session import is_retryable_collection_error
from .metadata import dedupe_papers

import requests


def resolve_source_year(args: argparse.Namespace, source: str) -> int:
    unified_year = getattr(args, "year", 0) or 0
    if unified_year > 0:
        return unified_year

    legacy_attr = f"{source}_year"
    legacy_year = getattr(args, legacy_attr, 0) or 0
    if legacy_year > 0:
        return legacy_year

    return DEFAULT_SOURCE_YEARS.get(source, 0)


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
    last_exc = None

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
