"""Command-line interface: argument parser and validation."""

import argparse

from .config import (
    DEFAULT_IMPORT_BATCH_SIZE,
    DEFAULT_IMPORT_ROOT,
    DIRECT_SOURCE_REQUIRED_FIELDS,
)
from .collect import resolve_conference_target
from .search import resolve_source_year


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
