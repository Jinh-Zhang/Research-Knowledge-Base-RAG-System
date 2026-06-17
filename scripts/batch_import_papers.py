"""Batch download papers from open sources and import them into the knowledge base.

This is the thin entry point. All logic lives in the ``paper_import`` package:

    paper_import/
      config.py           constants, endpoints, source/conference tables
      text_utils.py        pure string helpers
      http_session.py      requests.Session + network-error classification
      metadata.py          paper-record building and enrichment
      metadata_writer.py   sidecar txt / fallback markdown writers
      crawlers/            one module per source (decorator-registered)
      search.py            search dispatch + retries
      pdf.py               PDF download / validation / path helpers
      manifest.py          resume manifest read/write
      import_errors.py     import-error classification
      pipeline.py          import execution + batch loop
      collect.py           source resolution + collection orchestration
      cli.py               argparse parser + validation

Usage examples:
    python batch_import_papers.py --source arxiv --query "transformer" --max-results 10
    python batch_import_papers.py --conference cvpr --year 2024 --query "detection"
    python batch_import_papers.py --source url_file --url-file my_papers.txt
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger  # noqa: E402

from paper_import.cli import build_parser, validate_args  # noqa: E402
from paper_import.collect import collect_papers  # noqa: E402
from paper_import.http_session import build_requests_session  # noqa: E402
from paper_import.config import DEFAULT_IMPORT_RETRIES  # noqa: E402
from paper_import.manifest import (  # noqa: E402
    build_resume_manifest,
    build_resume_paths,
    build_resume_run_key,
    load_resume_manifest,
    save_resume_manifest,
    summarize_manifest,
)
from paper_import.pipeline import import_staged_papers, stage_downloads  # noqa: E402


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

    papers = []
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

        manifest = build_resume_manifest(
            args, papers, root_dir, manifest if not args.refresh_papers else None
        )
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
