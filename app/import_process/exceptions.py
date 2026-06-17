"""Typed exceptions for the paper-import pipeline.

Historically the import nodes raised plain ``RuntimeError`` / ``TimeoutError``
with a human-readable (often Chinese) message, and downstream code recovered
the *reason* by string-matching that message. That is fragile: messages drift,
languages change, and mojibake breaks matching.

These typed exceptions let nodes declare the failure *category* at the raise
site. Downstream classification can then use ``isinstance`` instead of string
matching. Nodes can be migrated incrementally; until a raise site is migrated,
the classifier falls back to regex on the message.

``ParseTimeoutError`` also subclasses the builtin ``TimeoutError`` so existing
``isinstance(exc, TimeoutError)`` checks keep working during migration.
"""


class PaperImportError(Exception):
    """Base class for all categorized paper-import failures."""


class ParseTimeoutError(PaperImportError, TimeoutError):
    """PDF parsing (e.g. MinerU polling) exceeded its time budget. Retryable."""


class ParseFailureError(PaperImportError):
    """The parsing backend reported a task failure. Retryable."""


class TextExtractionError(PaperImportError):
    """Parsing produced no usable text. Fall back to metadata-only import."""


class PdfContentError(PaperImportError):
    """The PDF is broken/unsupported. Fall back to metadata-only import."""
