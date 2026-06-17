"""Import-error classification (hybrid: typed exceptions + regex fallback).

Preferred path: import nodes raise the typed exceptions from
``app.import_process.exceptions``; classification is then a plain
``isinstance`` check and is fully reliable.

Fallback path: for not-yet-migrated raise sites, third-party exceptions, or
mojibake-corrupted messages, a single data-driven rule table matches the
normalized message text. All the fragile string knowledge lives in ONE table
(:data:`_MESSAGE_RULES`) instead of scattered ``is_xxx`` functions.

Public API (consumed by pipeline.py):
    - is_retryable_import_error(exc) -> bool
    - should_fallback_to_metadata_import(exc) -> bool
"""

import re
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Type

from app.import_process.exceptions import (
    ParseFailureError,
    ParseTimeoutError,
    PdfContentError,
    TextExtractionError,
)

from .text_utils import normalize_space


class ErrorCategory(Enum):
    """Why an import attempt failed, abstracted away from message wording."""

    TIMEOUT = auto()          # parsing exceeded its time budget
    PARSE_FAILURE = auto()    # parsing backend reported a task failure
    MISSING_TEXT = auto()     # parsing produced no usable text
    PDF_CONTENT = auto()      # the PDF itself is broken/unsupported


# Typed exception -> category. Checked first; authoritative when it matches.
_TYPE_RULES: Dict[Type[BaseException], ErrorCategory] = {
    ParseTimeoutError: ErrorCategory.TIMEOUT,
    ParseFailureError: ErrorCategory.PARSE_FAILURE,
    TextExtractionError: ErrorCategory.MISSING_TEXT,
    PdfContentError: ErrorCategory.PDF_CONTENT,
}

# Category -> compiled regex matched against the normalized message. Fallback
# only; used when no typed rule matches. Patterns are case-insensitive.
_MESSAGE_RULES: List[Tuple[ErrorCategory, "re.Pattern[str]"]] = [
    (
        ErrorCategory.TIMEOUT,
        re.compile(r"timeout|timed?\s*out|超时"),
    ),
    (
        ErrorCategory.PARSE_FAILURE,
        re.compile(r"parsing failed|retry limit reached|task failed|解析(任务)?失败|任务轮询"),
    ),
    (
        ErrorCategory.MISSING_TEXT,
        re.compile(r"no text|empty text|未能从文件中提取文本|提取文本"),
    ),
    (
        ErrorCategory.PDF_CONTENT,
        re.compile(
            r"please replace the file|unsupported pdf|file is (broken|damaged)|pdf兜底解析失败"
        ),
    ),
]


def _normalize_error_text(exc: Exception) -> str:
    """Lowercase the message, plus a gbk->utf-8 repair pass for mojibake."""
    message = normalize_space(str(exc))
    if not message:
        return ""

    parts = [message.lower()]
    try:
        repaired = message.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        repaired = ""
    repaired = normalize_space(repaired).lower()
    if repaired:
        parts.append(repaired)
    return " ".join(parts)


def classify_error(exc: Exception) -> Optional[ErrorCategory]:
    """Return the :class:`ErrorCategory` for ``exc``, or ``None`` if unknown.

    Typed exceptions win; message regex is the fallback for legacy/wrapped
    exceptions.
    """
    for exc_type, category in _TYPE_RULES.items():
        if isinstance(exc, exc_type):
            return category

    # Builtin TimeoutError (e.g. raised by not-yet-migrated nodes) is a clear
    # timeout signal without needing the message.
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT

    message = _normalize_error_text(exc)
    if not message:
        return None
    for category, pattern in _MESSAGE_RULES:
        if pattern.search(message):
            return category
    return None


# Categories that justify retrying the import as-is.
_RETRYABLE_CATEGORIES = {ErrorCategory.TIMEOUT, ErrorCategory.PARSE_FAILURE}

# Categories that justify giving up on the PDF and importing metadata only.
_FALLBACK_CATEGORIES = {
    ErrorCategory.TIMEOUT,
    ErrorCategory.PARSE_FAILURE,
    ErrorCategory.MISSING_TEXT,
    ErrorCategory.PDF_CONTENT,
}


def is_retryable_import_error(exc: Exception) -> bool:
    return classify_error(exc) in _RETRYABLE_CATEGORIES


def should_fallback_to_metadata_import(exc: Exception) -> bool:
    return classify_error(exc) in _FALLBACK_CATEGORIES
