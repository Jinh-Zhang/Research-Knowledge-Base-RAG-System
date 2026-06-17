"""Pure text helpers: whitespace normalization, HTML stripping, query matching.

These functions have no network or filesystem side effects.
"""

import re
from html import unescape


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


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text or "", flags=re.S)
    return normalize_space(unescape(text))


def text_matches_query(text: str, query: str) -> bool:
    query = normalize_space(query).lower()
    text = normalize_space(text).lower()
    if not query:
        return True

    lowered = query
    for prefix in ("all:", "ti:", "abs:", "cat:", "au:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break

    lowered = lowered.strip().strip('"').strip("'")
    lowered = normalize_space(lowered)
    if not lowered:
        return True

    tokens = [token for token in re.split(r"\s+", lowered) if token]
    if not tokens:
        return True

    return lowered in text or all(token in text for token in tokens)


def title_matches_query(title: str, query: str) -> bool:
    return text_matches_query(title, query)


def extract_year_text(text: str) -> str:
    match = re.search(r"(?<!\d)(20[0-4]\d)(?!\d)", text or "")
    return match.group(1) if match else ""


def split_authors_text(authors_text: str, pattern: str = r",") -> list:
    return [
        normalize_space(author)
        for author in re.split(pattern, authors_text or "")
        if normalize_space(author)
    ]
