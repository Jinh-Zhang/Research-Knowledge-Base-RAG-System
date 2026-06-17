"""HTTP session construction and network-error classification."""

from http.client import RemoteDisconnected

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    DEFAULT_USER_AGENT,
    REQUEST_RETRY_BACKOFF,
    REQUEST_RETRY_TOTAL,
)
from .text_utils import normalize_space


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
