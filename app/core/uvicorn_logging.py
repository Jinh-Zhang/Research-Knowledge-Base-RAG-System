import logging


class _IgnoreHealthAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        path = _extract_request_path(record)
        if not path:
            return True
        normalized_path = path.split("?", 1)[0].rstrip("/") or "/"
        return normalized_path != "/health"


def _extract_request_path(record: logging.LogRecord) -> str:
    args = record.args
    if isinstance(args, tuple) and len(args) >= 3:
        path = args[2]
        if isinstance(path, str):
            return path

    message = record.getMessage()
    for marker in ('"GET ', '"POST ', '"PUT ', '"DELETE ', '"PATCH ', '"OPTIONS ', '"HEAD '):
        if marker not in message:
            continue
        tail = message.split(marker, 1)[1]
        return tail.split(" ", 1)[0]

    return ""


def suppress_uvicorn_health_access_logs() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(f, _IgnoreHealthAccessFilter) for f in access_logger.filters):
        return
    access_logger.addFilter(_IgnoreHealthAccessFilter())
