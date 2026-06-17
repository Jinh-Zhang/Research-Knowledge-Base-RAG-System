"""Crawler registry.

Each source module registers its searcher with the ``@register("name")``
decorator. Importing this package triggers the per-source modules to register
themselves, so :data:`SOURCE_SEARCHERS` is populated as a side effect.

Adding a new source means dropping a new module in this package and decorating
its searcher; no central table needs editing.
"""

from typing import Any, Callable, Dict, List

SOURCE_SEARCHERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {}


def register(name: str) -> Callable:
    """Register a searcher function under ``name`` in :data:`SOURCE_SEARCHERS`."""

    def decorator(fn: Callable) -> Callable:
        SOURCE_SEARCHERS[name] = fn
        return fn

    return decorator


# Import side effects populate SOURCE_SEARCHERS. Keep this after `register`
# is defined so the modules can import it.
from . import (  # noqa: E402,F401
    aaai,
    acl,
    arxiv,
    colm,
    cvf,
    iclr,
    icml,
    ijcai,
    neurips,
    openreview,
)

__all__ = ["SOURCE_SEARCHERS", "register"]
