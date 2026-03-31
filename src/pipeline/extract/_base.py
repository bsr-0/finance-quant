"""Base mixin for extractors that use httpx.Client."""

from __future__ import annotations

import contextlib
from typing import Any


class HttpClientMixin:
    """Adds proper resource management for extractors using httpx.Client.

    Provides ``close()``, context-manager support (``__enter__``/``__exit__``),
    and a ``__del__`` safety-net so that the underlying ``httpx.Client`` is
    always cleaned up — even when callers forget to call ``close()`` or use a
    ``with`` block.
    """

    client: Any  # httpx.Client, typed as Any to avoid import dependency

    def close(self) -> None:
        """Explicitly release the underlying HTTP connection pool."""
        self.client.close()

    def __enter__(self) -> HttpClientMixin:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()
