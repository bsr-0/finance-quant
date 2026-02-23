"""Async processing pool for concurrent data operations."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncWorkerPool:
    """Manage async workers for I/O bound operations."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_workers)

    async def run_parallel(
        self, func: Callable[..., T], items: list[Any], *args, **kwargs
    ) -> list[T]:
        """Run function in parallel for all items."""
        loop = asyncio.get_event_loop()

        async def _run_with_semaphore(item):
            async with self._semaphore:
                return await loop.run_in_executor(
                    self._executor, lambda: func(item, *args, **kwargs)
                )

        tasks = [_run_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        successful = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
            else:
                successful.append(result)

        return successful

    async def run_async_tasks(self, tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
        """Run async tasks with concurrency limiting."""
        sem = asyncio.Semaphore(self.max_workers)

        async def _bounded_task(task):
            async with sem:
                return await task

        bounded_tasks = [_bounded_task(t) for t in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        successful = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async task failed: {result}")
            else:
                successful.append(result)

        return successful

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


# Global pool instance
_pool: AsyncWorkerPool | None = None


def get_async_pool(max_workers: int = 10) -> AsyncWorkerPool:
    """Get or create global async pool."""
    global _pool
    if _pool is None:
        _pool = AsyncWorkerPool(max_workers=max_workers)
    return _pool
