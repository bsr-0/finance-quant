"""Checkpoint helpers for resumable extraction loops."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from pipeline.infrastructure.checkpoint import CheckpointContext, CheckpointManager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_checkpoint_manager() -> CheckpointManager:
    """Return a CheckpointManager using the configured checkpoint directory."""
    settings = get_settings()
    return CheckpointManager(settings.infrastructure.checkpoint_dir)


def make_operation_id(source_name: str) -> str:
    """Build a deterministic operation ID for an extraction source.

    Uses only the source name so that re-running the same CLI command
    (which generates a new run_id) still finds the existing checkpoint.
    """
    return f"backfill_{source_name}"


def resumable_items(
    items: Sequence[T],
    ctx: CheckpointContext,
    key_fn: Callable[[T], str] | None = None,
) -> Iterable[tuple[int, T]]:
    """Yield ``(index, item)`` pairs, skipping already-completed items.

    Args:
        items: The full ordered list of items to process.
        ctx: An already-entered ``CheckpointContext``.
        key_fn: Extract a string key from an item. Defaults to ``str()``.

    Yields:
        ``(index, item)`` for each item not yet completed.
    """
    completed: set[str] = set(ctx.state.get("completed_keys", []))
    kfn = key_fn or str

    if completed:
        logger.info("Resuming: skipping %d already-completed items", len(completed))

    for i, item in enumerate(items):
        if kfn(item) in completed:
            continue
        yield i, item


def mark_item_done(
    ctx: CheckpointContext, item_key: str, index: int, total: int
) -> None:
    """Record that *item_key* has been processed and persist the checkpoint."""
    completed: list[str] = ctx.state.get("completed_keys", [])
    completed.append(item_key)
    ctx.update(
        completed_keys=completed,
        completed_items=len(completed),
        total_items=total,
        last_processed=item_key,
    )
    ctx.save()
