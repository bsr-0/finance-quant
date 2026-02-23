"""Checkpointing for long-running operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage checkpoints for resumable operations."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, operation_id: str) -> Path:
        """Get path for checkpoint file."""
        return self.checkpoint_dir / f"{operation_id}.json"

    def save_checkpoint(self, operation_id: str, state: dict, metadata: dict | None = None) -> Path:
        """Save checkpoint state."""
        checkpoint = {
            "operation_id": operation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "metadata": metadata or {},
        }

        checkpoint_path = self._get_checkpoint_path(operation_id)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, operation_id: str) -> dict | None:
        """Load checkpoint state if exists."""
        checkpoint_path = self._get_checkpoint_path(operation_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, operation_id: str) -> bool:
        """Delete checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(operation_id)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        return False

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint operation IDs."""
        return [f.stem for f in self.checkpoint_dir.glob("*.json")]

    def get_progress(self, operation_id: str) -> dict | None:
        """Get progress information from checkpoint."""
        checkpoint = self.load_checkpoint(operation_id)
        if not checkpoint:
            return None

        state = checkpoint.get("state", {})
        return {
            "timestamp": checkpoint.get("timestamp"),
            "completed_items": state.get("completed_items", 0),
            "total_items": state.get("total_items", 0),
            "last_processed": state.get("last_processed"),
        }


class CheckpointContext:
    """Context manager for checkpoint-aware operations."""

    def __init__(
        self, checkpoint_manager: CheckpointManager, operation_id: str, resume: bool = True
    ):
        self.checkpoint_manager = checkpoint_manager
        self.operation_id = operation_id
        self.resume = resume
        self.state: dict = {}
        self._initial_state: dict | None = None

    def __enter__(self):
        """Enter context and load checkpoint if resuming."""
        if self.resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(self.operation_id)
            if checkpoint:
                self._initial_state = checkpoint.get("state", {})
                self.state = self._initial_state.copy()
                logger.info(f"Resuming operation {self.operation_id} from checkpoint")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and save checkpoint if not complete."""
        if exc_type is not None:
            # Exception occurred - save checkpoint for retry
            self.checkpoint_manager.save_checkpoint(
                self.operation_id, self.state, {"error": str(exc_val)}
            )
            logger.warning(f"Operation {self.operation_id} failed, checkpoint saved")
        else:
            # Success - clean up checkpoint
            self.checkpoint_manager.delete_checkpoint(self.operation_id)
            logger.info(f"Operation {self.operation_id} completed successfully")

        return False  # Don't suppress exceptions

    def update(self, **kwargs):
        """Update checkpoint state."""
        self.state.update(kwargs)
        # Auto-save every N updates could be added here

    def save(self):
        """Manually save checkpoint."""
        self.checkpoint_manager.save_checkpoint(self.operation_id, self.state)

    def get_resumed_position(self) -> Any | None:
        """Get position to resume from."""
        if self._initial_state:
            return self._initial_state.get("last_processed")
        return None
