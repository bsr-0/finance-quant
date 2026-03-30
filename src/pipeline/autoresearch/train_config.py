"""Mutable experiment config — the agent's sandbox.

The LLM agent proposes changes to the ``TrainConfig`` returned by
``load_config()``.  The runner serialises the config to JSON, sends it to
the agent, receives a modified version, and writes it back via
``save_config()``.

Everything in this file is fair game for the agent to change:
  - model family & hyperparameters
  - feature column selection
  - walk-forward window sizes
  - target column

The *evaluator* (``prepare.py``) is never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.model_search import ModelSpec

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("data/autoresearch/train_config.json")


@dataclass
class TrainConfig:
    """Everything the agent can change about an experiment."""

    # Model
    model_family: str = "ridge"
    hyperparameters: dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})

    # Features — None means "use all available columns"
    feature_cols: list[str] | None = None

    # Walk-forward params
    train_size: int = 252
    test_size: int = 63
    embargo_size: int = 5
    expanding: bool = True

    # Target
    target_col: str = "fwd_return_1d"

    # Agent notes (free-form hypothesis description)
    hypothesis: str = ""

    def to_model_spec(self) -> ModelSpec:
        """Convert to a ModelSpec for the evaluator."""
        return ModelSpec(
            model_family=self.model_family,
            hyperparameters=dict(self.hyperparameters),
            feature_cols=list(self.feature_cols) if self.feature_cols else None,
            train_window=self.train_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_config(path: Path | None = None) -> TrainConfig:
    """Load config from disk, or return defaults if missing."""
    path = path or CONFIG_PATH
    if path.exists():
        with open(path) as f:
            return TrainConfig.from_dict(json.load(f))
    return TrainConfig()


def save_config(config: TrainConfig, path: Path | None = None) -> Path:
    """Persist config to disk."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Saved train config to %s", path)
    return path
