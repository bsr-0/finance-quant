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

# ---------------------------------------------------------------------------
# Valid options (reference for the agent — enforced in validate())
# ---------------------------------------------------------------------------

VALID_MODEL_FAMILIES = [
    "ridge",
    "lasso",
    "logistic",
    "random_forest",
    "gradient_boosting",
    "lightgbm",
    "xgboost",
]

# Hyperparameter ranges the agent can explore
HYPERPARAMETER_HINTS: dict[str, dict[str, list[Any]]] = {
    "ridge": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "logistic": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [500, 1000, 2000],
        "solver": ["lbfgs", "saga"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_leaf": [5, 10, 20, 50],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "min_samples_leaf": [5, 10, 20],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    },
}

# Feature groups available in the dataset
FEATURE_GROUPS: dict[str, list[str]] = {
    "price": [
        "price_latest",
        "price_change_1d",
        "price_change_7d",
        "volume_avg_20d",
        "volatility_20d",
    ],
    "technical": [
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "momentum_10",
        "roc_10",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_position",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "obv",
        "volume_sma_20",
    ],
    "seasonal": [
        "day_of_week",
        "month",
        "quarter",
        "week_of_year",
        "is_month_end",
        "is_quarter_end",
        "day_of_year",
    ],
    "macro": [
        "GDP",
        "UNRATE",
        "CPIAUCSL",
        "FEDFUNDS",
        "T10Y2Y",
        "VIXCLS",
        "DGS10",
        "DGS2",
        "TB3MS",
        "BAMLH0A0HYM2",
        "BAMLC0A4CBBB",
        "HOUST",
        "ICSA",
        "PAYEMS",
        "M2SL",
        "DCOILWTICO",
        "DTWEXBGS",
        "NFCI",
        "USSLIND",
        "T5YIE",
        "T10YIE",
    ],
    "fundamentals": [
        "pe_ratio",
        "pb_ratio",
        "debt_to_equity",
        "roe",
    ],
    "options": [
        "iv_30d",
        "put_call_volume_ratio",
        "skew_25d",
    ],
    "sentiment": [
        "insider_net_shares_90d",
        "insider_buy_count_90d",
        "short_interest_ratio",
    ],
    "positioning": [
        "cot_noncommercial_net",
        "cot_commercial_net",
        "cot_noncommercial_pct_oi",
    ],
    "events": [
        "days_to_next_earnings",
        "last_eps_surprise_pct",
        "institutional_holders_count",
    ],
}

ALL_FEATURES: list[str] = []
for _group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(_group)


@dataclass
class TrainConfig:
    """Everything the agent can change about an experiment."""

    # Model
    model_family: str = "ridge"
    hyperparameters: dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})

    # Features — None means "use all available columns"; list = explicit selection
    # The agent can also specify feature_groups instead of individual columns
    feature_cols: list[str] | None = None
    feature_groups: list[str] | None = None  # e.g. ["price", "technical", "macro"]

    # Walk-forward params
    train_size: int = 252  # ~1 year of trading days
    test_size: int = 63  # ~1 quarter
    embargo_size: int = 5
    expanding: bool = True

    # Target
    target_col: str = "fwd_return_1d"

    # Agent notes (free-form hypothesis description)
    hypothesis: str = ""

    def resolve_features(self, available_cols: list[str]) -> list[str] | None:
        """Resolve feature_groups and feature_cols into a final feature list.

        Priority: feature_cols > feature_groups > None (use all).
        Only returns columns that actually exist in the dataset.
        """
        if self.feature_cols is not None:
            return [c for c in self.feature_cols if c in available_cols]

        if self.feature_groups is not None:
            cols = []
            for group in self.feature_groups:
                if group in FEATURE_GROUPS:
                    cols.extend(FEATURE_GROUPS[group])
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = []
            for c in cols:
                if c not in seen and c in available_cols:
                    seen.add(c)
                    unique.append(c)
            return unique if unique else None

        return None  # use all

    def to_model_spec(self, available_cols: list[str] | None = None) -> ModelSpec:
        """Convert to a ModelSpec for the evaluator."""
        features = None
        if available_cols is not None:
            features = self.resolve_features(available_cols)
        elif self.feature_cols is not None:
            features = list(self.feature_cols)
        return ModelSpec(
            model_family=self.model_family,
            hyperparameters=dict(self.hyperparameters),
            feature_cols=features,
            train_window=self.train_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if self.model_family not in VALID_MODEL_FAMILIES:
            errors.append(
                f"Invalid model_family '{self.model_family}'. "
                f"Must be one of: {VALID_MODEL_FAMILIES}"
            )
        if self.train_size < 126:
            errors.append(f"train_size {self.train_size} < 126 minimum (6 months)")
        if self.test_size < 21:
            errors.append(f"test_size {self.test_size} < 21 minimum (1 month)")
        if self.embargo_size < 3:
            errors.append(f"embargo_size {self.embargo_size} < 3 minimum")
        if self.feature_groups is not None:
            invalid = [g for g in self.feature_groups if g not in FEATURE_GROUPS]
            if invalid:
                errors.append(
                    f"Invalid feature_groups: {invalid}. "
                    f"Must be from: {list(FEATURE_GROUPS.keys())}"
                )
        return errors


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
