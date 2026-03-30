"""AutoResearch integration — autonomous ML experiment loop.

Adapts Karpathy's AutoResearch pattern for quantitative finance:

* ``prepare.py``  — immutable evaluator (composite: Sharpe + drawdown penalty).
* ``train_config.py`` — mutable experiment config the LLM agent edits.
* ``runner.py`` — orchestration loop: propose → validate → evaluate → keep/revert.
* ``program.md`` — domain-specific research instructions for the LLM agent.
"""

from pipeline.autoresearch.prepare import EvalResult, evaluate
from pipeline.autoresearch.runner import AutoResearchRunner
from pipeline.autoresearch.train_config import (
    ALL_FEATURES,
    FEATURE_GROUPS,
    HYPERPARAMETER_HINTS,
    VALID_MODEL_FAMILIES,
    TrainConfig,
    load_config,
    save_config,
)

__all__ = [
    "ALL_FEATURES",
    "AutoResearchRunner",
    "EvalResult",
    "FEATURE_GROUPS",
    "HYPERPARAMETER_HINTS",
    "TrainConfig",
    "VALID_MODEL_FAMILIES",
    "evaluate",
    "load_config",
    "save_config",
]
