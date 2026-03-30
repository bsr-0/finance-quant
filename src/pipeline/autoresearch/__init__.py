"""AutoResearch integration — autonomous ML experiment loop.

Adapts Karpathy's AutoResearch pattern for quantitative finance:

* ``prepare.py``  — immutable evaluator (walk-forward Sharpe).
* ``train_config.py`` — mutable experiment config the LLM agent edits.
* ``runner.py`` — orchestration loop: propose → run → keep/revert.
"""

from pipeline.autoresearch.prepare import evaluate
from pipeline.autoresearch.runner import AutoResearchRunner
from pipeline.autoresearch.train_config import TrainConfig, load_config, save_config

__all__ = [
    "AutoResearchRunner",
    "TrainConfig",
    "evaluate",
    "load_config",
    "save_config",
]
