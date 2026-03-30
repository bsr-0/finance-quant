"""AutoResearch runner — the main experiment loop.

Implements the closed-loop optimisation cycle:

1. Load current config + experiment history
2. Send to LLM agent with program.md instructions
3. Receive proposed config change
4. Run immutable evaluator (prepare.evaluate)
5. If improved → keep; otherwise → revert
6. Log result and repeat
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.autoresearch.prepare import EvalResult, evaluate
from pipeline.autoresearch.train_config import TrainConfig, load_config, save_config
from pipeline.experiment_registry import ExperimentRegistry
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

PROGRAM_PATH = Path(__file__).parent / "program.md"
RESULTS_LOG_PATH = Path("data/autoresearch/results.tsv")


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------


@dataclass
class ExperimentEntry:
    """One row in the results log."""

    experiment_number: int
    timestamp: str
    hypothesis: str
    config_summary: str
    primary_metric: float
    status: str  # "keep" | "revert" | "error"
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None


def _load_history(path: Path | None = None) -> list[ExperimentEntry]:
    """Load experiment history from TSV."""
    path = path or RESULTS_LOG_PATH
    if not path.exists():
        return []
    entries: list[ExperimentEntry] = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            entries.append(
                ExperimentEntry(
                    experiment_number=int(parts[0]),
                    timestamp=parts[1],
                    hypothesis=parts[2],
                    config_summary=parts[3],
                    primary_metric=float(parts[4]),
                    status=parts[5],
                    elapsed_seconds=float(parts[6]),
                    error=parts[7] if len(parts) > 7 else None,
                )
            )
    return entries


def _append_result(entry: ExperimentEntry, path: Path | None = None) -> None:
    """Append a single result to the TSV log."""
    path = path or RESULTS_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write(
                "experiment_number\ttimestamp\thypothesis\tconfig_summary\t"
                "primary_metric\tstatus\telapsed_seconds\terror\n"
            )
        error_str = entry.error or ""
        f.write(
            f"{entry.experiment_number}\t{entry.timestamp}\t{entry.hypothesis}\t"
            f"{entry.config_summary}\t{entry.primary_metric:.6f}\t{entry.status}\t"
            f"{entry.elapsed_seconds:.1f}\t{error_str}\n"
        )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def _build_prompt(
    program: str,
    current_config: TrainConfig,
    best_metric: float,
    history: list[ExperimentEntry],
    available_features: list[str],
) -> str:
    """Construct the prompt sent to the LLM agent."""
    history_text = "No experiments yet." if not history else ""
    if history:
        recent = history[-20:]  # last 20 experiments
        lines = []
        for e in recent:
            lines.append(
                f"  #{e.experiment_number} [{e.status}] metric={e.primary_metric:.4f} "
                f"| {e.hypothesis[:80]}"
            )
        history_text = "\n".join(lines)

    return f"""{program}

## Current State

Best metric so far (neg_sharpe, lower=better): {best_metric:.6f}

### Current Config
```json
{json.dumps(current_config.to_dict(), indent=2)}
```

### Available Feature Columns
{json.dumps(available_features[:100])}

### Experiment History (most recent)
{history_text}

## Your Task

Propose a SINGLE modification to the config above that you believe will
improve the metric. Return ONLY the complete JSON config object.
"""


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call the Anthropic API and return the assistant's text response."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _parse_config_response(text: str) -> TrainConfig:
    """Parse the LLM's JSON response into a TrainConfig."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    return TrainConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class AutoResearchRunner:
    """Orchestrates the autonomous experiment loop.

    Usage::

        runner = AutoResearchRunner(df=feature_df)
        runner.run(max_experiments=100)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config_path: Path | None = None,
        results_path: Path | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_experiments: int = 100,
    ) -> None:
        self.df = df
        self.config_path = config_path
        self.results_path = results_path
        self.model = model
        self.max_experiments = max_experiments

        settings = get_settings()
        self.api_key = api_key or (settings.autoresearch.api_key if settings.autoresearch else None)
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. Set it in .env or pass api_key= directly."
            )

        self.program = PROGRAM_PATH.read_text()
        self.available_features = [c for c in df.columns if c != "fwd_return_1d"]

        self.config = load_config(config_path)
        self.history = _load_history(results_path)
        self.best_metric = float("inf")
        self.best_config = TrainConfig(**asdict(self.config))
        self.registry = ExperimentRegistry(
            storage_path=Path("data/autoresearch/experiment_registry.json")
        )

        # Initialise best metric from history
        keeps = [e for e in self.history if e.status == "keep"]
        if keeps:
            self.best_metric = min(e.primary_metric for e in keeps)

    def _run_one(self, experiment_number: int) -> ExperimentEntry:
        """Execute a single experiment cycle."""
        logger.info("=== Experiment #%d ===", experiment_number)

        # 1. Ask the LLM for a proposed config
        prompt = _build_prompt(
            program=self.program,
            current_config=self.config,
            best_metric=self.best_metric,
            history=self.history,
            available_features=self.available_features,
        )

        try:
            response_text = _call_anthropic(prompt, self.api_key, self.model)
            proposed = _parse_config_response(response_text)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ExperimentEntry(
                experiment_number=experiment_number,
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis="LLM call failed",
                config_summary="N/A",
                primary_metric=0.0,
                status="error",
                error=str(exc),
            )

        config_summary = f"{proposed.model_family} | {proposed.hyperparameters}"
        logger.info("Hypothesis: %s", proposed.hypothesis[:100])
        logger.info("Config: %s", config_summary[:100])

        # 2. Run the immutable evaluator
        result: EvalResult = evaluate(
            df=self.df,
            model_spec=proposed.to_model_spec(),
            target_col=proposed.target_col,
            train_size=proposed.train_size,
            test_size=proposed.test_size,
            embargo_size=proposed.embargo_size,
            expanding=proposed.expanding,
        )

        if result.error:
            logger.warning("Evaluation error: %s", result.error)
            return ExperimentEntry(
                experiment_number=experiment_number,
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis=proposed.hypothesis,
                config_summary=config_summary[:200],
                primary_metric=0.0,
                status="error",
                elapsed_seconds=result.elapsed_seconds,
                error=result.error,
            )

        # 3. Keep or revert
        improved = result.primary_metric < self.best_metric
        status = "keep" if improved else "revert"

        if improved:
            logger.info(
                "KEEP: %.6f < %.6f (improved by %.4f)",
                result.primary_metric,
                self.best_metric,
                self.best_metric - result.primary_metric,
            )
            self.best_metric = result.primary_metric
            self.best_config = proposed
            self.config = proposed
            save_config(self.config, self.config_path)
        else:
            logger.info(
                "REVERT: %.6f >= %.6f (no improvement)",
                result.primary_metric,
                self.best_metric,
            )

        # 4. Log to experiment registry
        exp = self.registry.create_experiment(
            problem_id="autoresearch",
            model_family=proposed.model_family,
            hyperparameters=proposed.hyperparameters,
            primary_metric="neg_sharpe",
            primary_metric_value=result.primary_metric,
            secondary_metrics=result.secondary_metrics,
            notes=f"[{status}] {proposed.hypothesis}",
        )
        if improved:
            self.registry.complete_experiment(
                exp.experiment_id,
                primary_metric="neg_sharpe",
                primary_metric_value=result.primary_metric,
                secondary_metrics=result.secondary_metrics,
                compute_cost_seconds=result.elapsed_seconds,
            )
        else:
            self.registry.reject_experiment(exp.experiment_id, reason="No improvement")

        return ExperimentEntry(
            experiment_number=experiment_number,
            timestamp=datetime.now(UTC).isoformat(),
            hypothesis=proposed.hypothesis,
            config_summary=config_summary[:200],
            primary_metric=result.primary_metric,
            status=status,
            secondary_metrics=result.secondary_metrics,
            elapsed_seconds=result.elapsed_seconds,
        )

    def run(self, max_experiments: int | None = None) -> list[ExperimentEntry]:
        """Run the full experiment loop.

        Parameters
        ----------
        max_experiments : int, optional
            Override the instance default.

        Returns
        -------
        list[ExperimentEntry]
            All experiment results from this run.
        """
        n = max_experiments or self.max_experiments
        start_num = len(self.history) + 1
        results: list[ExperimentEntry] = []

        logger.info("Starting AutoResearch loop: %d experiments planned", n)
        logger.info("Best metric so far: %.6f", self.best_metric)

        for i in range(n):
            experiment_number = start_num + i
            entry = self._run_one(experiment_number)
            results.append(entry)
            self.history.append(entry)
            _append_result(entry, self.results_path)

            logger.info(
                "Experiment #%d: %s (metric=%.6f, elapsed=%.1fs)",
                experiment_number,
                entry.status,
                entry.primary_metric,
                entry.elapsed_seconds,
            )

            # Brief pause between experiments to avoid API rate limits
            if i < n - 1:
                time.sleep(1)

        # Summary
        keeps = [e for e in results if e.status == "keep"]
        logger.info(
            "AutoResearch complete: %d experiments, %d improvements, best=%.6f",
            len(results),
            len(keeps),
            self.best_metric,
        )

        return results
