"""AutoResearch runner — the main experiment loop.

Implements the closed-loop optimisation cycle:

1. Load current config + experiment history
2. Send to LLM agent with program.md instructions
3. Receive proposed config change
4. Validate the config (model family, constraints)
5. Resolve features against the actual dataset columns
6. Run immutable evaluator (prepare.evaluate)
7. If improved → keep; otherwise → revert
8. Log result with full metrics and repeat
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.autoresearch.prepare import EvalResult, evaluate
from pipeline.autoresearch.train_config import (
    FEATURE_GROUPS,
    HYPERPARAMETER_HINTS,
    TrainConfig,
    load_config,
    save_config,
)
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
# Dataset summary (fed to the LLM for context)
# ---------------------------------------------------------------------------


def _dataset_summary(df: pd.DataFrame, target_col: str) -> str:
    """Compact dataset summary for the LLM prompt."""
    lines = [
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        f"Date range: {df.index.min()} to {df.index.max()}",
    ]
    if target_col in df.columns:
        target = df[target_col].dropna()
        lines.append(
            f"Target '{target_col}': mean={target.mean():.6f}, "
            f"std={target.std():.6f}, "
            f"skew={target.skew():.2f}, "
            f"pct_positive={float((target > 0).mean()):.1%}"
        )

    # Feature availability by group
    available = set(df.columns)
    group_coverage = []
    for group_name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in available]
        if present:
            group_coverage.append(f"  {group_name}: {len(present)}/{len(cols)} columns present")
    lines.append("Feature groups in dataset:")
    lines.extend(group_coverage)

    # Missing data summary
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.3].sort_values(ascending=False)
    if len(high_missing) > 0:
        lines.append(f"Columns with >30% missing: {len(high_missing)}")
        for col, pct in high_missing.head(5).items():
            lines.append(f"  {col}: {pct:.0%} missing")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def _format_history(history: list[ExperimentEntry]) -> str:
    """Format experiment history for the prompt, with secondary metrics."""
    if not history:
        return "No experiments yet — this is your first run."

    recent = history[-25:]  # last 25 experiments
    lines = []
    for e in recent:
        sec = e.secondary_metrics
        detail_parts = []
        if sec.get("neg_sharpe") is not None:
            detail_parts.append(f"sharpe={-sec['neg_sharpe']:.3f}")
        if sec.get("sortino") is not None:
            detail_parts.append(f"sortino={sec['sortino']:.3f}")
        if sec.get("max_drawdown") is not None:
            detail_parts.append(f"dd={sec['max_drawdown']:.4f}")
        if sec.get("hit_rate") is not None:
            detail_parts.append(f"hit={sec['hit_rate']:.1%}")
        detail = ", ".join(detail_parts)

        lines.append(
            f"  #{e.experiment_number} [{e.status:6s}] composite={e.primary_metric:+.4f} "
            f"({detail}) | {e.hypothesis[:70]}"
        )

    # Stats summary
    keeps = [e for e in history if e.status == "keep"]
    reverts = [e for e in history if e.status == "revert"]
    errors = [e for e in history if e.status == "error"]
    lines.append(
        f"\n  Total: {len(history)} experiments | "
        f"{len(keeps)} kept, {len(reverts)} reverted, {len(errors)} errors"
    )
    return "\n".join(lines)


def _build_prompt(
    program: str,
    current_config: TrainConfig,
    best_metric: float,
    best_secondary: dict[str, float],
    history: list[ExperimentEntry],
    available_features: list[str],
    dataset_summary: str,
) -> str:
    """Construct the prompt sent to the LLM agent."""
    # Best metrics breakdown
    best_parts = []
    if best_secondary:
        if "neg_sharpe" in best_secondary:
            best_parts.append(f"Sharpe: {-best_secondary['neg_sharpe']:.4f}")
        if "sortino" in best_secondary:
            best_parts.append(f"Sortino: {best_secondary['sortino']:.4f}")
        if "max_drawdown" in best_secondary:
            best_parts.append(f"Max DD: {best_secondary['max_drawdown']:.4f}")
        if "hit_rate" in best_secondary:
            best_parts.append(f"Hit Rate: {best_secondary['hit_rate']:.1%}")
    best_breakdown = " | ".join(best_parts) if best_parts else "N/A"

    return f"""{program}

## Current State

**Best composite score (lower = better): {best_metric:.6f}**
Best metrics breakdown: {best_breakdown}

### Dataset Summary
{dataset_summary}

### Current Best Config
```json
{json.dumps(current_config.to_dict(), indent=2)}
```

### Available Feature Columns ({len(available_features)} total)
{json.dumps(available_features, indent=2)}

### Hyperparameter Reference
{json.dumps(HYPERPARAMETER_HINTS, indent=2)}

### Experiment History (most recent)
{_format_history(history)}

## Your Task

Propose a SINGLE modification to the config above that you believe will
improve the composite score. Return ONLY the complete JSON config object.
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
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Find the JSON object boundaries (handle leading/trailing text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

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
        self.api_key = api_key or (
            settings.autoresearch.api_key if settings.autoresearch else None
        )
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. Set it in .env or pass api_key= directly."
            )

        self.program = PROGRAM_PATH.read_text()
        self.target_col = "fwd_return_1d"
        self.available_features = [c for c in df.columns if c != self.target_col]
        self.dataset_summary = _dataset_summary(df, self.target_col)

        self.config = load_config(config_path)
        self.history = _load_history(results_path)
        self.best_metric = float("inf")
        self.best_secondary: dict[str, float] = {}
        self.best_config = TrainConfig(**asdict(self.config))
        self.registry = ExperimentRegistry(
            storage_path=Path("data/autoresearch/experiment_registry.json")
        )

        # Initialise best metric from history
        keeps = [e for e in self.history if e.status == "keep"]
        if keeps:
            best_entry = min(keeps, key=lambda e: e.primary_metric)
            self.best_metric = best_entry.primary_metric
            self.best_secondary = best_entry.secondary_metrics

    def _run_one(self, experiment_number: int) -> ExperimentEntry:
        """Execute a single experiment cycle."""
        logger.info("=== Experiment #%d ===", experiment_number)

        # 1. Ask the LLM for a proposed config
        prompt = _build_prompt(
            program=self.program,
            current_config=self.config,
            best_metric=self.best_metric,
            best_secondary=self.best_secondary,
            history=self.history,
            available_features=self.available_features,
            dataset_summary=self.dataset_summary,
        )

        try:
            response_text = _call_anthropic(prompt, self.api_key, self.model)
            proposed = _parse_config_response(response_text)
        except Exception as exc:
            logger.warning("LLM call / parse failed: %s", exc)
            return ExperimentEntry(
                experiment_number=experiment_number,
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis="LLM call failed",
                config_summary="N/A",
                primary_metric=0.0,
                status="error",
                error=str(exc),
            )

        # 2. Validate the proposed config
        errors = proposed.validate()
        if errors:
            logger.warning("Config validation failed: %s", errors)
            return ExperimentEntry(
                experiment_number=experiment_number,
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis=proposed.hypothesis,
                config_summary=f"INVALID: {errors[0]}",
                primary_metric=0.0,
                status="error",
                error=f"Validation: {'; '.join(errors)}",
            )

        # 3. Resolve features against actual dataset
        model_spec = proposed.to_model_spec(available_cols=self.available_features)
        config_summary = (
            f"{proposed.model_family} | "
            f"features={len(model_spec.feature_cols) if model_spec.feature_cols else 'all'} | "
            f"{proposed.hyperparameters}"
        )
        logger.info("Hypothesis: %s", proposed.hypothesis[:120])
        logger.info("Config: %s", config_summary[:150])

        # 4. Run the immutable evaluator
        result: EvalResult = evaluate(
            df=self.df,
            model_spec=model_spec,
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

        # 5. Keep or revert
        improved = result.primary_metric < self.best_metric
        status = "keep" if improved else "revert"

        sec = result.secondary_metrics
        sharpe = -sec.get("neg_sharpe", 0.0)
        sortino = sec.get("sortino", 0.0)
        max_dd = sec.get("max_drawdown", 0.0)
        hit_rate = sec.get("hit_rate", 0.0)

        if improved:
            logger.info(
                "KEEP: composite=%.4f (was %.4f) | sharpe=%.3f sortino=%.3f dd=%.4f hit=%.1f%%",
                result.primary_metric,
                self.best_metric,
                sharpe,
                sortino,
                max_dd,
                hit_rate * 100,
            )
            self.best_metric = result.primary_metric
            self.best_secondary = dict(sec)
            self.best_config = proposed
            self.config = proposed
            save_config(self.config, self.config_path)
        else:
            logger.info(
                "REVERT: composite=%.4f (best=%.4f) | sharpe=%.3f sortino=%.3f dd=%.4f hit=%.1f%%",
                result.primary_metric,
                self.best_metric,
                sharpe,
                sortino,
                max_dd,
                hit_rate * 100,
            )

        # 6. Log to experiment registry
        exp = self.registry.create_experiment(
            problem_id="autoresearch",
            model_family=proposed.model_family,
            hyperparameters=proposed.hyperparameters,
            primary_metric="composite",
            primary_metric_value=result.primary_metric,
            secondary_metrics=result.secondary_metrics,
            notes=f"[{status}] {proposed.hypothesis}",
        )
        if improved:
            self.registry.complete_experiment(
                exp.experiment_id,
                primary_metric="composite",
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
        logger.info("Dataset: %s", self.dataset_summary.split("\n")[0])

        for i in range(n):
            experiment_number = start_num + i
            entry = self._run_one(experiment_number)
            results.append(entry)
            self.history.append(entry)
            _append_result(entry, self.results_path)

            sec = entry.secondary_metrics
            sharpe_str = f"sharpe={-sec['neg_sharpe']:.3f}" if "neg_sharpe" in sec else ""
            logger.info(
                "Experiment #%d: %s (composite=%.4f %s, %.1fs)",
                experiment_number,
                entry.status,
                entry.primary_metric,
                sharpe_str,
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
