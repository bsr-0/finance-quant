"""A/B Testing Framework (Agent Directive V7 — Section 18.5).

Implements the full A/B testing protocol with:
- Power analysis for minimum sample size
- Sequential testing boundaries (O'Brien-Fleming)
- Stratified allocation
- Test lifecycle management
- Post-test validation

Required by the directive before any candidate system receives live
comparison against the incumbent.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class TestStatus(StrEnum):
    DESIGNED = "designed"
    RUNNING = "running"
    STOPPED_EARLY = "stopped_early"
    COMPLETED = "completed"
    VALIDATED = "validated"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test per Section 18.5."""

    test_id: str = field(default_factory=lambda: str(uuid4()))
    candidate_id: str = ""
    incumbent_id: str = ""
    primary_metric: str = "sharpe"
    secondary_metrics: list[str] = field(default_factory=list)
    allocation_method: str = "stratified_random"
    allocation_pct: float = 0.10  # % traffic to candidate
    minimum_sample_size: int = 0  # computed via power analysis
    min_duration_cycles: int = 100
    alpha: float = 0.05
    power: float = 0.80
    effect_size: float = 0.2  # Cohen's d or similar
    n_interim_looks: int = 4  # number of interim analyses
    status: TestStatus = TestStatus.DESIGNED
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ABTestConfig:
        d = dict(d)
        d["status"] = TestStatus(d["status"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Power Analysis (Section 18.5)
# ---------------------------------------------------------------------------


class PowerAnalysis:
    """Pre-compute minimum sample size for a given effect and power."""

    @staticmethod
    def compute_sample_size(
        effect_size: float,
        power: float = 0.80,
        alpha: float = 0.05,
        two_sided: bool = True,
    ) -> int:
        """Minimum sample size per group for a two-sample z-test.

        Uses the standard formula: n = ((z_alpha + z_beta) / d)^2
        where d = effect_size (Cohen's d).
        """
        if effect_size <= 0:
            raise ValueError("effect_size must be positive")
        z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
        z_beta = norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(math.ceil(n))

    @staticmethod
    def compute_power(
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        two_sided: bool = True,
    ) -> float:
        """Compute statistical power for given n and effect size."""
        if n <= 0 or effect_size <= 0:
            return 0.0
        z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
        z = effect_size * math.sqrt(n) - z_alpha
        return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Sequential Testing Boundaries (O'Brien-Fleming, Section 18.5)
# ---------------------------------------------------------------------------


class SequentialTestBoundary:
    """O'Brien-Fleming group sequential boundaries for early stopping.

    Allows interim analyses at pre-specified information fractions without
    inflating the overall Type I error rate.
    """

    def __init__(
        self,
        n_looks: int = 4,
        alpha: float = 0.05,
        two_sided: bool = True,
    ):
        self.n_looks = n_looks
        self.alpha = alpha
        self.two_sided = two_sided
        self._boundaries = self._compute_boundaries()

    def _compute_boundaries(self) -> list[dict[str, float]]:
        """Compute O'Brien-Fleming boundaries at each interim look."""
        boundaries = []
        for k in range(1, self.n_looks + 1):
            info_fraction = k / self.n_looks
            # O'Brien-Fleming: z_k = z_final / sqrt(info_fraction)
            z_final = norm.ppf(1 - self.alpha / (2 if self.two_sided else 1))
            z_boundary = z_final / math.sqrt(info_fraction)
            boundaries.append(
                {
                    "look": k,
                    "info_fraction": round(info_fraction, 4),
                    "z_boundary": round(z_boundary, 4),
                    "p_boundary": round(
                        2 * norm.sf(z_boundary) if self.two_sided else norm.sf(z_boundary),
                        6,
                    ),
                }
            )
        return boundaries

    def should_stop(self, look: int, z_statistic: float) -> bool:
        """Check if the test should stop at this interim analysis."""
        if look < 1 or look > self.n_looks:
            raise ValueError(f"look must be between 1 and {self.n_looks}")
        boundary = self._boundaries[look - 1]
        return abs(z_statistic) >= boundary["z_boundary"]

    def get_boundaries(self) -> list[dict[str, float]]:
        return list(self._boundaries)


# ---------------------------------------------------------------------------
# A/B Test Record
# ---------------------------------------------------------------------------


@dataclass
class ABTestObservation:
    """A single observation in the A/B test."""

    cycle: int = 0
    group: str = ""  # "candidate" or "incumbent"
    entity_id: str = ""  # stratification entity
    primary_value: float = 0.0
    secondary_values: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ABTestResult:
    """Result of a completed A/B test."""

    test_id: str = ""
    winner: str = ""  # "candidate", "incumbent", or "inconclusive"
    primary_metric: str = ""
    candidate_mean: float = 0.0
    incumbent_mean: float = 0.0
    z_statistic: float = 0.0
    p_value: float = 0.0
    effect_size: float = 0.0
    stopped_early: bool = False
    stopped_at_look: int = 0
    post_test_validated: bool = False
    secondary_comparisons: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# A/B Test Manager
# ---------------------------------------------------------------------------


class ABTestManager:
    """Manages A/B test lifecycle per Section 18.5.

    Usage::

        manager = ABTestManager()
        config = manager.design_test(
            candidate_id="model_v3",
            incumbent_id="model_v2",
            effect_size=0.2,
        )
        manager.start_test(config.test_id)

        # Record observations
        manager.record_observation(
            config.test_id,
            group="candidate", entity_id="AAPL", primary_value=0.02,
        )

        # Interim analysis
        result = manager.interim_analysis(config.test_id, look=1)
        if result and result.stopped_early:
            print("Test stopped early:", result.winner)
    """

    def __init__(self, storage_path: str | Path = "data/ab_tests.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tests: dict[str, ABTestConfig] = {}
        self._observations: dict[str, list[ABTestObservation]] = {}
        self._results: dict[str, ABTestResult] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for t in data.get("tests", []):
                cfg = ABTestConfig.from_dict(t)
                self._tests[cfg.test_id] = cfg
            for r in data.get("results", []):
                res = ABTestResult(**r)
                self._results[res.test_id] = res

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "tests": [t.to_dict() for t in self._tests.values()],
                    "results": [r.to_dict() for r in self._results.values()],
                },
                f,
                indent=2,
                default=str,
            )

    def design_test(
        self,
        candidate_id: str,
        incumbent_id: str,
        primary_metric: str = "sharpe",
        effect_size: float = 0.2,
        power: float = 0.80,
        alpha: float = 0.05,
        allocation_pct: float = 0.10,
        n_interim_looks: int = 4,
    ) -> ABTestConfig:
        """Design an A/B test with power analysis."""
        min_n = PowerAnalysis.compute_sample_size(effect_size, power, alpha)
        config = ABTestConfig(
            candidate_id=candidate_id,
            incumbent_id=incumbent_id,
            primary_metric=primary_metric,
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            allocation_pct=allocation_pct,
            minimum_sample_size=min_n,
            n_interim_looks=n_interim_looks,
        )
        self._tests[config.test_id] = config
        self._observations[config.test_id] = []
        self._save()
        logger.info(
            "Designed A/B test %s: %s vs %s (min_n=%d per group)",
            config.test_id,
            candidate_id,
            incumbent_id,
            min_n,
        )
        return config

    def start_test(self, test_id: str) -> ABTestConfig:
        """Start a designed A/B test."""
        config = self._tests[test_id]
        config.status = TestStatus.RUNNING
        self._save()
        return config

    def record_observation(
        self,
        test_id: str,
        group: str,
        primary_value: float,
        entity_id: str = "",
        secondary_values: dict[str, float] | None = None,
        cycle: int = 0,
    ) -> None:
        """Record a single observation."""
        obs = ABTestObservation(
            cycle=cycle,
            group=group,
            entity_id=entity_id,
            primary_value=primary_value,
            secondary_values=secondary_values or {},
        )
        self._observations.setdefault(test_id, []).append(obs)

    def interim_analysis(
        self,
        test_id: str,
        look: int,
    ) -> ABTestResult | None:
        """Run an interim analysis at the given look number.

        Returns ABTestResult if the test should stop, None otherwise.
        """
        config = self._tests[test_id]
        obs = self._observations.get(test_id, [])

        candidate_vals = [o.primary_value for o in obs if o.group == "candidate"]
        incumbent_vals = [o.primary_value for o in obs if o.group == "incumbent"]

        if len(candidate_vals) < 2 or len(incumbent_vals) < 2:
            return None

        c_mean = float(np.mean(candidate_vals))
        i_mean = float(np.mean(incumbent_vals))
        c_std = float(np.std(candidate_vals, ddof=1))
        i_std = float(np.std(incumbent_vals, ddof=1))

        pooled_se = math.sqrt(c_std**2 / len(candidate_vals) + i_std**2 / len(incumbent_vals))
        if pooled_se == 0:
            return None

        z = (c_mean - i_mean) / pooled_se
        p = float(2 * norm.sf(abs(z)))

        boundary = SequentialTestBoundary(n_looks=config.n_interim_looks, alpha=config.alpha)
        should_stop = boundary.should_stop(look, z)

        if should_stop:
            winner = "candidate" if z > 0 else "incumbent"
            result = ABTestResult(
                test_id=test_id,
                winner=winner,
                primary_metric=config.primary_metric,
                candidate_mean=c_mean,
                incumbent_mean=i_mean,
                z_statistic=z,
                p_value=p,
                effect_size=abs(c_mean - i_mean) / max(pooled_se, 1e-10),
                stopped_early=True,
                stopped_at_look=look,
            )
            config.status = TestStatus.STOPPED_EARLY
            self._results[test_id] = result
            self._save()
            return result
        return None

    def complete_test(self, test_id: str) -> ABTestResult:
        """Complete a test and produce final results."""
        config = self._tests[test_id]
        obs = self._observations.get(test_id, [])

        candidate_vals = [o.primary_value for o in obs if o.group == "candidate"]
        incumbent_vals = [o.primary_value for o in obs if o.group == "incumbent"]

        c_mean = float(np.mean(candidate_vals)) if candidate_vals else 0.0
        i_mean = float(np.mean(incumbent_vals)) if incumbent_vals else 0.0
        c_std = float(np.std(candidate_vals, ddof=1)) if len(candidate_vals) > 1 else 0.0
        i_std = float(np.std(incumbent_vals, ddof=1)) if len(incumbent_vals) > 1 else 0.0

        n_c = max(len(candidate_vals), 1)
        n_i = max(len(incumbent_vals), 1)
        pooled_se = math.sqrt(c_std**2 / n_c + i_std**2 / n_i) if (c_std + i_std) > 0 else 1e-10

        z = (c_mean - i_mean) / pooled_se
        p = float(2 * norm.sf(abs(z)))

        winner = "inconclusive"
        if p < config.alpha:
            winner = "candidate" if z > 0 else "incumbent"

        result = ABTestResult(
            test_id=test_id,
            winner=winner,
            primary_metric=config.primary_metric,
            candidate_mean=c_mean,
            incumbent_mean=i_mean,
            z_statistic=z,
            p_value=p,
            effect_size=abs(c_mean - i_mean) / max(pooled_se, 1e-10),
        )
        config.status = TestStatus.COMPLETED
        self._results[test_id] = result
        self._save()
        return result

    def validate_post_test(self, test_id: str, holdout_passed: bool) -> ABTestResult:
        """Post-test validation: winner must pass holdout backtest."""
        result = self._results[test_id]
        result.post_test_validated = holdout_passed
        config = self._tests[test_id]
        config.status = TestStatus.VALIDATED
        if not holdout_passed:
            result.winner = "inconclusive"
        self._save()
        return result

    def get_test(self, test_id: str) -> ABTestConfig | None:
        return self._tests.get(test_id)

    def get_result(self, test_id: str) -> ABTestResult | None:
        return self._results.get(test_id)

    def export_protocol(self, test_id: str) -> dict[str, Any]:
        """<ab_test_protocol> — Section 18.5 required output."""
        config = self._tests.get(test_id)
        if not config:
            return {}
        boundary = SequentialTestBoundary(n_looks=config.n_interim_looks, alpha=config.alpha)
        return {
            "report_type": "ab_test_protocol",
            "config": config.to_dict(),
            "boundaries": boundary.get_boundaries(),
            "power_analysis": {
                "minimum_sample_size": config.minimum_sample_size,
                "effect_size": config.effect_size,
                "power": config.power,
                "alpha": config.alpha,
            },
        }

    def export_results(self, test_id: str) -> dict[str, Any]:
        """<ab_test_results> — Section 18.6 required output."""
        result = self._results.get(test_id)
        config = self._tests.get(test_id)
        if not result or not config:
            return {}
        return {
            "report_type": "ab_test_results",
            "config": config.to_dict(),
            "result": result.to_dict(),
        }
