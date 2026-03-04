"""Risk constraint definitions and portfolio compliance checking.

Provides a machine-readable representation of risk constraints that can
be evaluated against a portfolio and rendered into a risk parameter table
in the strategy memo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    POSITION_WEIGHT = "position_weight"
    POSITION_NOTIONAL = "position_notional"
    SECTOR_EXPOSURE = "sector_exposure"
    COUNTRY_EXPOSURE = "country_exposure"
    REGION_EXPOSURE = "region_exposure"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    MAX_DRAWDOWN = "max_drawdown"
    PORTFOLIO_RISK = "portfolio_risk"
    FACTOR_EXPOSURE = "factor_exposure"
    ADV_PARTICIPATION = "adv_participation"
    CONCENTRATION = "concentration"
    TURNOVER = "turnover"


class ConstraintSeverity(Enum):
    HARD = "hard"      # Must never be violated
    SOFT = "soft"      # Warning, but allowed temporarily
    ADVISORY = "advisory"  # Information only


# ---------------------------------------------------------------------------
# Individual constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConstraint:
    """A single risk constraint with a limit and severity."""

    name: str
    constraint_type: ConstraintType
    limit_value: float
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    description: str = ""
    unit: str = ""  # e.g., "%", "bps", "$"
    applies_to: str = ""  # e.g., sector name, factor name, or "" for portfolio-level

    def check(self, actual_value: float) -> ConstraintCheckResult:
        """Check if the actual value violates this constraint."""
        violated = actual_value > self.limit_value
        return ConstraintCheckResult(
            constraint=self,
            actual_value=actual_value,
            violated=violated,
            breach_amount=max(0, actual_value - self.limit_value),
        )


@dataclass
class ConstraintCheckResult:
    """Result of checking a single constraint."""

    constraint: RiskConstraint
    actual_value: float
    violated: bool
    breach_amount: float = 0.0

    @property
    def severity(self) -> ConstraintSeverity:
        return self.constraint.severity

    @property
    def is_hard_breach(self) -> bool:
        return self.violated and self.severity == ConstraintSeverity.HARD


# ---------------------------------------------------------------------------
# Risk constraint set
# ---------------------------------------------------------------------------

@dataclass
class RiskConstraintSet:
    """Complete set of risk constraints for a strategy."""

    constraints: list[RiskConstraint] = field(default_factory=list)

    def add(self, constraint: RiskConstraint) -> RiskConstraintSet:
        self.constraints.append(constraint)
        return self

    def evaluate_portfolio(
        self,
        weights: pd.Series,
        sector_map: dict[str, str] | None = None,
        country_map: dict[str, str] | None = None,
        volatilities: pd.Series | None = None,
        adv: pd.Series | None = None,
        prices: pd.Series | None = None,
        shares: pd.Series | None = None,
        current_drawdown: float = 0.0,
        daily_turnover: float = 0.0,
    ) -> list[ConstraintCheckResult]:
        """Evaluate all constraints against the current portfolio.

        Args:
            weights: Portfolio weights per ticker (fraction of capital).
            sector_map: Ticker → sector mapping.
            country_map: Ticker → country mapping.
            volatilities: Annualized vol per ticker.
            adv: Average daily volume (shares) per ticker.
            prices: Current prices per ticker.
            shares: Target shares per ticker.
            current_drawdown: Current portfolio drawdown (as positive fraction).
            daily_turnover: Daily turnover as fraction of capital.

        Returns:
            List of ``ConstraintCheckResult`` for each constraint.
        """
        results: list[ConstraintCheckResult] = []

        for constraint in self.constraints:
            actual = self._compute_actual(
                constraint, weights, sector_map, country_map,
                volatilities, adv, prices, shares,
                current_drawdown, daily_turnover,
            )
            results.append(constraint.check(actual))

        return results

    def _compute_actual(
        self,
        constraint: RiskConstraint,
        weights: pd.Series,
        sector_map: dict[str, str] | None,
        country_map: dict[str, str] | None,
        volatilities: pd.Series | None,
        adv: pd.Series | None,
        prices: pd.Series | None,
        shares: pd.Series | None,
        current_drawdown: float,
        daily_turnover: float,
    ) -> float:
        ct = constraint.constraint_type

        if ct == ConstraintType.POSITION_WEIGHT:
            return float(weights.abs().max()) if not weights.empty else 0.0

        if ct == ConstraintType.GROSS_EXPOSURE:
            return float(weights.abs().sum())

        if ct == ConstraintType.NET_EXPOSURE:
            return float(abs(weights.sum()))

        if ct == ConstraintType.SECTOR_EXPOSURE:
            if sector_map is None:
                return 0.0
            target_sector = constraint.applies_to
            sector_weights = pd.Series({
                t: w for t, w in weights.items()
                if sector_map.get(t, "") == target_sector
            })
            return float(sector_weights.abs().sum())

        if ct == ConstraintType.COUNTRY_EXPOSURE:
            if country_map is None:
                return 0.0
            target_country = constraint.applies_to
            country_weights = pd.Series({
                t: w for t, w in weights.items()
                if country_map.get(t, "") == target_country
            })
            return float(country_weights.abs().sum())

        if ct == ConstraintType.MAX_DRAWDOWN:
            return current_drawdown

        if ct == ConstraintType.PORTFOLIO_RISK:
            if volatilities is None:
                return 0.0
            # Simplified: sum of |w_i| * sigma_i (ignores correlations)
            risk = sum(
                abs(weights.get(t, 0)) * volatilities.get(t, 0)
                for t in weights.index
            )
            return float(risk)

        if ct == ConstraintType.ADV_PARTICIPATION:
            if adv is None or shares is None or prices is None:
                return 0.0
            max_part = 0.0
            for t in shares.index:
                if adv.get(t, 0) > 0:
                    max_part = max(max_part, abs(shares.get(t, 0)) / adv[t])
            return max_part

        if ct == ConstraintType.CONCENTRATION:
            if weights.empty:
                return 0.0
            top_n = min(5, len(weights))
            return float(weights.abs().nlargest(top_n).sum())

        if ct == ConstraintType.TURNOVER:
            return daily_turnover

        return 0.0

    def get_violations(
        self, results: list[ConstraintCheckResult]
    ) -> list[ConstraintCheckResult]:
        """Return only violated constraints."""
        return [r for r in results if r.violated]

    def get_hard_violations(
        self, results: list[ConstraintCheckResult]
    ) -> list[ConstraintCheckResult]:
        """Return only hard constraint violations."""
        return [r for r in results if r.is_hard_breach]

    def to_table(self) -> pd.DataFrame:
        """Render constraints as a table suitable for memo inclusion."""
        records = []
        for c in self.constraints:
            records.append({
                "Constraint": c.name,
                "Type": c.constraint_type.value,
                "Limit": f"{c.limit_value:.2%}" if c.unit == "%" else f"{c.limit_value}",
                "Severity": c.severity.value,
                "Applies To": c.applies_to or "Portfolio",
                "Description": c.description,
            })
        return pd.DataFrame(records)

    def to_markdown_table(self) -> str:
        """Render constraints as a Markdown table."""
        df = self.to_table()
        lines = ["| " + " | ".join(df.columns) + " |"]
        lines.append("|" + "|".join("---" for _ in df.columns) + "|")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built constraint sets
# ---------------------------------------------------------------------------

def institutional_constraints(
    max_position_weight: float = 0.05,
    max_sector_exposure: float = 0.30,
    max_country_exposure: float = 0.40,
    max_gross_exposure: float = 2.0,
    max_net_exposure: float = 1.0,
    max_drawdown: float = 0.15,
    max_adv_participation: float = 0.10,
    max_turnover: float = 0.20,
    sectors: list[str] | None = None,
    countries: list[str] | None = None,
) -> RiskConstraintSet:
    """Standard institutional risk constraint set."""
    cs = RiskConstraintSet()

    cs.add(RiskConstraint(
        name="Max Single Position Weight",
        constraint_type=ConstraintType.POSITION_WEIGHT,
        limit_value=max_position_weight,
        severity=ConstraintSeverity.HARD,
        description=f"No single position may exceed {max_position_weight:.0%} of capital",
        unit="%",
    ))
    cs.add(RiskConstraint(
        name="Max Gross Exposure",
        constraint_type=ConstraintType.GROSS_EXPOSURE,
        limit_value=max_gross_exposure,
        severity=ConstraintSeverity.HARD,
        description=f"Total gross exposure capped at {max_gross_exposure:.0%}",
        unit="%",
    ))
    cs.add(RiskConstraint(
        name="Max Net Exposure",
        constraint_type=ConstraintType.NET_EXPOSURE,
        limit_value=max_net_exposure,
        severity=ConstraintSeverity.HARD,
        description=f"Net exposure capped at {max_net_exposure:.0%}",
        unit="%",
    ))
    cs.add(RiskConstraint(
        name="Max Portfolio Drawdown",
        constraint_type=ConstraintType.MAX_DRAWDOWN,
        limit_value=max_drawdown,
        severity=ConstraintSeverity.HARD,
        description=f"Strategy halts at {max_drawdown:.0%} drawdown from peak",
        unit="%",
    ))
    cs.add(RiskConstraint(
        name="Max ADV Participation",
        constraint_type=ConstraintType.ADV_PARTICIPATION,
        limit_value=max_adv_participation,
        severity=ConstraintSeverity.HARD,
        description=f"Max {max_adv_participation:.0%} of average daily volume per trade",
        unit="%",
    ))
    cs.add(RiskConstraint(
        name="Max Daily Turnover",
        constraint_type=ConstraintType.TURNOVER,
        limit_value=max_turnover,
        severity=ConstraintSeverity.SOFT,
        description=f"Daily turnover target below {max_turnover:.0%}",
        unit="%",
    ))

    for sector in (sectors or []):
        cs.add(RiskConstraint(
            name=f"Sector Cap: {sector}",
            constraint_type=ConstraintType.SECTOR_EXPOSURE,
            limit_value=max_sector_exposure,
            severity=ConstraintSeverity.HARD,
            description=f"{sector} sector capped at {max_sector_exposure:.0%}",
            unit="%",
            applies_to=sector,
        ))

    for country in (countries or []):
        cs.add(RiskConstraint(
            name=f"Country Cap: {country}",
            constraint_type=ConstraintType.COUNTRY_EXPOSURE,
            limit_value=max_country_exposure,
            severity=ConstraintSeverity.HARD,
            description=f"{country} capped at {max_country_exposure:.0%}",
            unit="%",
            applies_to=country,
        ))

    return cs
