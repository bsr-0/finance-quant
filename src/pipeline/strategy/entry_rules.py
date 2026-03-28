"""Entry rule abstraction for systematic strategies.

Implements explicit, programmatically checkable entry conditions with
AND/OR logic, gating filters, and signal threshold rules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry condition results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryCheckResult:
    """Result of evaluating a single entry condition."""

    condition_name: str
    passed: bool
    value: float = np.nan
    threshold: float = np.nan
    reason: str = ""


@dataclass
class EntryDecision:
    """Aggregated entry decision for a single instrument on a single date."""

    ticker: str
    date: pd.Timestamp
    eligible: bool
    signal_value: float = np.nan
    checks: list[EntryCheckResult] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[EntryCheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def failure_reasons(self) -> list[str]:
        return [c.reason or c.condition_name for c in self.failed_checks]


# ---------------------------------------------------------------------------
# Individual entry conditions
# ---------------------------------------------------------------------------


class EntryCondition(ABC):
    """Base class for a single entry condition (predicate)."""

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        """Evaluate this condition. Returns pass/fail with details."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable condition name."""


@dataclass
class EntryContext:
    """Context data available when evaluating entry conditions.

    Carries portfolio state, market data, and configuration needed
    by individual entry conditions.
    """

    portfolio_equity: float = 0.0
    available_cash: float = 0.0
    open_position_count: int = 0
    max_positions: int = 10
    current_portfolio_risk_pct: float = 0.0
    max_portfolio_risk_pct: float = 0.06
    held_tickers: set[str] = field(default_factory=set)
    regime: str = "BULL"
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    sector_exposures: dict[str, float] = field(default_factory=dict)
    sector_caps: dict[str, float] = field(default_factory=dict)
    ticker_sector: str = ""
    days_since_last_entry: int = 999
    min_entry_spacing_days: int = 0
    atr_pct: float = 0.0
    atr_pct_min: float = 0.0
    atr_pct_max: float = float("inf")


# ---------------------------------------------------------------------------
# Built-in entry conditions
# ---------------------------------------------------------------------------


class SignalThresholdCondition(EntryCondition):
    """Signal must exceed a threshold."""

    def __init__(self, threshold: float = 0.0, condition_name: str = "signal_threshold") -> None:
        self.threshold = threshold
        self._name = condition_name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = signal_value >= self.threshold
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=signal_value,
            threshold=self.threshold,
            reason="" if passed else f"Signal {signal_value:.4f} < threshold {self.threshold:.4f}",
        )


class RegimeCondition(EntryCondition):
    """Block entries in specified regimes."""

    def __init__(self, blocked_regimes: list[str] | None = None) -> None:
        self.blocked_regimes = blocked_regimes or ["BEAR"]

    @property
    def name(self) -> str:
        return "regime_filter"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = context.regime not in self.blocked_regimes
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            reason="" if passed else f"Regime {context.regime} is blocked",
        )


class MaxPositionsCondition(EntryCondition):
    """Enforce maximum number of open positions."""

    @property
    def name(self) -> str:
        return "max_positions"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = context.open_position_count < context.max_positions
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=float(context.open_position_count),
            threshold=float(context.max_positions),
            reason="" if passed else f"At max positions ({context.max_positions})",
        )


class NoDuplicatePositionCondition(EntryCondition):
    """Prevent duplicate positions in the same instrument."""

    @property
    def name(self) -> str:
        return "no_duplicate_position"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = ticker not in context.held_tickers
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            reason="" if passed else f"Already holding {ticker}",
        )


class RiskBudgetCondition(EntryCondition):
    """Check portfolio risk budget availability."""

    @property
    def name(self) -> str:
        return "risk_budget"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = context.current_portfolio_risk_pct < context.max_portfolio_risk_pct
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=context.current_portfolio_risk_pct,
            threshold=context.max_portfolio_risk_pct,
            reason="" if passed else "Portfolio risk budget exhausted",
        )


class SectorExposureCondition(EntryCondition):
    """Check sector exposure limits."""

    def __init__(self, default_cap: float = 0.30) -> None:
        self.default_cap = default_cap

    @property
    def name(self) -> str:
        return "sector_exposure"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        sector = context.ticker_sector
        if not sector:
            return EntryCheckResult(condition_name=self.name, passed=True)

        current_exposure = context.sector_exposures.get(sector, 0.0)
        cap = context.sector_caps.get(sector, self.default_cap)
        passed = current_exposure < cap
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=current_exposure,
            threshold=cap,
            reason="" if passed else f"Sector {sector} at {current_exposure:.1%} >= cap {cap:.1%}",
        )


class VolatilityFilterCondition(EntryCondition):
    """Ensure instrument volatility is within acceptable range."""

    @property
    def name(self) -> str:
        return "volatility_filter"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        if context.atr_pct_min <= 0 and context.atr_pct_max == float("inf"):
            return EntryCheckResult(condition_name=self.name, passed=True)

        passed = context.atr_pct_min <= context.atr_pct <= context.atr_pct_max
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=context.atr_pct,
            reason=(
                ""
                if passed
                else (
                    f"ATR% {context.atr_pct:.2f} outside "
                    f"[{context.atr_pct_min:.2f}, {context.atr_pct_max:.2f}]"
                )
            ),
        )


class MinCashCondition(EntryCondition):
    """Ensure sufficient cash for the trade."""

    def __init__(self, min_trade_value: float = 0.0) -> None:
        self.min_trade_value = min_trade_value

    @property
    def name(self) -> str:
        return "min_cash"

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryCheckResult:
        passed = context.available_cash > self.min_trade_value
        return EntryCheckResult(
            condition_name=self.name,
            passed=passed,
            value=context.available_cash,
            threshold=self.min_trade_value,
            reason="" if passed else "Insufficient cash",
        )


# ---------------------------------------------------------------------------
# Entry rule set (AND of all conditions)
# ---------------------------------------------------------------------------


@dataclass
class EntryRuleSet:
    """An ordered set of entry conditions that must all pass (AND logic).

    Evaluates conditions in order and short-circuits on the first failure
    unless ``evaluate_all`` is True.
    """

    conditions: list[EntryCondition] = field(default_factory=list)
    evaluate_all: bool = False

    def add(self, condition: EntryCondition) -> EntryRuleSet:
        self.conditions.append(condition)
        return self

    def evaluate(
        self,
        ticker: str,
        date: pd.Timestamp,
        signal_value: float,
        context: EntryContext,
    ) -> EntryDecision:
        """Evaluate all conditions and return the entry decision."""
        checks: list[EntryCheckResult] = []
        all_passed = True

        for cond in self.conditions:
            result = cond.evaluate(ticker, date, signal_value, context)
            checks.append(result)
            if not result.passed:
                all_passed = False
                if not self.evaluate_all:
                    break

        return EntryDecision(
            ticker=ticker,
            date=date,
            eligible=all_passed,
            signal_value=signal_value,
            checks=checks,
        )


# ---------------------------------------------------------------------------
# Pre-built entry rule sets
# ---------------------------------------------------------------------------


def institutional_entry_rules(
    signal_threshold: float = 0.0,
    blocked_regimes: list[str] | None = None,
    max_sector_exposure: float = 0.30,
) -> EntryRuleSet:
    """Standard institutional entry rule set."""
    rules = EntryRuleSet(evaluate_all=True)
    rules.add(SignalThresholdCondition(signal_threshold))
    rules.add(RegimeCondition(blocked_regimes or ["BEAR"]))
    rules.add(MaxPositionsCondition())
    rules.add(NoDuplicatePositionCondition())
    rules.add(RiskBudgetCondition())
    rules.add(SectorExposureCondition(default_cap=max_sector_exposure))
    return rules
