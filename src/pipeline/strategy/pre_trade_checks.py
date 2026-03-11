"""Pre-trade checks for signal-level validation.

Verifies that each signal is actionable before it reaches the trader:
tradability, liquidity, risk limits, data freshness, and portfolio
constraints.  Returns a structured result per signal with pass/fail
detail so the trader can see exactly why a signal was filtered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from pipeline.strategy.risk import DrawdownLevel, RiskState
from pipeline.strategy.signals import SignalScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckDetail:
    """Result of a single pre-trade check."""

    name: str
    passed: bool
    reason: str = ""


@dataclass
class PreTradeCheckResult:
    """Aggregated pre-trade check result for one signal."""

    ticker: str
    passed: bool
    checks: list[CheckDetail] = field(default_factory=list)

    @property
    def failure_reasons(self) -> list[str]:
        return [c.reason for c in self.checks if not c.passed]


def run_pre_trade_checks(
    signal: SignalScore,
    price_df: pd.DataFrame,
    risk_state: RiskState | None = None,
    held_tickers: set[str] | None = None,
    equity: float = 0.0,
    cash: float = 0.0,
    min_volume: float = 50_000,
    min_price: float = 5.0,
    max_staleness_days: int = 3,
) -> PreTradeCheckResult:
    """Run all pre-trade checks on a single signal.

    Args:
        signal: Signal score from ``SignalEngine``.
        price_df: OHLCV DataFrame for this ticker (with indicators).
        risk_state: Current portfolio risk state (from ``SwingRiskManager``).
        held_tickers: Set of tickers currently held.
        equity: Current portfolio equity.
        cash: Available cash.
        min_volume: Minimum average daily volume (shares).
        min_price: Minimum share price.
        max_staleness_days: Max days since last trade data.

    Returns:
        ``PreTradeCheckResult`` with pass/fail and detailed checks.
    """
    checks: list[CheckDetail] = []
    held_tickers = held_tickers or set()

    # 1. Already holding
    if signal.symbol in held_tickers:
        checks.append(CheckDetail(
            "no_duplicate", False, f"Already holding {signal.symbol}",
        ))
    else:
        checks.append(CheckDetail("no_duplicate", True))

    # 2. Data freshness
    if price_df.empty:
        checks.append(CheckDetail(
            "data_freshness", False, f"No price data for {signal.symbol}",
        ))
    else:
        last_date = price_df.index[-1]
        signal_date = signal.date
        if hasattr(signal_date, "normalize"):
            signal_date = signal_date.normalize()
        if hasattr(last_date, "normalize"):
            last_date = last_date.normalize()
        gap_days = (signal_date - last_date).days if signal_date >= last_date else 0
        if gap_days > max_staleness_days:
            checks.append(CheckDetail(
                "data_freshness", False,
                f"Data stale: last date {last_date.date()}, gap {gap_days} days",
            ))
        else:
            checks.append(CheckDetail("data_freshness", True))

    # 3. Minimum price
    if not price_df.empty:
        latest_close = float(price_df.iloc[-1]["close"])
        if latest_close < min_price:
            checks.append(CheckDetail(
                "min_price", False,
                f"Price ${latest_close:.2f} below minimum ${min_price:.2f}",
            ))
        else:
            checks.append(CheckDetail("min_price", True))
    else:
        checks.append(CheckDetail("min_price", False, "No price data"))

    # 4. Minimum volume
    if not price_df.empty and "volume" in price_df.columns:
        avg_vol = float(price_df["volume"].tail(20).mean())
        if avg_vol < min_volume:
            checks.append(CheckDetail(
                "min_volume", False,
                f"Avg volume {avg_vol:,.0f} below minimum {min_volume:,.0f}",
            ))
        else:
            checks.append(CheckDetail("min_volume", True))
    else:
        checks.append(CheckDetail("min_volume", False, "No volume data"))

    # 5. Risk state
    if risk_state is not None:
        if risk_state.drawdown_level >= DrawdownLevel.ORANGE:
            checks.append(CheckDetail(
                "risk_state", False,
                f"Drawdown level {risk_state.drawdown_level.name} blocks new entries",
            ))
        elif not risk_state.can_open_new:
            checks.append(CheckDetail(
                "risk_state", False, "Risk state blocks new entries (cooldown or consecutive losses)",
            ))
        else:
            checks.append(CheckDetail("risk_state", True))

    # 6. Sufficient cash
    if equity > 0 and not price_df.empty:
        latest_close = float(price_df.iloc[-1]["close"])
        if cash < latest_close:
            checks.append(CheckDetail(
                "cash_available", False,
                f"Cash ${cash:.2f} insufficient for min 1 share at ${latest_close:.2f}",
            ))
        else:
            checks.append(CheckDetail("cash_available", True))

    all_passed = all(c.passed for c in checks)
    return PreTradeCheckResult(
        ticker=signal.symbol,
        passed=all_passed,
        checks=checks,
    )


def filter_signals(
    signals: list[SignalScore],
    price_data: dict[str, pd.DataFrame],
    risk_state: RiskState | None = None,
    held_tickers: set[str] | None = None,
    equity: float = 0.0,
    cash: float = 0.0,
    min_volume: float = 50_000,
    min_price: float = 5.0,
) -> tuple[list[SignalScore], list[PreTradeCheckResult]]:
    """Filter a list of signals through pre-trade checks.

    Returns:
        Tuple of (passed_signals, all_check_results).
    """
    passed: list[SignalScore] = []
    results: list[PreTradeCheckResult] = []

    for sig in signals:
        df = price_data.get(sig.symbol, pd.DataFrame())
        result = run_pre_trade_checks(
            signal=sig,
            price_df=df,
            risk_state=risk_state,
            held_tickers=held_tickers,
            equity=equity,
            cash=cash,
            min_volume=min_volume,
            min_price=min_price,
        )
        results.append(result)
        if result.passed:
            passed.append(sig)
        else:
            logger.info(
                "Pre-trade REJECT %s: %s",
                sig.symbol, "; ".join(result.failure_reasons),
            )

    logger.info(
        "Pre-trade filter: %d/%d signals passed",
        len(passed), len(signals),
    )
    return passed, results
