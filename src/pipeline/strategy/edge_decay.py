"""Edge decay monitoring for the swing strategy.

Tracks rolling performance metrics and raises alerts when the strategy's
statistical edge shows signs of degradation.  Three alert levels:

- YELLOW: 1 metric breached  → reduce size, review trades
- ORANGE: 2+ metrics breached → halt entries, recalibrate
- RED:    3+ metrics for 3+ months → full shutdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from pipeline.features.risk_metrics import hurst_exponent
from pipeline.infrastructure.notifier import AlertSeverity, notify

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    GREEN = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3


@dataclass
class DecayMetrics:
    """Snapshot of edge-decay monitoring metrics."""

    date: pd.Timestamp | None = None
    rolling_win_rate: float = np.nan
    rolling_profit_factor: float = np.nan
    rolling_sharpe: float = np.nan
    signal_hit_rate: float = np.nan
    avg_winner_loser_ratio: float = np.nan
    equity_hurst: float = np.nan
    breached_count: int = 0
    alert_level: AlertLevel = AlertLevel.GREEN
    breached_metrics: list[str] = field(default_factory=list)


class EdgeDecayMonitor:
    """Monitor strategy edge and raise alerts on decay."""

    def __init__(
        self,
        window: int = 60,
        min_trades: int = 10,
        win_rate_floor: float = 0.45,
        profit_factor_floor: float = 1.0,
        sharpe_floor: float = 0.0,
        hit_rate_decay_pct: float = 0.20,
        wl_ratio_floor: float = 1.0,
        hurst_floor: float = 0.45,
        orange_months: int = 2,
        red_months: int = 3,
    ) -> None:
        self.window = window
        self.min_trades = min_trades
        self.win_rate_floor = win_rate_floor
        self.profit_factor_floor = profit_factor_floor
        self.sharpe_floor = sharpe_floor
        self.hit_rate_decay_pct = hit_rate_decay_pct
        self.wl_ratio_floor = wl_ratio_floor
        self.hurst_floor = hurst_floor
        self.orange_months = orange_months
        self.red_months = red_months

        # Running state
        self._trade_pnls: list[float] = []
        self._daily_returns: list[float] = []
        self._equity_curve: list[float] = []
        self._signal_hits: list[bool] = []  # Did signal reach target_1?
        self._inception_hit_rate: float | None = None
        self._breach_history: list[int] = []  # breached_count per month

    def record_trade(self, pnl: float, hit_target: bool) -> None:
        """Record a completed trade result."""
        self._trade_pnls.append(pnl)
        self._signal_hits.append(hit_target)

    def record_daily_return(self, daily_return: float, equity: float) -> None:
        """Record a daily portfolio return and equity value."""
        self._daily_returns.append(daily_return)
        self._equity_curve.append(equity)

    def evaluate(self, date: pd.Timestamp | None = None) -> DecayMetrics:
        """Compute all decay metrics and determine the alert level."""
        metrics = DecayMetrics(date=date)
        breaches: list[str] = []

        recent_trades = self._trade_pnls[-self.window :]
        if len(recent_trades) < self.min_trades:
            return metrics  # Not enough data

        # --- 1. Rolling win rate ---
        wins = [p for p in recent_trades if p > 0]
        losses = [p for p in recent_trades if p <= 0]
        metrics.rolling_win_rate = len(wins) / len(recent_trades)
        if metrics.rolling_win_rate < self.win_rate_floor:
            breaches.append(f"win_rate={metrics.rolling_win_rate:.2f} < {self.win_rate_floor}")

        # --- 2. Rolling profit factor ---
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        metrics.rolling_profit_factor = (
            total_wins / total_losses if total_losses > 0 else float("inf")
        )
        if metrics.rolling_profit_factor < self.profit_factor_floor:
            breaches.append(
                f"profit_factor={metrics.rolling_profit_factor:.2f} < {self.profit_factor_floor}"
            )

        # --- 3. Rolling Sharpe (daily returns) ---
        recent_returns = self._daily_returns[-self.window :]
        if len(recent_returns) >= 20:
            arr = np.array(recent_returns)
            mu = np.mean(arr)
            sigma = np.std(arr)
            metrics.rolling_sharpe = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0
            if metrics.rolling_sharpe < self.sharpe_floor:
                breaches.append(f"sharpe={metrics.rolling_sharpe:.2f} < {self.sharpe_floor}")

        # --- 4. Signal hit rate vs inception ---
        recent_hits = self._signal_hits[-self.window :]
        if len(recent_hits) >= self.min_trades:
            metrics.signal_hit_rate = sum(recent_hits) / len(recent_hits)
            if self._inception_hit_rate is None and len(self._signal_hits) >= self.min_trades:
                self._inception_hit_rate = sum(self._signal_hits) / len(self._signal_hits)
            if self._inception_hit_rate is not None and self._inception_hit_rate > 0:
                decay = (self._inception_hit_rate - metrics.signal_hit_rate) / self._inception_hit_rate
                if decay > self.hit_rate_decay_pct:
                    breaches.append(
                        f"signal_hit_rate decay={decay:.0%} > {self.hit_rate_decay_pct:.0%}"
                    )

        # --- 5. Average winner / average loser ratio ---
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        metrics.avg_winner_loser_ratio = float(avg_win / avg_loss) if avg_loss > 0 else float("inf")
        if metrics.avg_winner_loser_ratio < self.wl_ratio_floor:
            breaches.append(
                f"wl_ratio={metrics.avg_winner_loser_ratio:.2f} < {self.wl_ratio_floor}"
            )

        # --- 6. Hurst exponent of equity curve ---
        if len(self._equity_curve) >= 100:
            equity_series = pd.Series(self._equity_curve[-252:])
            metrics.equity_hurst = hurst_exponent(equity_series)
            if not np.isnan(metrics.equity_hurst) and metrics.equity_hurst < self.hurst_floor:
                breaches.append(
                    f"equity_hurst={metrics.equity_hurst:.2f} < {self.hurst_floor}"
                )

        # --- Determine alert level ---
        metrics.breached_count = len(breaches)
        metrics.breached_metrics = breaches

        if len(breaches) >= 3:
            # Check if sustained
            self._breach_history.append(len(breaches))
            sustained_months = sum(1 for b in self._breach_history[-self.red_months:] if b >= 3)
            if sustained_months >= self.red_months:
                metrics.alert_level = AlertLevel.RED
                logger.critical("EDGE DECAY RED: %d metrics breached for %d months: %s",
                                len(breaches), sustained_months, breaches)
                notify(
                    AlertSeverity.CRITICAL,
                    "Edge Decay RED — Strategy Shutdown",
                    f"{len(breaches)} metrics breached for {sustained_months} consecutive months.",
                    {"breached": breaches, "sustained_months": sustained_months},
                )
            else:
                metrics.alert_level = AlertLevel.ORANGE
                logger.warning("EDGE DECAY ORANGE: %d metrics breached: %s", len(breaches), breaches)
                notify(
                    AlertSeverity.WARNING,
                    "Edge Decay ORANGE",
                    f"{len(breaches)} metrics breached. Consider halting entries.",
                    {"breached": breaches},
                )
        elif len(breaches) >= 2:
            self._breach_history.append(len(breaches))
            metrics.alert_level = AlertLevel.ORANGE
            logger.warning("EDGE DECAY ORANGE: %d metrics breached: %s", len(breaches), breaches)
            notify(
                AlertSeverity.WARNING,
                "Edge Decay ORANGE",
                f"{len(breaches)} metrics breached. Consider halting entries.",
                {"breached": breaches},
            )
        elif len(breaches) >= 1:
            self._breach_history.append(len(breaches))
            metrics.alert_level = AlertLevel.YELLOW
            logger.info("EDGE DECAY YELLOW: %s", breaches)
        else:
            self._breach_history.append(0)
            metrics.alert_level = AlertLevel.GREEN

        return metrics

    def summary(self) -> dict:
        """Return a summary dict of the latest evaluation."""
        m = self.evaluate()
        return {
            "win_rate": m.rolling_win_rate,
            "profit_factor": m.rolling_profit_factor,
            "sharpe": m.rolling_sharpe,
            "signal_hit_rate": m.signal_hit_rate,
            "wl_ratio": m.avg_winner_loser_ratio,
            "equity_hurst": m.equity_hurst,
            "alert_level": m.alert_level.name,
            "breached": m.breached_metrics,
        }
