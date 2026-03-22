"""Track daily prediction outcomes: did signals hit target or stop out?

Maintains a JSON history file that accumulates signal predictions and their
resolved outcomes over time, enabling performance reporting on the static site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single signal prediction and its outcome."""

    signal_date: str
    ticker: str
    score: int
    confidence: str
    entry_price: float
    stop_price: float
    target_price: float
    regime: str
    direction: str = "long"
    # Outcome fields (filled in when resolved)
    outcome: str = "active"  # active | hit_target | stopped_out | expired | missed
    resolved_date: str | None = None
    resolved_price: float | None = None
    pnl_pct: float | None = None
    days_held: int | None = None


@dataclass
class PredictionHistory:
    """Full prediction history stored as JSON."""

    predictions: list[dict] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {"predictions": self.predictions, "last_updated": self.last_updated}

    @classmethod
    def from_dict(cls, data: dict) -> PredictionHistory:
        return cls(
            predictions=data.get("predictions", []),
            last_updated=data.get("last_updated", ""),
        )


class PerformanceTracker:
    """Tracks signal predictions and resolves their outcomes against price data."""

    def __init__(self, history_path: str | Path, max_holding_days: int = 15):
        self.history_path = Path(history_path)
        self.max_holding_days = max_holding_days
        self.history = self._load_history()

    def _load_history(self) -> PredictionHistory:
        if self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text())
                return PredictionHistory.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt history file, starting fresh")
        return PredictionHistory()

    def save(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history.last_updated = datetime.utcnow().isoformat()
        self.history_path.write_text(
            json.dumps(self.history.to_dict(), indent=2, default=str)
        )

    def add_signals(self, signals_df: pd.DataFrame, signal_date: str) -> int:
        """Add new signals from today's generate-signals output.

        Returns the number of new predictions added.
        """
        existing = {
            (p["signal_date"], p["ticker"])
            for p in self.history.predictions
        }

        added = 0
        for _, row in signals_df.iterrows():
            key = (signal_date, str(row["ticker"]))
            if key in existing:
                continue

            record = PredictionRecord(
                signal_date=signal_date,
                ticker=str(row["ticker"]),
                score=int(row["score"]),
                confidence=str(row.get("confidence", "MEDIUM")),
                entry_price=float(row["entry_price"]),
                stop_price=float(row["stop_price"]),
                target_price=float(row["target_1"]),
                regime=str(row.get("regime", "unknown")),
                direction=str(row.get("direction", "long")),
            )
            self.history.predictions.append(asdict(record))
            added += 1

        return added

    def resolve_outcomes(self, price_data: dict[str, pd.DataFrame], as_of: str) -> dict:
        """Check active predictions against current prices and resolve outcomes.

        Args:
            price_data: dict of ticker -> DataFrame with OHLCV data (date-indexed).
            as_of: current date string (YYYY-MM-DD) to evaluate against.

        Returns:
            Summary dict with counts of each outcome type.
        """
        as_of_date = pd.Timestamp(as_of)
        summary = {"hit_target": 0, "stopped_out": 0, "expired": 0, "still_active": 0}

        for pred in self.history.predictions:
            if pred["outcome"] != "active":
                continue

            ticker = pred["ticker"]
            if ticker not in price_data:
                continue

            df = price_data[ticker]
            signal_date = pd.Timestamp(pred["signal_date"])
            entry_price = pred["entry_price"]
            stop_price = pred["stop_price"]
            target_price = pred["target_price"]

            # Get price bars after signal date
            future_bars = df[df.index > signal_date]
            if future_bars.empty:
                summary["still_active"] += 1
                continue

            days_elapsed = (as_of_date - signal_date).days
            resolved = False

            for bar_date, bar in future_bars.iterrows():
                if bar_date > as_of_date:
                    break

                high = bar.get("high", bar.get("close", 0))
                low = bar.get("low", bar.get("close", 0))
                close = bar["close"]

                # Check stop hit (low touches stop)
                if low <= stop_price:
                    pred["outcome"] = "stopped_out"
                    pred["resolved_date"] = str(bar_date.date())
                    pred["resolved_price"] = stop_price
                    pred["pnl_pct"] = round(
                        (stop_price - entry_price) / entry_price * 100, 2
                    )
                    pred["days_held"] = (bar_date - signal_date).days
                    summary["stopped_out"] += 1
                    resolved = True
                    break

                # Check target hit (high touches target)
                if high >= target_price:
                    pred["outcome"] = "hit_target"
                    pred["resolved_date"] = str(bar_date.date())
                    pred["resolved_price"] = target_price
                    pred["pnl_pct"] = round(
                        (target_price - entry_price) / entry_price * 100, 2
                    )
                    pred["days_held"] = (bar_date - signal_date).days
                    summary["hit_target"] += 1
                    resolved = True
                    break

            if not resolved:
                if days_elapsed >= self.max_holding_days:
                    # Expired: use last available close
                    last_bar = future_bars[future_bars.index <= as_of_date]
                    if not last_bar.empty:
                        last_close = last_bar.iloc[-1]["close"]
                        pred["outcome"] = "expired"
                        pred["resolved_date"] = str(as_of_date.date())
                        pred["resolved_price"] = float(last_close)
                        pred["pnl_pct"] = round(
                            (last_close - entry_price) / entry_price * 100, 2
                        )
                        pred["days_held"] = days_elapsed
                        summary["expired"] += 1
                else:
                    summary["still_active"] += 1

        return summary

    def get_stats(self) -> dict:
        """Compute aggregate performance statistics."""
        preds = self.history.predictions
        total = len(preds)
        if total == 0:
            return {"total": 0, "active": 0, "resolved": 0, "win_rate": 0.0}

        active = sum(1 for p in preds if p["outcome"] == "active")
        hit = sum(1 for p in preds if p["outcome"] == "hit_target")
        stopped = sum(1 for p in preds if p["outcome"] == "stopped_out")
        expired = sum(1 for p in preds if p["outcome"] == "expired")
        resolved = hit + stopped + expired

        win_rate = hit / resolved * 100 if resolved > 0 else 0.0

        pnls = [p["pnl_pct"] for p in preds if p["pnl_pct"] is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        avg_win = (
            sum(p for p in pnls if p > 0) / max(sum(1 for p in pnls if p > 0), 1)
        )
        avg_loss = (
            sum(p for p in pnls if p < 0) / max(sum(1 for p in pnls if p < 0), 1)
        )

        return {
            "total": total,
            "active": active,
            "resolved": resolved,
            "hit_target": hit,
            "stopped_out": stopped,
            "expired": expired,
            "win_rate": round(win_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
        }

    def get_recent_predictions(self, days: int = 30) -> list[dict]:
        """Get predictions from the last N days, sorted by date descending."""
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        recent = [
            p for p in self.history.predictions if p["signal_date"] >= cutoff
        ]
        return sorted(recent, key=lambda p: (p["signal_date"], -p["score"]), reverse=True)

    def get_ticker_history(self, ticker: str) -> list[dict]:
        """Get all predictions for a specific ticker."""
        return [
            p for p in self.history.predictions if p["ticker"] == ticker
        ]
