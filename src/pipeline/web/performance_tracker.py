"""Track daily prediction outcomes: did signals hit target or stop out?

Maintains a JSON history file that accumulates signal predictions and their
resolved outcomes over time, enabling performance reporting on the static site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pipeline.web.outcome_resolution import ResolutionPolicy, resolve_one

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

    def __init__(
        self,
        history_path: str | Path,
        max_holding_days: int | None = None,
        policy: ResolutionPolicy | None = None,
    ):
        """
        Args:
            history_path: JSON history file.
            max_holding_days: Deprecated alias for ``policy.max_holding_bars``.
                Retained for existing callers; interpreted as a bar count.
            policy: Resolution rules.  Takes precedence over
                ``max_holding_days`` when both are supplied.
        """
        self.history_path = Path(history_path)
        if policy is None:
            policy = (
                ResolutionPolicy()
                if max_holding_days is None
                else ResolutionPolicy(max_holding_bars=max_holding_days)
            )
        elif max_holding_days is not None:
            logger.warning(
                "Both policy and max_holding_days given; ignoring max_holding_days=%s",
                max_holding_days,
            )
        self.policy = policy
        self.history = self._load_history()

    @property
    def max_holding_days(self) -> int:
        """Deprecated: holding limit is now counted in bars, not calendar days."""
        return self.policy.max_holding_bars

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
        self.history.last_updated = datetime.now(UTC).isoformat()
        self.history_path.write_text(json.dumps(self.history.to_dict(), indent=2, default=str))

    def add_signals(self, signals_df: pd.DataFrame, signal_date: str) -> int:
        """Add new signals from today's generate-signals output.

        Returns the number of new predictions added.
        """
        existing = {(p["signal_date"], p["ticker"]) for p in self.history.predictions}

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
        summary = {
            "hit_target": 0,
            "stopped_out": 0,
            "expired": 0,
            "still_active": 0,
            "unresolvable": 0,
        }

        for pred in self.history.predictions:
            if pred["outcome"] != "active":
                continue

            ticker = pred["ticker"]
            if ticker not in price_data:
                # Universe membership is refetched periodically, so a dropped or
                # delisted ticker would otherwise sit active forever, invisible
                # in every summary.
                summary["unresolvable"] += 1
                pred["unresolvable_reason"] = "no_price_data"
                logger.warning(
                    "No price data for %s (signal %s); prediction unresolvable",
                    ticker,
                    pred["signal_date"],
                )
                continue

            df = price_data[ticker]
            signal_date = pd.Timestamp(pred["signal_date"])

            # Bars strictly after the signal bar and visible as of the
            # evaluation date.  resolve_one truncates this to the holding
            # window before checking any barrier.
            future_bars = df[(df.index > signal_date) & (df.index <= as_of_date)]
            if future_bars.empty:
                summary["still_active"] += 1
                continue

            resolution = resolve_one(
                bars=future_bars,
                entry=pred["entry_price"],
                stop=pred["stop_price"],
                target=pred["target_price"],
                policy=self.policy,
                direction=str(pred.get("direction", "long")),
            )

            if resolution is None:
                summary["still_active"] += 1
                continue

            pred["outcome"] = resolution.outcome
            pred["resolved_date"] = str(resolution.resolved_date.date())
            pred["resolved_price"] = round(resolution.fill_price, 4)
            pred["pnl_pct"] = round(resolution.pnl_pct, 2)
            pred["bars_held"] = resolution.bars_held
            # Calendar days retained as a derived field for the site templates.
            pred["days_held"] = (resolution.resolved_date - signal_date).days
            pred["same_bar_ambiguous"] = resolution.same_bar_ambiguous
            pred["gapped"] = resolution.gapped
            summary[resolution.outcome] += 1

        return summary

    def get_stats(self) -> dict:
        """Compute aggregate performance statistics.

        Two distinct notions of "win" were previously conflated.  Both are now
        reported explicitly:

        - ``target_hit_rate``: fraction of resolved trades that reached the
          profit target.  A trade that expired at +3% does not count.
        - ``profitable_rate``: fraction of resolved trades with positive P&L,
          including profitable expiries.  This is the economically meaningful
          one and is what ``win_rate`` now aliases.
        """
        preds = self.history.predictions
        total = len(preds)
        empty = {
            "total": 0,
            "active": 0,
            "resolved": 0,
            "hit_target": 0,
            "stopped_out": 0,
            "expired": 0,
            "unresolvable": 0,
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "profitable_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "n_ambiguous": 0,
            "n_gapped": 0,
            "mean_bars_held": 0.0,
        }
        if total == 0:
            return empty

        active = sum(1 for p in preds if p["outcome"] == "active")
        hit = sum(1 for p in preds if p["outcome"] == "hit_target")
        stopped = sum(1 for p in preds if p["outcome"] == "stopped_out")
        expired = sum(1 for p in preds if p["outcome"] == "expired")
        unresolvable = sum(1 for p in preds if p.get("unresolvable_reason"))
        resolved = hit + stopped + expired

        pnls = [p["pnl_pct"] for p in preds if p["pnl_pct"] is not None]
        profitable = sum(1 for p in pnls if p > 0)

        target_hit_rate = hit / resolved * 100 if resolved > 0 else 0.0
        profitable_rate = profitable / len(pnls) * 100 if pnls else 0.0

        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        avg_win = sum(p for p in pnls if p > 0) / max(profitable, 1)
        avg_loss = sum(p for p in pnls if p < 0) / max(sum(1 for p in pnls if p < 0), 1)

        bars = [p["bars_held"] for p in preds if p.get("bars_held") is not None]

        return {
            "total": total,
            "active": active,
            "resolved": resolved,
            "hit_target": hit,
            "stopped_out": stopped,
            "expired": expired,
            "unresolvable": unresolvable,
            "win_rate": round(profitable_rate, 1),
            "target_hit_rate": round(target_hit_rate, 1),
            "profitable_rate": round(profitable_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "n_ambiguous": sum(1 for p in preds if p.get("same_bar_ambiguous")),
            "n_gapped": sum(1 for p in preds if p.get("gapped")),
            "mean_bars_held": round(sum(bars) / len(bars), 1) if bars else 0.0,
        }

    def get_recent_predictions(self, days: int = 30) -> list[dict]:
        """Get predictions from the last N days, sorted by date descending."""
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        recent = [p for p in self.history.predictions if p["signal_date"] >= cutoff]
        return sorted(recent, key=lambda p: (p["signal_date"], -p["score"]), reverse=True)

    def get_ticker_history(self, ticker: str) -> list[dict]:
        """Get all predictions for a specific ticker."""
        return [p for p in self.history.predictions if p["ticker"] == ticker]
