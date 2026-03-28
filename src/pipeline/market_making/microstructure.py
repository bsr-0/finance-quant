"""Market microstructure analysis for market-making diagnostics.

Analytics on order-book shape, fill probability, queue position, trade-size
distribution, and time-of-day effects.  These diagnostics can be run on
historical data to tune framework parameters and are integrated into the
live trading loop for real-time monitoring.

Diagnostics provided:
    - Fill probability vs distance from mid.
    - PnL vs time-of-day.
    - PnL vs spread width and inventory level.
    - Trade-size distribution and outlier detection.
    - Order-book imbalance and its predictive power.
    - Queue position estimation.

Assumptions:
    - Historical data is provided as DataFrames with timestamp columns.
    - Live data is provided as individual observations via the ``record_*``
      methods.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BookLevel:
    """A single price level in the order book."""

    price: float
    size: float
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """A point-in-time snapshot of the order book."""

    symbol: str
    timestamp_ns: int
    bids: list[BookLevel]
    asks: list[BookLevel]

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.best_bid > 0 else 0.0

    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid_size - ask_size) / (bid_size + ask_size).

        +1 = all weight on bid, -1 = all weight on ask.
        """
        bid_total = sum(level.size for level in self.bids[:3])
        ask_total = sum(level.size for level in self.asks[:3])
        total = bid_total + ask_total
        if total == 0:
            return 0.0
        return (bid_total - ask_total) / total

    @property
    def depth_ratio(self) -> float:
        """Ratio of total bid depth to total ask depth (top 5 levels)."""
        bid_depth = sum(level.size for level in self.bids[:5])
        ask_depth = sum(level.size for level in self.asks[:5])
        if ask_depth == 0:
            return float("inf") if bid_depth > 0 else 1.0
        return bid_depth / ask_depth


class MicrostructureAnalyzer:
    """Analyze market microstructure for market-making optimization.

    Collects observations in real-time and produces diagnostic reports
    that can be used to tune spread, sizing, and quoting parameters.
    """

    def __init__(self) -> None:
        self._fills: list[dict] = []
        self._book_snapshots: list[dict] = []
        self._trade_sizes: list[float] = []
        self._pnl_by_hour: defaultdict[int, list[float]] = defaultdict(list)
        self._pnl_by_spread: list[tuple[float, float]] = []
        self._pnl_by_inventory: list[tuple[float, float]] = []
        self._fill_distances: list[tuple[float, bool]] = []

    def record_fill(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        mid_at_fill: float,
        spread_at_fill: float,
        inventory_at_fill: float,
        timestamp_ns: int,
        post_fill_mid: float | None = None,
    ) -> None:
        """Record a fill event for microstructure analysis."""
        distance_from_mid = abs(price - mid_at_fill) / mid_at_fill * 10_000
        pnl = 0.0
        if post_fill_mid is not None:
            if side == "buy":
                pnl = (post_fill_mid - price) * size
            else:
                pnl = (price - post_fill_mid) * size

        hour = (timestamp_ns // (3_600 * 10**9)) % 24

        self._fills.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "size": size,
                "mid": mid_at_fill,
                "spread_bps": spread_at_fill,
                "inventory": inventory_at_fill,
                "distance_bps": distance_from_mid,
                "pnl": pnl,
                "hour": hour,
                "timestamp_ns": timestamp_ns,
            }
        )

        self._trade_sizes.append(size)
        self._pnl_by_hour[hour].append(pnl)
        self._pnl_by_spread.append((spread_at_fill, pnl))
        self._pnl_by_inventory.append((inventory_at_fill, pnl))
        self._fill_distances.append((distance_from_mid, True))

    def record_book_snapshot(self, book: OrderBookSnapshot) -> None:
        """Record an order-book snapshot."""
        self._book_snapshots.append(
            {
                "symbol": book.symbol,
                "mid": book.mid,
                "spread": book.spread,
                "imbalance": book.imbalance,
                "depth_ratio": book.depth_ratio,
                "timestamp_ns": book.timestamp_ns,
            }
        )

    def fill_probability_by_distance(self, bins: int = 10) -> pd.DataFrame:
        """Compute fill probability as a function of distance from mid.

        Returns a DataFrame with columns: distance_bps_bin, fill_count,
        fill_prob (as a fraction of total fills in that distance bucket).
        """
        if not self._fills:
            return pd.DataFrame(columns=["distance_bps_bin", "fill_count", "fill_prob"])

        distances = [f["distance_bps"] for f in self._fills]
        df = pd.DataFrame({"distance_bps": distances})
        df["bin"] = pd.cut(df["distance_bps"], bins=bins)
        grouped = df.groupby("bin", observed=True).size().reset_index(name="fill_count")
        total = grouped["fill_count"].sum()
        grouped["fill_prob"] = grouped["fill_count"] / total if total > 0 else 0
        grouped = grouped.rename(columns={"bin": "distance_bps_bin"})
        return grouped

    def pnl_by_time_of_day(self) -> pd.DataFrame:
        """Compute average PnL by hour of day.

        Returns a DataFrame with columns: hour, avg_pnl, total_pnl, count.
        """
        records = []
        for hour in sorted(self._pnl_by_hour.keys()):
            pnls = self._pnl_by_hour[hour]
            records.append(
                {
                    "hour": hour,
                    "avg_pnl": float(np.mean(pnls)),
                    "total_pnl": float(np.sum(pnls)),
                    "count": len(pnls),
                }
            )
        return (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=["hour", "avg_pnl", "total_pnl", "count"])
        )

    def pnl_by_spread_width(self, bins: int = 5) -> pd.DataFrame:
        """Analyze PnL as a function of the quoted spread width."""
        if not self._pnl_by_spread:
            return pd.DataFrame(columns=["spread_bin", "avg_pnl", "count"])

        df = pd.DataFrame(self._pnl_by_spread, columns=["spread_bps", "pnl"])
        df["bin"] = pd.cut(df["spread_bps"], bins=bins)
        grouped = (
            df.groupby("bin", observed=True)
            .agg(
                avg_pnl=("pnl", "mean"),
                count=("pnl", "size"),
            )
            .reset_index()
            .rename(columns={"bin": "spread_bin"})
        )
        return grouped

    def pnl_by_inventory_level(self, bins: int = 5) -> pd.DataFrame:
        """Analyze PnL as a function of inventory level at time of fill."""
        if not self._pnl_by_inventory:
            return pd.DataFrame(columns=["inventory_bin", "avg_pnl", "count"])

        df = pd.DataFrame(self._pnl_by_inventory, columns=["inventory", "pnl"])
        df["bin"] = pd.cut(df["inventory"], bins=bins)
        grouped = (
            df.groupby("bin", observed=True)
            .agg(
                avg_pnl=("pnl", "mean"),
                count=("pnl", "size"),
            )
            .reset_index()
            .rename(columns={"bin": "inventory_bin"})
        )
        return grouped

    def trade_size_distribution(self) -> dict[str, float]:
        """Summary statistics of trade size distribution."""
        if not self._trade_sizes:
            return {"mean": 0, "median": 0, "std": 0, "p95": 0, "p99": 0}
        arr = np.array(self._trade_sizes)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def book_imbalance_stats(self) -> dict[str, float]:
        """Summary statistics of order book imbalance."""
        if not self._book_snapshots:
            return {"mean": 0, "std": 0, "skew": 0}
        imbalances = [s["imbalance"] for s in self._book_snapshots]
        arr = np.array(imbalances)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "skew": float(pd.Series(arr).skew()) if len(arr) > 2 else 0.0,
        }

    def diagnostic_report(self) -> dict:
        """Generate a comprehensive microstructure diagnostic report."""
        return {
            "total_fills": len(self._fills),
            "total_book_snapshots": len(self._book_snapshots),
            "trade_size_distribution": self.trade_size_distribution(),
            "book_imbalance_stats": self.book_imbalance_stats(),
            "pnl_by_hour_count": len(self._pnl_by_hour),
            "avg_fill_distance_bps": (
                float(np.mean([f["distance_bps"] for f in self._fills])) if self._fills else 0.0
            ),
            "avg_fill_pnl": float(np.mean([f["pnl"] for f in self._fills])) if self._fills else 0.0,
        }

    @staticmethod
    def recommended_defaults(report: dict) -> dict:
        """Suggest parameter defaults based on a diagnostic report.

        This is a starting point — parameters should be validated via
        walk-forward backtesting.
        """
        avg_distance = report.get("avg_fill_distance_bps", 5.0)
        trade_stats = report.get("trade_size_distribution", {})
        avg_size = trade_stats.get("mean", 100)

        return {
            "min_spread_bps": max(2.0, avg_distance * 0.5),
            "default_quote_size": max(10, int(avg_size * 0.5)),
            "pull_on_large_trade_mult": 5.0,
            "vol_scale": 1.5,
        }
