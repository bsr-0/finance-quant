"""Event-driven backtesting engine.

Processes market data as a chronological stream of events (trades, quotes,
fills, risk updates) rather than vectorized end-of-day bars.  The same
event-handling logic can be used in both backtest and live modes to avoid
strategy drift.

Architecture:
    The event loop processes events from a priority queue ordered by
    timestamp.  Events include:
    - MARKET_DATA: Price/quote updates from historical data.
    - ORDER: New order submissions from the strategy.
    - FILL: Execution confirmations (simulated in backtest, real in live).
    - RISK: Risk limit updates and checks.
    - TIMER: Periodic callbacks (e.g. hedge rebalancing).

    The queue is deterministic: events at the same timestamp are ordered
    by type priority, then by insertion order.

Assumptions:
    - All timestamps are comparable (pd.Timestamp or int nanoseconds).
    - The strategy interface is the same for backtest and live.
    - Randomness is seeded for reproducibility.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Priority for deterministic ordering of same-timestamp events."""

    MARKET_DATA = 0
    RISK_CHECK = 1
    TIMER = 2
    ORDER = 3
    FILL = 4


@dataclass(order=True)
class Event:
    """A single event in the backtest event stream.

    Events are ordered by (timestamp, priority, sequence) to ensure
    deterministic processing.

    Attributes:
        timestamp: Event time.
        priority: Event type priority for same-timestamp ordering.
        sequence: Insertion order for tie-breaking.
        event_type: Human-readable event category.
        data: Arbitrary payload.
    """

    timestamp: pd.Timestamp
    priority: int
    sequence: int
    event_type: str = field(compare=False, default="")
    data: dict = field(compare=False, default_factory=dict)


@dataclass
class Order:
    """A simulated order."""

    symbol: str
    side: str           # "buy" or "sell"
    quantity: float
    order_type: str     # "limit" or "market"
    limit_price: float = 0.0
    timestamp: pd.Timestamp | None = None
    order_id: str = ""
    filled: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class FillEvent:
    """A simulated fill."""

    order: Order
    fill_price: float
    fill_quantity: float
    timestamp: pd.Timestamp
    fees: float = 0.0


@dataclass
class EventEngineConfig:
    """Configuration for the event-driven backtest engine.

    Attributes:
        initial_capital: Starting cash balance.
        maker_fee_bps: Fee for passive (maker) fills in basis points.
        taker_fee_bps: Fee for aggressive (taker) fills in basis points.
        maker_rebate_bps: Rebate for providing liquidity (negative fee).
        slippage_bps: Additional slippage per fill in basis points.
        partial_fills: Whether to simulate partial fills based on depth.
        latency_ms: Simulated order-to-fill latency in milliseconds.
        seed: Random seed for reproducibility.
    """

    initial_capital: float = 1_000_000.0
    maker_fee_bps: float = 0.5
    taker_fee_bps: float = 3.0
    maker_rebate_bps: float = 0.2
    slippage_bps: float = 1.0
    partial_fills: bool = False
    latency_ms: float = 0.0
    seed: int = 42


class EventDrivenBacktester:
    """Event-driven backtesting engine.

    The engine maintains a priority queue of events and processes them
    chronologically.  The strategy interacts via callbacks.

    Usage::

        engine = EventDrivenBacktester(config)

        # Load market data as events
        engine.load_market_data(tick_df)

        # Register strategy callbacks
        engine.on_market_data = my_strategy.on_data
        engine.on_fill = my_strategy.on_fill

        # Run
        result = engine.run()
    """

    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Event queue (min-heap)
        self._event_queue: list[Event] = []
        self._sequence: int = 0

        # Portfolio state
        self._cash: float = self.config.initial_capital
        self._positions: dict[str, float] = {}  # symbol → quantity
        self._avg_costs: dict[str, float] = {}   # symbol → avg cost
        self._realized_pnl: float = 0.0
        self._total_fees: float = 0.0

        # Latest market data
        self._last_price: dict[str, float] = {}
        self._last_bid: dict[str, float] = {}
        self._last_ask: dict[str, float] = {}

        # History for analysis
        self._snapshots: list[dict] = []
        self._fills: list[FillEvent] = []
        self._orders: list[Order] = []

        # Callbacks
        self.on_market_data: Callable[[dict], list[Order] | None] | None = None
        self.on_fill: Callable[[FillEvent], None] | None = None
        self.on_timer: Callable[[pd.Timestamp], list[Order] | None] | None = None

    def push_event(
        self,
        timestamp: pd.Timestamp,
        event_type: str,
        data: dict,
        priority: EventPriority = EventPriority.MARKET_DATA,
    ) -> None:
        """Push an event onto the queue."""
        self._sequence += 1
        event = Event(
            timestamp=timestamp,
            priority=int(priority),
            sequence=self._sequence,
            event_type=event_type,
            data=data,
        )
        heapq.heappush(self._event_queue, event)

    def load_market_data(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
    ) -> None:
        """Load historical data as market data events.

        Supports both tick-level and OHLCV bar data.

        Args:
            df: DataFrame indexed by timestamp with at least a 'price'
                or 'close' column.  Optional: bid, ask, volume, symbol.
            symbol: Symbol name (used if 'symbol' column not in df).
        """
        for ts, row in df.iterrows():
            sym = row.get("symbol", symbol or "UNKNOWN")
            data: dict[str, Any] = {"symbol": sym}

            # Support both tick and bar formats
            if "price" in row:
                data["price"] = float(row["price"])
            elif "close" in row:
                data["price"] = float(row["close"])
                for col in ("open", "high", "low", "volume"):
                    if col in row:
                        data[col] = float(row[col])

            if "bid" in row and "ask" in row:
                data["bid"] = float(row["bid"])
                data["ask"] = float(row["ask"])

            if "volume" in row:
                data["volume"] = float(row["volume"])

            self.push_event(
                timestamp=pd.Timestamp(ts),
                event_type="market_data",
                data=data,
            )

    def load_timer_events(
        self,
        timestamps: list[pd.Timestamp],
    ) -> None:
        """Schedule periodic timer events."""
        for ts in timestamps:
            self.push_event(
                timestamp=ts,
                event_type="timer",
                data={},
                priority=EventPriority.TIMER,
            )

    def submit_order(self, order: Order, current_time: pd.Timestamp) -> None:
        """Submit an order from the strategy.

        The order is placed on the event queue with simulated latency.
        """
        self._orders.append(order)
        fill_time = current_time + pd.Timedelta(
            milliseconds=self.config.latency_ms
        )
        self.push_event(
            timestamp=fill_time,
            event_type="order",
            data={"order": order},
            priority=EventPriority.ORDER,
        )

    def run(self) -> BacktestRunResult:
        """Execute the backtest by draining the event queue.

        Returns:
            ``BacktestRunResult`` with equity curve, trades, and metrics.
        """
        prev_nav = self.config.initial_capital

        while self._event_queue:
            event = heapq.heappop(self._event_queue)

            if event.event_type == "market_data":
                self._handle_market_data(event)
            elif event.event_type == "order":
                self._handle_order(event)
            elif event.event_type == "timer":
                self._handle_timer(event)

            # Record snapshot
            nav = self._compute_nav()
            daily_ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0
            self._snapshots.append({
                "timestamp": event.timestamp,
                "nav": nav,
                "cash": self._cash,
                "positions_value": nav - self._cash,
                "realized_pnl": self._realized_pnl,
                "total_fees": self._total_fees,
                "daily_return": daily_ret,
                "num_positions": sum(1 for v in self._positions.values() if v != 0),
            })
            prev_nav = nav

        return self._build_result()

    def _handle_market_data(self, event: Event) -> None:
        """Process a market data event."""
        data = event.data
        sym = data.get("symbol", "")
        if "price" in data:
            self._last_price[sym] = data["price"]
        if "bid" in data:
            self._last_bid[sym] = data["bid"]
        if "ask" in data:
            self._last_ask[sym] = data["ask"]

        # Invoke strategy callback
        if self.on_market_data:
            orders = self.on_market_data(data)
            if orders:
                for order in orders:
                    self.submit_order(order, event.timestamp)

    def _handle_order(self, event: Event) -> None:
        """Simulate order execution."""
        order: Order = event.data["order"]
        sym = order.symbol
        last_px = self._last_price.get(sym, 0)

        if last_px <= 0:
            order.rejected = True
            order.reject_reason = "No price available"
            return

        # Determine fill price
        if order.order_type == "market":
            if order.side == "buy":
                base_px = self._last_ask.get(sym, last_px)
            else:
                base_px = self._last_bid.get(sym, last_px)
            slippage = base_px * (self.config.slippage_bps / 10_000)
            fill_px = base_px + slippage if order.side == "buy" else base_px - slippage
            fee_bps = self.config.taker_fee_bps
        elif order.order_type == "limit":
            if order.side == "buy" and order.limit_price >= last_px:
                fill_px = order.limit_price
                fee_bps = self.config.maker_fee_bps
            elif order.side == "sell" and order.limit_price <= last_px:
                fill_px = order.limit_price
                fee_bps = self.config.maker_fee_bps
            else:
                # Limit not filled
                return
        else:
            return

        fill_qty = order.quantity
        notional = fill_qty * fill_px
        fees = notional * (fee_bps / 10_000)

        # Update cash and positions
        if order.side == "buy":
            self._cash -= notional + fees
            old_pos = self._positions.get(sym, 0)
            old_cost = self._avg_costs.get(sym, 0)
            if old_pos >= 0:
                total_cost = old_cost * old_pos + fill_px * fill_qty
                new_pos = old_pos + fill_qty
                self._avg_costs[sym] = total_cost / new_pos if new_pos > 0 else 0
            else:
                closed = min(abs(old_pos), fill_qty)
                pnl = (self._avg_costs.get(sym, fill_px) - fill_px) * closed
                self._realized_pnl += pnl
                new_pos = old_pos + fill_qty
                if new_pos > 0:
                    self._avg_costs[sym] = fill_px
            self._positions[sym] = new_pos
        else:
            self._cash += notional - fees
            old_pos = self._positions.get(sym, 0)
            if old_pos > 0:
                closed = min(old_pos, fill_qty)
                pnl = (fill_px - self._avg_costs.get(sym, fill_px)) * closed
                self._realized_pnl += pnl
                new_pos = old_pos - fill_qty
                if new_pos < 0:
                    self._avg_costs[sym] = fill_px
            else:
                old_cost = self._avg_costs.get(sym, 0)
                total_cost = abs(old_cost * old_pos) + fill_px * fill_qty
                new_pos = old_pos - fill_qty
                self._avg_costs[sym] = total_cost / abs(new_pos) if new_pos != 0 else 0
            self._positions[sym] = new_pos

        self._total_fees += fees

        # Record fill
        order.filled = True
        order.fill_price = fill_px
        order.fill_quantity = fill_qty
        fill_event = FillEvent(
            order=order,
            fill_price=fill_px,
            fill_quantity=fill_qty,
            timestamp=event.timestamp,
            fees=fees,
        )
        self._fills.append(fill_event)

        if self.on_fill:
            self.on_fill(fill_event)

    def _handle_timer(self, event: Event) -> None:
        """Process a timer event."""
        if self.on_timer:
            orders = self.on_timer(event.timestamp)
            if orders:
                for order in orders:
                    self.submit_order(order, event.timestamp)

    def _compute_nav(self) -> float:
        """Compute net asset value."""
        positions_value = sum(
            qty * self._last_price.get(sym, 0)
            for sym, qty in self._positions.items()
        )
        return self._cash + positions_value

    def _build_result(self) -> BacktestRunResult:
        """Build the final backtest result from accumulated state."""
        if not self._snapshots:
            return BacktestRunResult(
                equity_curve=pd.Series(dtype=float),
                fills=self._fills,
                orders=self._orders,
                final_positions=dict(self._positions),
                total_fees=self._total_fees,
                realized_pnl=self._realized_pnl,
            )

        eq_df = pd.DataFrame(self._snapshots)
        eq_df = eq_df.set_index("timestamp")
        # Deduplicate timestamps by keeping the last snapshot per timestamp
        eq_df = eq_df[~eq_df.index.duplicated(keep="last")]

        return BacktestRunResult(
            equity_curve=eq_df["nav"],
            fills=self._fills,
            orders=self._orders,
            final_positions=dict(self._positions),
            total_fees=self._total_fees,
            realized_pnl=self._realized_pnl,
            snapshot_df=eq_df,
        )


@dataclass
class BacktestRunResult:
    """Result of an event-driven backtest run."""

    equity_curve: pd.Series
    fills: list[FillEvent]
    orders: list[Order]
    final_positions: dict[str, float]
    total_fees: float
    realized_pnl: float
    snapshot_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1

    @property
    def sharpe_ratio(self) -> float:
        if self.snapshot_df.empty or "daily_return" not in self.snapshot_df.columns:
            return np.nan
        rets = self.snapshot_df["daily_return"].dropna()
        if len(rets) < 20 or rets.std() == 0:
            return np.nan
        return float(rets.mean() / rets.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak.replace(0, np.nan)
        return float(dd.min())

    def summary(self) -> dict[str, float]:
        eq = self.equity_curve
        if eq.empty:
            return {}
        return {
            "initial_nav": float(eq.iloc[0]),
            "final_nav": float(eq.iloc[-1]),
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_fills": len(self.fills),
            "total_fees": self.total_fees,
            "realized_pnl": self.realized_pnl,
        }
