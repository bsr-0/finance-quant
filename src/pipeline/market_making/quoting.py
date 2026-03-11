"""Event-driven quote generation and throttling for market-making.

Processes a stream of market events (trades, quote updates, volatility
changes) and produces updated bid/ask quotes with minimal latency.

Design:
    The ``QuoteEngine`` is the central component that wires together the
    spread calculator, inventory manager, and adverse selection detector.
    It operates on an event-by-event basis and enforces:

    - **Minimum time-in-force**: quotes are not updated more frequently
      than ``min_quote_life_ms`` to avoid excessive message rates.
    - **Pull triggers**: quotes are pulled entirely when large trades or
      sudden microstructure shifts are detected.
    - **Latency budget**: a timing wrapper logs when the quote computation
      exceeds the allowed budget.

Assumptions:
    - All timestamps are in nanoseconds (int64) for consistency with
      exchange-grade data.
    - The engine is single-threaded; concurrency is handled by the caller
      (e.g. an event loop or message queue).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pipeline.market_making.spread import SpreadCalculator, SpreadConfig
from pipeline.market_making.inventory import InventoryManager, InventoryConfig, InventoryLevel

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events processed by the quoting engine."""

    TRADE = "trade"
    QUOTE_UPDATE = "quote_update"
    BOOK_UPDATE = "book_update"
    VOLATILITY_UPDATE = "volatility_update"
    TIMER = "timer"


@dataclass
class MarketEvent:
    """A single market event to be processed.

    Attributes:
        event_type: Category of the event.
        symbol: Instrument this event pertains to.
        timestamp_ns: Event timestamp in nanoseconds since epoch.
        price: Relevant price (trade price, mid quote, etc.).
        quantity: Trade size or book quantity change.
        bid: Best bid price (for quote/book events).
        ask: Best ask price (for quote/book events).
        bid_size: Size resting at the best bid.
        ask_size: Size resting at the best ask.
        volatility: Updated volatility estimate (for vol events).
        extra: Arbitrary key-value data for extensibility.
    """

    event_type: EventType
    symbol: str
    timestamp_ns: int = 0
    price: float = 0.0
    quantity: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    volatility: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class QuoteConfig:
    """Configuration for the quoting engine.

    Attributes:
        min_quote_life_ms: Minimum time between quote updates (throttle).
        latency_budget_us: Maximum allowed computation time per event
            (microseconds).  Exceeded budgets are logged as warnings.
        pull_on_large_trade_mult: Pull quotes if a trade is larger than
            this multiple of recent average trade size.
        fade_ticks_on_pull: Number of ticks to fade the quote after a pull
            before re-quoting.
        default_quote_size: Default number of shares/contracts per side.
        max_quote_size: Maximum number per side.
        vol_update_ewm_span: EWM span for updating the volatility estimate
            from trade data.
        latency_warn_us: Threshold in microseconds for a latency warning.
    """

    min_quote_life_ms: float = 50.0
    latency_budget_us: float = 500.0
    pull_on_large_trade_mult: float = 5.0
    fade_ticks_on_pull: int = 2
    default_quote_size: float = 100.0
    max_quote_size: float = 10_000.0
    vol_update_ewm_span: int = 100
    latency_warn_us: float = 1000.0


@dataclass
class QuoteUpdate:
    """Output of the quoting engine: a two-sided quote to send."""

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp_ns: int
    spread_bps: float
    pulled: bool = False
    reason: str = ""
    latency_us: float = 0.0


class QuoteEngine:
    """Event-driven market-making quote engine.

    Processes ``MarketEvent`` objects and produces ``QuoteUpdate`` objects.
    Coordinates spread calculation, inventory management, and throttling.
    """

    def __init__(
        self,
        spread_calc: SpreadCalculator | None = None,
        inventory_mgr: InventoryManager | None = None,
        config: QuoteConfig | None = None,
        spread_config: SpreadConfig | None = None,
        inventory_config: InventoryConfig | None = None,
    ) -> None:
        self.config = config or QuoteConfig()
        self.spread_calc = spread_calc or SpreadCalculator(spread_config)
        self.inventory_mgr = inventory_mgr or InventoryManager(inventory_config)

        # Per-symbol state
        self._last_quote_time: dict[str, int] = {}
        self._last_trade_sizes: dict[str, list[float]] = {}
        self._current_vol: dict[str, float] = {}
        self._current_mid: dict[str, float] = {}
        self._pulled_until: dict[str, int] = {}
        self._event_count: int = 0
        self._total_latency_us: float = 0.0

    def on_event(self, event: MarketEvent) -> QuoteUpdate | None:
        """Process a single market event and return a quote update.

        Returns ``None`` if no quote change is needed (e.g. throttled,
        or event is not relevant).
        """
        t_start = time.perf_counter_ns()
        self._event_count += 1

        result = self._handle_event(event)

        elapsed_us = (time.perf_counter_ns() - t_start) / 1_000
        self._total_latency_us += elapsed_us

        if result is not None:
            result.latency_us = elapsed_us

        if elapsed_us > self.config.latency_warn_us:
            logger.warning(
                "Latency budget exceeded: %.0f us (budget: %.0f us) for %s %s",
                elapsed_us, self.config.latency_budget_us,
                event.event_type.value, event.symbol,
            )

        return result

    def _handle_event(self, event: MarketEvent) -> QuoteUpdate | None:
        sym = event.symbol

        if event.event_type == EventType.TRADE:
            return self._on_trade(event)
        elif event.event_type == EventType.QUOTE_UPDATE:
            return self._on_quote_update(event)
        elif event.event_type == EventType.BOOK_UPDATE:
            return self._on_book_update(event)
        elif event.event_type == EventType.VOLATILITY_UPDATE:
            self._current_vol[sym] = event.volatility
            return self._maybe_requote(sym, event.timestamp_ns)
        elif event.event_type == EventType.TIMER:
            return self._maybe_requote(sym, event.timestamp_ns)
        return None

    def _on_trade(self, event: MarketEvent) -> QuoteUpdate | None:
        sym = event.symbol

        # Track recent trade sizes for large-trade detection
        sizes = self._last_trade_sizes.setdefault(sym, [])
        sizes.append(abs(event.quantity))
        if len(sizes) > 200:
            sizes[:] = sizes[-200:]

        # Update mid estimate from trade price (simple EWM)
        if sym in self._current_mid:
            alpha = 2.0 / (self.config.vol_update_ewm_span + 1)
            self._current_mid[sym] = (
                alpha * event.price + (1 - alpha) * self._current_mid[sym]
            )
        else:
            self._current_mid[sym] = event.price

        # Check for large trade → pull quotes
        if len(sizes) >= 10:
            avg_size = float(np.mean(sizes[-50:]))
            if avg_size > 0 and abs(event.quantity) > self.config.pull_on_large_trade_mult * avg_size:
                pull_duration_ns = int(self.config.min_quote_life_ms * 1_000_000 * 2)
                self._pulled_until[sym] = event.timestamp_ns + pull_duration_ns
                logger.info(
                    "PULL %s: large trade size=%.0f vs avg=%.0f",
                    sym, abs(event.quantity), avg_size,
                )
                return QuoteUpdate(
                    symbol=sym, bid=0, ask=0, bid_size=0, ask_size=0,
                    timestamp_ns=event.timestamp_ns, spread_bps=0,
                    pulled=True, reason="large_trade",
                )

        return self._maybe_requote(sym, event.timestamp_ns)

    def _on_quote_update(self, event: MarketEvent) -> QuoteUpdate | None:
        sym = event.symbol
        if event.bid > 0 and event.ask > 0:
            self._current_mid[sym] = (event.bid + event.ask) / 2
        return self._maybe_requote(sym, event.timestamp_ns)

    def _on_book_update(self, event: MarketEvent) -> QuoteUpdate | None:
        return self._maybe_requote(event.symbol, event.timestamp_ns)

    def _maybe_requote(self, symbol: str, timestamp_ns: int) -> QuoteUpdate | None:
        """Re-compute and return a quote if enough time has passed."""
        cfg = self.config
        min_life_ns = int(cfg.min_quote_life_ms * 1_000_000)

        # Check pull state
        if symbol in self._pulled_until:
            if timestamp_ns < self._pulled_until[symbol]:
                return None
            del self._pulled_until[symbol]

        # Throttle: don't re-quote too frequently
        last_time = self._last_quote_time.get(symbol, 0)
        if timestamp_ns - last_time < min_life_ns:
            return None

        # Need a fair value to quote
        mid = self._current_mid.get(symbol)
        if mid is None or mid <= 0:
            return None

        vol = self._current_vol.get(symbol, 0.02)
        inv_norm = self.inventory_mgr.normalized_inventory(symbol)
        inv_level = self.inventory_mgr.get_inventory_level(symbol)

        # Order-book imbalance (if available from last book event)
        # For now, default to 0; real implementation would track
        # bid/ask sizes from book updates.
        imbalance = 0.0

        # Stress level (simple heuristic: vol > 50% annualized = stress)
        stress = min(1.0, max(0.0, (vol - 0.30) / 0.30))

        quoted = self.spread_calc.compute(
            fair_value=mid,
            volatility=vol,
            inventory_normalized=inv_norm,
            order_book_imbalance=imbalance,
            stress_level=stress,
        )

        # Size adjustment based on inventory level
        size_mult = self.inventory_mgr.quote_size_multiplier(symbol)
        base_size = cfg.default_quote_size * size_mult

        # At BREACH level, only quote on the side that reduces inventory
        bid_size = base_size
        ask_size = base_size
        if inv_level == InventoryLevel.BREACH:
            inv = self.inventory_mgr.get_or_create(symbol)
            if inv.position > 0:
                bid_size = 0.0  # Don't buy more
            else:
                ask_size = 0.0  # Don't sell more

        bid_size = min(bid_size, cfg.max_quote_size)
        ask_size = min(ask_size, cfg.max_quote_size)

        self._last_quote_time[symbol] = timestamp_ns

        return QuoteUpdate(
            symbol=symbol,
            bid=quoted.bid,
            ask=quoted.ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ns=timestamp_ns,
            spread_bps=quoted.spread_bps,
        )

    @property
    def diagnostics(self) -> dict:
        """Return engine-level diagnostics."""
        avg_latency = (
            self._total_latency_us / self._event_count
            if self._event_count > 0
            else 0.0
        )
        return {
            "event_count": self._event_count,
            "avg_latency_us": avg_latency,
            "symbols_quoted": len(self._current_mid),
            "symbols_pulled": len(self._pulled_until),
        }
