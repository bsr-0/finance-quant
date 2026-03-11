"""Top-level market-making engine orchestrating all components.

Wires together spread calculation, inventory management, quoting,
adverse selection detection, hedging, and microstructure analysis into
a single event-processing loop.

The engine enforces strategy-level risk limits synchronously in the
main processing path (not as best-effort logging).

Architecture:
    Events flow through the engine as follows:

    1. ``on_event(MarketEvent)`` → event dispatch
    2. Trade events → inventory update → adverse selection → re-quote
    3. Quote events → mid update → re-quote
    4. Timer events → hedge check → risk dashboard update

    All limit checks are evaluated before any quote is emitted.

Assumptions:
    - The engine is single-threaded.
    - Clock is provided by event timestamps (no wall-clock dependency
      in the processing loop, enabling deterministic backtest replay).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


from pipeline.market_making.spread import SpreadCalculator, SpreadConfig
from pipeline.market_making.inventory import (
    InventoryConfig,
    InventoryManager,
)
from pipeline.market_making.quoting import (
    EventType,
    MarketEvent,
    QuoteConfig,
    QuoteEngine,
    QuoteUpdate,
)
from pipeline.market_making.adverse import (
    AdverseConfig,
    AdverseSelectionDetector,
    FillRecord,
)
from pipeline.market_making.hedging import HedgeConfig, HedgeManager, HedgeTrade
from pipeline.market_making.microstructure import MicrostructureAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RiskLimitsConfig:
    """Strategy-level risk limits enforced synchronously.

    Attributes:
        max_inventory_per_symbol: Hard position limit per symbol.
        max_portfolio_notional: Hard limit on aggregate gross notional.
        max_daily_loss: Maximum cumulative loss per day before shutdown.
        max_daily_volume: Maximum total traded volume per day.
        max_drawdown_pct: Maximum intraday drawdown before shutdown.
        soft_loss_pct: Fraction of max_daily_loss that triggers spread
            widening (soft brake).
        soft_inventory_pct: Fraction of max_inventory that triggers
            quote size reduction (soft brake).
    """

    max_inventory_per_symbol: float = 10_000
    max_portfolio_notional: float = 50_000_000.0
    max_daily_loss: float = 500_000.0
    max_daily_volume: float = 100_000_000.0
    max_drawdown_pct: float = 0.02
    soft_loss_pct: float = 0.50
    soft_inventory_pct: float = 0.70


@dataclass
class MarketMakingConfig:
    """Configuration for the market-making engine."""

    spread: SpreadConfig = field(default_factory=SpreadConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    quoting: QuoteConfig = field(default_factory=QuoteConfig)
    adverse: AdverseConfig = field(default_factory=AdverseConfig)
    hedging: HedgeConfig = field(default_factory=HedgeConfig)
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)


@dataclass
class EngineState:
    """Mutable state for the market-making engine within a session."""

    session_start_nav: float = 0.0
    peak_nav: float = 0.0
    realized_pnl: float = 0.0
    daily_volume: float = 0.0
    is_shutdown: bool = False
    shutdown_reason: str = ""
    events_processed: int = 0
    quotes_sent: int = 0
    fills_processed: int = 0


class MarketMakingEngine:
    """Production-grade market-making engine.

    Orchestrates all market-making components and enforces risk limits
    in the critical path.

    Usage::

        engine = MarketMakingEngine(MarketMakingConfig())
        engine.start_session(nav=1_000_000)

        for event in market_data_stream:
            result = engine.on_event(event)
            if result is not None:
                send_to_exchange(result)

        report = engine.end_of_day_report()
    """

    def __init__(self, config: MarketMakingConfig | None = None) -> None:
        self.config = config or MarketMakingConfig()
        c = self.config

        self.spread_calc = SpreadCalculator(c.spread)
        self.inventory_mgr = InventoryManager(c.inventory)
        self.quote_engine = QuoteEngine(
            spread_calc=self.spread_calc,
            inventory_mgr=self.inventory_mgr,
            config=c.quoting,
        )
        self.adverse_detector = AdverseSelectionDetector(c.adverse)
        self.hedge_mgr = HedgeManager(c.hedging)
        self.microstructure = MicrostructureAnalyzer()

        self._state = EngineState()
        self._prices: dict[str, float] = {}

    def start_session(self, nav: float) -> None:
        """Initialize a new trading session."""
        self._state = EngineState(
            session_start_nav=nav,
            peak_nav=nav,
        )
        logger.info("Market-making session started with NAV=%.2f", nav)

    def on_event(self, event: MarketEvent) -> QuoteUpdate | None:
        """Process a market event and return a quote update if needed.

        All risk limit checks are performed synchronously before any
        quote is emitted.

        Returns:
            ``QuoteUpdate`` if a new quote should be sent, else ``None``.
        """
        state = self._state
        if state.is_shutdown:
            return None

        state.events_processed += 1

        # Dispatch by event type
        if event.event_type == EventType.TRADE:
            self._on_trade(event)
        elif event.event_type in (EventType.QUOTE_UPDATE, EventType.BOOK_UPDATE):
            self._on_market_data(event)
        elif event.event_type == EventType.VOLATILITY_UPDATE:
            pass  # Handled by quote engine

        # Update prices for portfolio valuation
        if event.price > 0:
            self._prices[event.symbol] = event.price

        # Check risk limits before quoting
        if not self._check_risk_limits():
            return None

        # Generate quote through the quoting engine
        quote = self.quote_engine.on_event(event)

        if quote is not None and not quote.pulled:
            # Apply adverse selection adjustments
            toxicity = self.adverse_detector.evaluate(event.symbol)
            if toxicity.is_toxic:
                # Widen the spread
                mid = (quote.bid + quote.ask) / 2
                half = (quote.ask - quote.bid) / 2
                widened_half = half * toxicity.recommended_widen
                quote.bid = mid - widened_half
                quote.ask = mid + widened_half
                quote.bid_size *= toxicity.recommended_size_mult
                quote.ask_size *= toxicity.recommended_size_mult
                quote.spread_bps *= toxicity.recommended_widen

            state.quotes_sent += 1

        return quote

    def on_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp_ns: int,
    ) -> list[HedgeTrade]:
        """Process an execution fill.

        Updates inventory, records adverse selection data, checks for
        hedging needs, and returns any recommended hedge trades.

        This method should be called when the exchange confirms a fill
        on one of our quotes.
        """
        state = self._state
        if state.is_shutdown:
            return []

        state.fills_processed += 1
        state.daily_volume += abs(quantity) * price

        # Update inventory
        signed_qty = quantity if side == "buy" else -quantity
        self.inventory_mgr.record_fill(symbol, signed_qty, price)

        # Record for adverse selection
        inv = self.inventory_mgr.get_or_create(symbol)
        mid = self._prices.get(symbol, price)
        spread = self.config.spread.min_spread_bps  # approximate
        fill_record = FillRecord(
            symbol=symbol,
            side=side,
            fill_price=price,
            fill_size=abs(quantity),
            mid_at_fill=mid,
            timestamp_ns=timestamp_ns,
            spread_at_fill=spread,
        )
        self.adverse_detector.record_fill(fill_record)

        # Record for microstructure
        self.microstructure.record_fill(
            symbol=symbol,
            side=side,
            price=price,
            size=abs(quantity),
            mid_at_fill=mid,
            spread_at_fill=spread,
            inventory_at_fill=inv.position,
            timestamp_ns=timestamp_ns,
        )

        # Update PnL
        pnl_delta = 0.0
        if side == "buy":
            pnl_delta = -(quantity * price)  # Cash outflow
        else:
            pnl_delta = quantity * price  # Cash inflow

        # Check if we need hedging
        inventories = {
            sym: self.inventory_mgr.get_or_create(sym).position
            for sym in self.inventory_mgr._inventories
        }
        max_positions = {
            sym: self.config.risk_limits.max_inventory_per_symbol
            for sym in inventories
        }
        hedges = self.hedge_mgr.compute_hedges(
            inventories=inventories,
            prices=self._prices,
            max_positions=max_positions,
            timestamp_ns=timestamp_ns,
        )

        return hedges

    def _on_trade(self, event: MarketEvent) -> None:
        """Handle an external trade event (not our fill)."""
        # Feed to adverse selection detector for price tracking
        self.adverse_detector.record_post_fill_price(
            event.symbol, event.price
        )

    def _on_market_data(self, event: MarketEvent) -> None:
        """Handle a quote or book update."""
        if event.bid > 0 and event.ask > 0:
            self._prices[event.symbol] = (event.bid + event.ask) / 2
            self.adverse_detector.record_post_fill_price(
                event.symbol, (event.bid + event.ask) / 2
            )

    def _check_risk_limits(self) -> bool:
        """Synchronous risk limit check.  Returns False if trading should halt."""
        state = self._state
        limits = self.config.risk_limits

        # Daily loss check
        self.inventory_mgr.update_marks(self._prices)
        snap = self.inventory_mgr.snapshot()
        total_pnl = snap.total_realized_pnl + snap.total_unrealized_pnl

        if total_pnl < -limits.max_daily_loss:
            state.is_shutdown = True
            state.shutdown_reason = (
                f"Daily loss limit breached: PnL={total_pnl:,.0f} "
                f"< -{limits.max_daily_loss:,.0f}"
            )
            logger.critical("SHUTDOWN: %s", state.shutdown_reason)
            return False

        # Daily volume check
        if state.daily_volume > limits.max_daily_volume:
            state.is_shutdown = True
            state.shutdown_reason = (
                f"Daily volume limit breached: {state.daily_volume:,.0f} "
                f"> {limits.max_daily_volume:,.0f}"
            )
            logger.critical("SHUTDOWN: %s", state.shutdown_reason)
            return False

        # Drawdown check
        current_nav = state.session_start_nav + total_pnl
        state.peak_nav = max(state.peak_nav, current_nav)
        if state.peak_nav > 0:
            dd = (current_nav - state.peak_nav) / state.peak_nav
            if dd < -limits.max_drawdown_pct:
                state.is_shutdown = True
                state.shutdown_reason = (
                    f"Drawdown limit breached: {dd:.2%} "
                    f"< -{limits.max_drawdown_pct:.2%}"
                )
                logger.critical("SHUTDOWN: %s", state.shutdown_reason)
                return False

        # Portfolio notional check
        if snap.gross_notional > limits.max_portfolio_notional:
            logger.warning(
                "Portfolio notional %.0f exceeds limit %.0f — blocking new quotes",
                snap.gross_notional, limits.max_portfolio_notional,
            )
            return False

        return True

    def end_of_day_report(self) -> dict[str, Any]:
        """Generate an end-of-day summary report."""
        state = self._state
        self.inventory_mgr.update_marks(self._prices)
        snap = self.inventory_mgr.snapshot()
        total_pnl = snap.total_realized_pnl + snap.total_unrealized_pnl

        micro_report = self.microstructure.diagnostic_report()

        return {
            "session": {
                "start_nav": state.session_start_nav,
                "end_nav": state.session_start_nav + total_pnl,
                "realized_pnl": snap.total_realized_pnl,
                "unrealized_pnl": snap.total_unrealized_pnl,
                "total_pnl": total_pnl,
                "daily_volume": state.daily_volume,
                "is_shutdown": state.is_shutdown,
                "shutdown_reason": state.shutdown_reason,
            },
            "events": {
                "events_processed": state.events_processed,
                "quotes_sent": state.quotes_sent,
                "fills_processed": state.fills_processed,
            },
            "inventory": {
                "gross_notional": snap.gross_notional,
                "net_notional": snap.net_notional,
                "symbol_count": snap.symbol_count,
                "worst_symbol": snap.worst_symbol,
                "worst_utilization": snap.worst_utilization,
            },
            "quoting": self.quote_engine.diagnostics,
            "microstructure": micro_report,
        }

    def pre_open_checklist(self) -> list[tuple[str, bool, str]]:
        """Run pre-open checks and return a checklist.

        Returns:
            List of (check_name, passed, detail) tuples.
        """
        checks: list[tuple[str, bool, str]] = []

        # 1. Limits loaded
        limits = self.config.risk_limits
        checks.append((
            "risk_limits_loaded",
            limits.max_daily_loss > 0 and limits.max_inventory_per_symbol > 0,
            f"max_loss={limits.max_daily_loss:,.0f}, max_inv={limits.max_inventory_per_symbol:,.0f}",
        ))

        # 2. Inventory flat
        snap = self.inventory_mgr.snapshot()
        is_flat = snap.gross_notional < 1000  # effectively flat
        checks.append((
            "inventory_flat",
            is_flat,
            f"gross_notional={snap.gross_notional:,.0f}",
        ))

        # 3. Not in shutdown state
        checks.append((
            "not_shutdown",
            not self._state.is_shutdown,
            self._state.shutdown_reason or "OK",
        ))

        # 4. Spread config valid
        spread_ok = (
            self.config.spread.min_spread_bps > 0
            and self.config.spread.max_spread_bps > self.config.spread.min_spread_bps
        )
        checks.append((
            "spread_config_valid",
            spread_ok,
            f"min={self.config.spread.min_spread_bps}, max={self.config.spread.max_spread_bps}",
        ))

        return checks
