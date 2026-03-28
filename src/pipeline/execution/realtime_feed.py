"""Real-time price feed via Alpaca WebSocket and REST polling fallback.

Provides continuous intraday price updates for stop enforcement and position
monitoring.  Two backends:

1. **WebSocket** (preferred): Connects to Alpaca's streaming data API for
   sub-second trade/quote updates.  Uses the free IEX feed by default.

2. **REST polling** (fallback): Periodically fetches latest trade prices via
   the Alpaca data REST API.  Works without WebSocket support but has higher
   latency (configurable interval, default 30s).

Usage::

    from pipeline.execution.realtime_feed import RealtimePriceFeed

    feed = RealtimePriceFeed.from_env(symbols=["AAPL", "MSFT"])
    feed.start()

    # Get latest price (non-blocking)
    quote = feed.get_latest("AAPL")
    if quote:
        print(f"AAPL: ${quote.price:.2f} at {quote.timestamp}")

    feed.stop()
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PriceQuote:
    """A real-time price observation."""

    symbol: str
    price: float
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = ""

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.price

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def age_seconds(self) -> float:
        return (datetime.now(UTC) - self.timestamp).total_seconds()


# ---------------------------------------------------------------------------
# Callback type for price updates
# ---------------------------------------------------------------------------
PriceCallback = Callable[[PriceQuote], None]


class RealtimePriceFeed:
    """Unified real-time price feed with WebSocket and polling backends.

    Thread-safe: all price reads/writes go through a lock.  The feed runs
    on a background daemon thread so it won't block the main process.
    """

    def __init__(
        self,
        symbols: list[str],
        api_key: str,
        secret_key: str,
        *,
        mode: str = "websocket",
        data_feed: str = "iex",
        poll_interval: int = 30,
        stale_threshold: int = 120,
        reconnect_max_retries: int = 5,
        reconnect_backoff_base: float = 2.0,
        on_price: PriceCallback | None = None,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._api_key = api_key
        self._secret_key = secret_key
        self._mode = mode
        self._data_feed = data_feed
        self._poll_interval = poll_interval
        self._stale_threshold = stale_threshold
        self._reconnect_max_retries = reconnect_max_retries
        self._reconnect_backoff_base = reconnect_backoff_base
        self._on_price = on_price

        self._lock = threading.Lock()
        self._latest: dict[str, PriceQuote] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._connected = threading.Event()
        self._stop_event = threading.Event()

    @classmethod
    def from_env(
        cls,
        symbols: list[str] | None = None,
        on_price: PriceCallback | None = None,
    ) -> RealtimePriceFeed:
        """Create a feed from environment variables and config."""
        settings = get_settings()
        rt = settings.realtime_feed

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for real-time feed"
            )

        syms = symbols or settings.prices.universe

        return cls(
            symbols=syms,
            api_key=api_key,
            secret_key=secret_key,
            mode=rt.mode,
            data_feed=rt.data_feed,
            poll_interval=rt.poll_interval_seconds,
            stale_threshold=rt.stale_threshold_seconds,
            reconnect_max_retries=rt.reconnect_max_retries,
            reconnect_backoff_base=rt.reconnect_backoff_base,
            on_price=on_price,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the feed on a background thread."""
        if self._running:
            logger.warning("Feed already running")
            return

        self._running = True
        self._stop_event.clear()

        if self._mode == "websocket":
            self._thread = threading.Thread(
                target=self._ws_loop, daemon=True, name="realtime-ws"
            )
        else:
            self._thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="realtime-poll"
            )

        self._thread.start()
        logger.info(
            "Real-time feed started: mode=%s, symbols=%d, feed=%s",
            self._mode,
            len(self._symbols),
            self._data_feed,
        )

    def stop(self) -> None:
        """Stop the feed gracefully."""
        self._running = False
        self._stop_event.set()

        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info("Real-time feed stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def get_latest(self, symbol: str) -> PriceQuote | None:
        """Get the latest price for a symbol (thread-safe)."""
        with self._lock:
            return self._latest.get(symbol.upper())

    def get_all_latest(self) -> dict[str, PriceQuote]:
        """Get all latest prices (thread-safe snapshot)."""
        with self._lock:
            return dict(self._latest)

    def get_price(self, symbol: str) -> float | None:
        """Convenience: get just the price float, or None if unavailable."""
        quote = self.get_latest(symbol)
        return quote.price if quote else None

    def is_stale(self, symbol: str) -> bool:
        """Check if the latest quote for a symbol is stale."""
        quote = self.get_latest(symbol)
        if quote is None:
            return True
        return quote.age_seconds > self._stale_threshold

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def add_symbols(self, symbols: list[str]) -> None:
        """Add symbols to the subscription (takes effect on reconnect)."""
        new = [s.upper() for s in symbols if s.upper() not in self._symbols]
        if new:
            self._symbols.extend(new)
            logger.info("Added symbols to feed: %s", new)

    # ------------------------------------------------------------------
    # Internal: update handler
    # ------------------------------------------------------------------

    def _update_price(self, quote: PriceQuote) -> None:
        with self._lock:
            self._latest[quote.symbol] = quote

        if self._on_price:
            try:
                self._on_price(quote)
            except Exception:
                logger.exception("Error in price callback for %s", quote.symbol)

    # ------------------------------------------------------------------
    # WebSocket backend
    # ------------------------------------------------------------------

    def _ws_loop(self) -> None:
        """WebSocket connection loop with automatic reconnection."""
        retries = 0

        while self._running and retries <= self._reconnect_max_retries:
            try:
                self._ws_connect_and_stream()
                retries = 0  # reset on clean disconnect
            except Exception as e:
                if not self._running:
                    break
                retries += 1
                delay = self._reconnect_backoff_base ** retries
                logger.warning(
                    "WebSocket disconnected (%s), retry %d/%d in %.1fs",
                    e, retries, self._reconnect_max_retries, delay,
                )
                self._connected.clear()
                if self._stop_event.wait(timeout=delay):
                    break

        if self._running and retries > self._reconnect_max_retries:
            logger.error(
                "WebSocket max retries exceeded, falling back to polling"
            )
            self._mode = "polling"
            self._poll_loop()

    def _ws_connect_and_stream(self) -> None:
        """Single WebSocket connection lifecycle."""
        import websocket

        ws_url = f"wss://stream.data.alpaca.markets/v2/{self._data_feed}"

        ws = websocket.WebSocket()
        ws.connect(ws_url)
        self._ws = ws

        # Authenticate
        auth_msg = {
            "action": "auth",
            "key": self._api_key,
            "secret": self._secret_key,
        }
        ws.send(json.dumps(auth_msg))

        # Read auth response
        response = json.loads(ws.recv())
        if isinstance(response, list):
            for msg in response:
                if msg.get("T") == "error":
                    raise RuntimeError(f"WebSocket auth failed: {msg}")
        logger.debug("WebSocket auth response: %s", response)

        # Subscribe to trades and quotes
        sub_msg = {
            "action": "subscribe",
            "trades": self._symbols,
            "quotes": self._symbols,
        }
        ws.send(json.dumps(sub_msg))

        sub_response = json.loads(ws.recv())
        logger.debug("WebSocket subscription response: %s", sub_response)
        self._connected.set()
        logger.info("WebSocket connected and subscribed to %d symbols", len(self._symbols))

        # Stream loop
        while self._running:
            try:
                raw = ws.recv()
                if not raw:
                    break
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    self._handle_ws_message(msg)
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                break

        ws.close()
        self._ws = None
        self._connected.clear()

    def _handle_ws_message(self, msg: dict[str, Any]) -> None:
        """Parse an Alpaca WebSocket message into a PriceQuote."""
        msg_type = msg.get("T", "")

        if msg_type == "t":  # trade
            ts = self._parse_alpaca_timestamp(msg.get("t", ""))
            quote = PriceQuote(
                symbol=msg["S"],
                price=float(msg["p"]),
                volume=int(msg.get("s", 0)),
                timestamp=ts,
                source="alpaca_ws_trade",
            )
            # Carry forward high/low from existing quote
            existing = self._latest.get(quote.symbol)
            if existing:
                quote.high = max(existing.high, quote.price)
                quote.low = min(existing.low, quote.price) if existing.low > 0 else quote.price
                quote.bid = existing.bid
                quote.ask = existing.ask
            else:
                quote.high = quote.price
                quote.low = quote.price
            self._update_price(quote)

        elif msg_type == "q":  # quote
            ts = self._parse_alpaca_timestamp(msg.get("t", ""))
            bid = float(msg.get("bp", 0))
            ask = float(msg.get("ap", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

            existing = self._latest.get(msg["S"])
            price = existing.price if existing else mid

            quote = PriceQuote(
                symbol=msg["S"],
                price=price if price > 0 else mid,
                bid=bid,
                ask=ask,
                high=existing.high if existing else 0.0,
                low=existing.low if existing else 0.0,
                volume=existing.volume if existing else 0,
                timestamp=ts,
                source="alpaca_ws_quote",
            )
            self._update_price(quote)

    @staticmethod
    def _parse_alpaca_timestamp(ts_str: str) -> datetime:
        """Parse an Alpaca RFC-3339 timestamp."""
        if not ts_str:
            return datetime.now(UTC)
        try:
            # Alpaca uses RFC-3339 like "2024-01-15T14:30:00.123456789Z"
            # Trim nanoseconds to microseconds for Python
            if "." in ts_str:
                base, frac = ts_str.split(".")
                frac = frac.rstrip("Z")[:6]
                ts_str = f"{base}.{frac}+00:00"
            elif ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return datetime.now(UTC)

    # ------------------------------------------------------------------
    # Polling backend
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """REST polling loop: fetch latest trades on an interval."""
        self._connected.set()
        logger.info(
            "Polling feed started: interval=%ds, symbols=%d",
            self._poll_interval, len(self._symbols),
        )

        while self._running:
            try:
                self._poll_once()
            except Exception:
                logger.exception("Polling error")

            if self._stop_event.wait(timeout=self._poll_interval):
                break

        self._connected.clear()

    def _poll_once(self) -> None:
        """Fetch latest trades for all symbols via Alpaca REST API."""
        # Alpaca data API: GET /v2/stocks/snapshots
        base_url = "https://data.alpaca.markets/v2/stocks/snapshots"
        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

        # Batch in groups of 50 (API limit)
        for i in range(0, len(self._symbols), 50):
            batch = self._symbols[i : i + 50]
            params = {
                "symbols": ",".join(batch),
                "feed": self._data_feed,
            }

            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(base_url, headers=headers, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                for symbol, snap in data.items():
                    trade = snap.get("latestTrade", {})
                    quote_data = snap.get("latestQuote", {})
                    daily_bar = snap.get("dailyBar", {})

                    price = float(trade.get("p", 0))
                    if price <= 0:
                        continue

                    ts = self._parse_alpaca_timestamp(trade.get("t", ""))

                    quote = PriceQuote(
                        symbol=symbol,
                        price=price,
                        bid=float(quote_data.get("bp", 0)),
                        ask=float(quote_data.get("ap", 0)),
                        high=float(daily_bar.get("h", price)),
                        low=float(daily_bar.get("l", price)),
                        volume=int(daily_bar.get("v", 0)),
                        timestamp=ts,
                        source="alpaca_rest_snapshot",
                    )
                    self._update_price(quote)

            except httpx.HTTPStatusError as e:
                logger.warning("Snapshot API error for batch %d: %s", i, e)
            except Exception:
                logger.exception("Failed to poll snapshot batch %d", i)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_for_positions(
        cls,
        position_symbols: list[str],
        on_price: PriceCallback | None = None,
    ) -> RealtimePriceFeed:
        """Create a feed for currently held positions + SPY (for regime)."""
        symbols = list(set(position_symbols + ["SPY"]))
        return cls.from_env(symbols=symbols, on_price=on_price)
