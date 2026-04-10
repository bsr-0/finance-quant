"""Polymarket prediction market extractor."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.extract._base import HttpClientMixin
from pipeline.historical.coverage import compute_polymarket_coverage, record_coverage_metrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class PolymarketExtractor(HttpClientMixin):
    """Extract data from Polymarket prediction market."""

    def __init__(self):
        settings = get_settings().polymarket
        self.clob_url = settings.base_url
        self.gamma_url = settings.gamma_url
        self.rate_limit = settings.rate_limit_per_sec
        self.universe_mode = settings.universe_mode
        self.top_volume_limit = settings.top_volume_limit
        self.explicit_markets = settings.explicit_markets
        self.orderbook_snapshot_freq = settings.orderbook_snapshot_freq
        self.trades_page_size = settings.trades_page_size
        self.trades_max_pages = settings.trades_max_pages
        self.client = httpx.Client(timeout=30.0)
        self._last_request_time: float | None = None

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = 1.0 / self.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, url: str, params: dict | None = None) -> dict | list:
        """Make API request with retry logic."""
        self._rate_limit()
        response = self.client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        sort: str = "volume",
    ) -> list[dict]:
        """Get list of markets from Gamma API."""
        url = f"{self.gamma_url}/markets"
        params = {
            "limit": limit,
            "offset": offset,
            "active": active,
            "closed": closed,
            "sort": sort,
        }
        data = self._make_request(url, params)
        # Gamma API may return a JSON array directly or a dict with a "markets" key
        if isinstance(data, list):
            return data
        return data.get("markets", [])

    def get_market(self, market_id: str) -> dict | None:
        """Get detailed market information."""
        url = f"{self.gamma_url}/markets/{market_id}"
        try:
            return self._make_request(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def select_markets(self, max_markets: int | None = None) -> list[dict]:
        """Select markets based on configured universe mode."""
        markets: list[dict] = []

        if max_markets is None and self.universe_mode == "top_volume":
            max_markets = self.top_volume_limit

        if self.universe_mode == "explicit_list" and self.explicit_markets:
            for market_id in self.explicit_markets:
                market = self.get_market(market_id)
                if market:
                    markets.append(market)
        else:
            offset = 0
            while True:
                batch = self.get_markets(limit=20, offset=offset, active=True)
                if not batch:
                    break
                markets.extend(batch)
                offset += len(batch)
                if len(batch) < 20:
                    break
                if (
                    self.universe_mode == "top_volume"
                    and max_markets is not None
                    and len(markets) >= max_markets
                ):
                    break

        if self.universe_mode == "top_volume":
            markets = markets[:max_markets] if max_markets is not None else markets
        elif max_markets is not None:
            markets = markets[:max_markets]

        return markets

    def get_all_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch all markets for coverage auditing."""
        markets: list[dict] = []
        offset = 0
        while True:
            batch = self.get_markets(limit=limit, offset=offset, active=active, closed=closed)
            if not batch:
                break
            markets.extend(batch)
            offset += len(batch)
            if len(batch) < limit:
                break
        return markets

    def get_market_trades(
        self, market_id: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[dict]:
        """Get trades for a market."""
        url = f"{self.clob_url}/trades"
        trades: list[dict] = []
        cursor: str | None = None
        offset = 0

        for _ in range(self.trades_max_pages):
            params: dict = {"market": market_id, "limit": self.trades_page_size}
            if cursor:
                params["cursor"] = cursor
            else:
                params["offset"] = offset

            data = self._make_request(url, params)
            batch = data.get("trades") or data.get("data") or []
            if not batch:
                break

            trades.extend(batch)

            cursor = (
                data.get("next_cursor")
                or data.get("nextCursor")
                or data.get("cursor")
                or data.get("next")
            )
            if cursor:
                continue

            offset += len(batch)
            if len(batch) < self.trades_page_size:
                break

        # Filter by date if specified
        if start_date or end_date:
            filtered_trades = []
            for trade in trades:
                trade_time = datetime.fromtimestamp(trade.get("timestamp", 0), tz=UTC)
                if start_date and trade_time < start_date:
                    continue
                if end_date and trade_time > end_date:
                    continue
                filtered_trades.append(trade)
            trades = filtered_trades

        return trades

    def get_orderbook(self, market_id: str) -> dict | None:
        """Get current orderbook for a market."""
        url = f"{self.clob_url}/book"
        params = {"market": market_id}
        try:
            return self._make_request(url, params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_price_history(
        self, market_id: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """Get historical price data for a market."""
        # Polymarket doesn't have a direct price history endpoint
        # We derive prices from trades
        trades = self.get_market_trades(market_id, start_date, end_date)

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df.get("size", 0), errors="coerce")

        return df

    def extract_to_raw(
        self,
        output_dir: Path,
        start_date: date | None = None,
        end_date: date | None = None,
        max_markets: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, list[Path]]:
        """Extract Polymarket data to raw lake."""
        output_dir = Path(output_dir) / "polymarket"
        markets_dir = output_dir / "markets"
        prices_dir = output_dir / "prices"
        trades_dir = output_dir / "trades"
        orderbooks_dir = output_dir / "orderbooks"

        for d in [markets_dir, prices_dir, trades_dir, orderbooks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, list[Path]] = {
            "markets": [],
            "prices": [],
            "trades": [],
            "orderbooks": [],
        }

        # Get markets based on universe mode
        logger.info("Fetching Polymarket markets...")
        markets = self.select_markets(max_markets=max_markets)

        logger.info(f"Found {len(markets)} markets to extract (mode={self.universe_mode})")

        # Universe audit: measure selection bias and coverage
        fixes = get_settings().historical_fixes
        if fixes.polymarket_universe_audit:
            try:
                all_markets = self.get_all_markets(active=True, closed=False)
                metrics = compute_polymarket_coverage(markets, all_markets)
                if max_markets is not None:
                    metrics["max_markets"] = float(max_markets)
                if fixes.selection_weight_mode == "volume":
                    metrics["selection_weighted_pct"] = metrics.get("volume_share_pct", 0.0)
                else:
                    metrics["selection_weighted_pct"] = metrics.get("selection_coverage_pct", 0.0)
                record_coverage_metrics("polymarket", metrics)
            except Exception as exc:
                logger.warning(f"Polymarket coverage audit failed: {exc}")

        # Extract each market
        start_dt = datetime.combine(start_date, datetime.min.time()) if start_date else None
        end_dt = datetime.combine(end_date, datetime.max.time()) if end_date else None

        for market in markets:
            market_id = market.get("id") or market.get("marketId")
            if not market_id:
                continue

            try:
                # Save market metadata
                market_df = pd.DataFrame([market])
                market_df["extracted_at"] = datetime.now(UTC)
                market_df["run_id"] = run_id

                market_path = markets_dir / f"{market_id}.parquet"
                market_df.to_parquet(market_path, index=False)
                saved_files["markets"].append(market_path)

                # Get trades
                trades = self.get_market_trades(market_id, start_dt, end_dt)
                if trades:
                    trades_df = pd.DataFrame(trades)
                    if "timestamp" in trades_df.columns:
                        trades_df["timestamp"] = pd.to_datetime(
                            trades_df["timestamp"],
                            unit="s",
                            utc=True,
                        )
                    trades_df["market_id"] = market_id
                    trades_df["extracted_at"] = datetime.now(UTC)
                    trades_df["run_id"] = run_id

                    trades_path = trades_dir / f"{market_id}_trades.parquet"
                    trades_df.to_parquet(trades_path, index=False)
                    saved_files["trades"].append(trades_path)
                    logger.info(f"Saved {len(trades_df)} trades for {market_id}")

                # Get price history
                prices_df = self.get_price_history(market_id, start_dt, end_dt)
                if not prices_df.empty:
                    prices_df["market_id"] = market_id
                    prices_df["extracted_at"] = datetime.now(UTC)
                    prices_df["run_id"] = run_id

                    prices_path = prices_dir / f"{market_id}_prices.parquet"
                    prices_df.to_parquet(prices_path, index=False)
                    saved_files["prices"].append(prices_path)

                # Capture orderbook snapshot (current)
                if self.orderbook_snapshot_freq and self.orderbook_snapshot_freq.lower() != "off":
                    orderbook = self.get_orderbook(market_id)
                else:
                    orderbook = None

                if orderbook:
                    ob_ts = datetime.now(UTC)
                    bids = orderbook.get("bids") or []
                    asks = orderbook.get("asks") or []
                    best_bid = max((float(b[0]) for b in bids if len(b) >= 2), default=None)
                    best_ask = min((float(a[0]) for a in asks if len(a) >= 2), default=None)
                    spread = None
                    if best_bid is not None and best_ask is not None:
                        spread = float(best_ask) - float(best_bid)

                    ob_df = pd.DataFrame(
                        [
                            {
                                "market_id": market_id,
                                "ts": ob_ts,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "spread": spread,
                                "bids": bids,
                                "asks": asks,
                                "extracted_at": ob_ts,
                                "run_id": run_id,
                            }
                        ]
                    )
                    ob_path = orderbooks_dir / f"{market_id}_orderbook_{ob_ts:%Y%m%d%H%M%S}.parquet"
                    ob_df.to_parquet(ob_path, index=False)
                    saved_files["orderbooks"].append(ob_path)

            except Exception as e:
                logger.error(f"Error extracting market {market_id}: {e}")
                continue

        return saved_files


def extract_polymarket(
    output_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    max_markets: int | None = None,
    run_id: str | None = None,
) -> dict[str, list[Path]]:
    """CLI-friendly wrapper for Polymarket extraction."""
    extractor = PolymarketExtractor()

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None

    return extractor.extract_to_raw(
        output_dir=output_dir,
        start_date=start,
        end_date=end,
        max_markets=max_markets,
        run_id=run_id,
    )
