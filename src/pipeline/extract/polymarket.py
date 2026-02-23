"""Polymarket prediction market extractor."""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class PolymarketExtractor:
    """Extract data from Polymarket prediction market."""
    
    def __init__(self):
        settings = get_settings().polymarket
        self.clob_url = settings.base_url
        self.gamma_url = settings.gamma_url
        self.rate_limit = settings.rate_limit_per_sec
        self.client = httpx.Client(timeout=30.0)
        self._last_request_time: Optional[float] = None
    
    def __del__(self):
        self.client.close()
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = 1.0 / self.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """Make API request with retry logic."""
        self._rate_limit()
        response = self.client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()
    
    def get_markets(
        self, 
        limit: int = 100, 
        offset: int = 0,
        active: bool = True
    ) -> List[dict]:
        """Get list of markets from Gamma API."""
        url = f"{self.gamma_url}/markets"
        params = {
            "limit": limit,
            "offset": offset,
            "active": active,
            "closed": False,
            "sort": "volume"
        }
        data = self._make_request(url, params)
        return data.get("markets", [])
    
    def get_market(self, market_id: str) -> Optional[dict]:
        """Get detailed market information."""
        url = f"{self.gamma_url}/markets/{market_id}"
        try:
            return self._make_request(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_market_trades(
        self, 
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[dict]:
        """Get trades for a market."""
        url = f"{self.clob_url}/trades"
        params = {"market": market_id}
        
        data = self._make_request(url, params)
        trades = data.get("trades", [])
        
        # Filter by date if specified
        if start_date or end_date:
            filtered_trades = []
            for trade in trades:
                trade_time = datetime.fromtimestamp(trade.get("timestamp", 0), tz=timezone.utc)
                if start_date and trade_time < start_date:
                    continue
                if end_date and trade_time > end_date:
                    continue
                filtered_trades.append(trade)
            trades = filtered_trades
        
        return trades
    
    def get_orderbook(self, market_id: str) -> Optional[dict]:
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
        self, 
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical price data for a market."""
        # Polymarket doesn't have a direct price history endpoint
        # We derive prices from trades
        trades = self.get_market_trades(market_id, start_date, end_date)
        
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df.get("size", 0), errors="coerce")
        
        return df
    
    def extract_to_raw(
        self,
        output_dir: Path,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        max_markets: int = 50,
        run_id: Optional[str] = None
    ) -> Dict[str, List[Path]]:
        """Extract Polymarket data to raw lake."""
        output_dir = Path(output_dir) / "polymarket"
        markets_dir = output_dir / "markets"
        prices_dir = output_dir / "prices"
        trades_dir = output_dir / "trades"
        
        for d in [markets_dir, prices_dir, trades_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        saved_files = {"markets": [], "prices": [], "trades": []}
        
        # Get active markets
        logger.info("Fetching Polymarket markets...")
        markets = []
        offset = 0
        while len(markets) < max_markets:
            batch = self.get_markets(limit=20, offset=offset, active=True)
            if not batch:
                break
            markets.extend(batch)
            offset += len(batch)
            if len(batch) < 20:
                break
        
        markets = markets[:max_markets]
        logger.info(f"Found {len(markets)} markets to extract")
        
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
                market_df["extracted_at"] = datetime.now(timezone.utc)
                market_df["run_id"] = run_id
                
                market_path = markets_dir / f"{market_id}.parquet"
                market_df.to_parquet(market_path, index=False)
                saved_files["markets"].append(market_path)
                
                # Get trades
                trades = self.get_market_trades(market_id, start_dt, end_dt)
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df["market_id"] = market_id
                    trades_df["extracted_at"] = datetime.now(timezone.utc)
                    trades_df["run_id"] = run_id
                    
                    trades_path = trades_dir / f"{market_id}_trades.parquet"
                    trades_df.to_parquet(trades_path, index=False)
                    saved_files["trades"].append(trades_path)
                    logger.info(f"Saved {len(trades_df)} trades for {market_id}")
                
                # Get price history
                prices_df = self.get_price_history(market_id, start_dt, end_dt)
                if not prices_df.empty:
                    prices_df["market_id"] = market_id
                    prices_df["extracted_at"] = datetime.now(timezone.utc)
                    prices_df["run_id"] = run_id
                    
                    prices_path = prices_dir / f"{market_id}_prices.parquet"
                    prices_df.to_parquet(prices_path, index=False)
                    saved_files["prices"].append(prices_path)
                    
            except Exception as e:
                logger.error(f"Error extracting market {market_id}: {e}")
                continue
        
        return saved_files


def extract_polymarket(
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_markets: int = 50,
    run_id: Optional[str] = None
) -> Dict[str, List[Path]]:
    """CLI-friendly wrapper for Polymarket extraction."""
    extractor = PolymarketExtractor()
    
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    
    return extractor.extract_to_raw(
        output_dir=output_dir,
        start_date=start,
        end_date=end,
        max_markets=max_markets,
        run_id=run_id
    )
