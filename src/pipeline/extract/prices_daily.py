"""Daily OHLCV price data extractor."""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class YahooFinanceExtractor:
    """Extract price data from Yahoo Finance."""
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    def __del__(self):
        self.client.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_ticker_data(
        self, 
        ticker: str, 
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get historical OHLCV data for a ticker."""
        # Yahoo Finance API endpoint
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        
        # Convert dates to timestamps
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history,div,splits"
        }
        
        url = f"{base_url}{ticker}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [{}])[0]
        
        if not result or "timestamp" not in result:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Extract OHLCV data
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps, unit="s").date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume")
        })
        
        # Add adjusted close if available
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        if adjclose:
            df["adj_close"] = adjclose
        else:
            df["adj_close"] = df["close"]
        
        df["ticker"] = ticker
        
        # Remove rows with null prices
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        return df
    
    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: List[str],
        start_date: date,
        end_date: date,
        run_id: Optional[str] = None
    ) -> List[Path]:
        """Extract price data to raw lake."""
        output_dir = Path(output_dir) / "prices"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for ticker in tickers:
            logger.info(f"Extracting prices for {ticker}")
            try:
                df = self.get_ticker_data(ticker, start_date, end_date)
                if df.empty:
                    continue
                
                # Add metadata
                df["extracted_at"] = datetime.utcnow()
                df["run_id"] = run_id
                
                # Save to parquet
                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                logger.info(f"Saved {len(df)} price records for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to extract {ticker}: {e}")
                continue
        
        return saved_files


class PriceExtractor:
    """Generic price extractor that delegates to appropriate source."""
    
    def __init__(self):
        settings = get_settings().prices
        self.source = settings.source
        self.universe = settings.universe
        
        if self.source == "yahoo":
            self._extractor = YahooFinanceExtractor()
        else:
            raise ValueError(f"Unsupported price source: {self.source}")
    
    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_id: Optional[str] = None
    ) -> List[Path]:
        """Extract price data to raw lake."""
        symbols = tickers or self.universe
        start = start_date or date.fromisoformat(get_settings().default_start_date)
        end = end_date or date.fromisoformat(get_settings().default_end_date)
        
        return self._extractor.extract_to_raw(
            output_dir=output_dir,
            tickers=symbols,
            start_date=start,
            end_date=end,
            run_id=run_id
        )


def extract_prices(
    output_dir: Path,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None
) -> List[Path]:
    """CLI-friendly wrapper for price extraction."""
    extractor = PriceExtractor()
    
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    
    return extractor.extract_to_raw(
        output_dir=output_dir,
        tickers=tickers,
        start_date=start,
        end_date=end,
        run_id=run_id
    )
