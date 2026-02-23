"""Daily OHLCV price data extractor."""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# Number of consecutive missing trading days that signals delisting
_DELISTING_GAP_DAYS = 30


class YahooFinanceExtractor:
    """Extract price data from Yahoo Finance."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        self._circuit = get_circuit_breaker(
            "yahoo_finance", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("prices_extractor")

    def __del__(self):
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_ticker_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Get historical OHLCV data for a ticker."""

        def _do_request() -> dict:
            base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
            start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history,div,splits",
            }
            url = f"{base_url}{ticker}"
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        data = self._circuit.call(_do_request)
        result = data.get("chart", {}).get("result", [{}])[0]

        if not result or "timestamp" not in result:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()

        # Extract OHLCV data
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s").date,
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )

        # Add adjusted close if available
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        if adjclose:
            df["adj_close"] = adjclose
        else:
            df["adj_close"] = df["close"]

        df["ticker"] = ticker

        # Remove rows with null prices
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Extract corporate actions (splits & dividends) from the response
        events = result.get("events", {})
        splits = events.get("splits", {})
        dividends = events.get("dividends", {})

        if splits:
            split_dates = {
                pd.Timestamp(int(ts), unit="s").date(): v.get("splitRatio", "1:1")
                for ts, v in splits.items()
            }
            df["split_ratio"] = df["date"].map(split_dates)
        else:
            df["split_ratio"] = None

        if dividends:
            div_amounts = {
                pd.Timestamp(int(ts), unit="s").date(): v.get("amount", 0.0)
                for ts, v in dividends.items()
            }
            df["dividend"] = df["date"].map(div_amounts).fillna(0.0)
        else:
            df["dividend"] = 0.0

        return df

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: List[str],
        start_date: date,
        end_date: date,
        run_id: Optional[str] = None,
    ) -> List[Path]:
        """Extract price data to raw lake."""
        output_dir = Path(output_dir) / "prices"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for ticker in tickers:
            logger.info(f"Extracting prices for {ticker}")
            try:
                with self._metrics.time_operation(f"extract_{ticker}"):
                    df = self.get_ticker_data(ticker, start_date, end_date)
                if df.empty:
                    continue

                # Add metadata
                df["extracted_at"] = datetime.now(timezone.utc)
                df["run_id"] = run_id

                # Save to parquet
                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("prices", len(df))
                logger.info(f"Saved {len(df)} price records for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to extract {ticker}: {e}")
                continue

        return saved_files


def detect_delisted_symbols(prices_dir: Path) -> Dict[str, date]:
    """Detect potentially delisted symbols from extracted price data.

    Scans parquet files in *prices_dir* and flags any ticker whose most recent
    trading day is more than ``_DELISTING_GAP_DAYS`` before the latest trading
    day across all tickers.

    Returns:
        Mapping of ticker -> last observed trading date for suspected delistings.
    """
    parquet_files = list(prices_dir.glob("*.parquet"))
    if not parquet_files:
        return {}

    ticker_last_dates: Dict[str, date] = {}
    global_max_date: Optional[date] = None

    for path in parquet_files:
        try:
            df = pd.read_parquet(path, columns=["ticker", "date"])
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for ticker, group in df.groupby("ticker"):
                last = group["date"].max()
                if ticker not in ticker_last_dates or last > ticker_last_dates[ticker]:
                    ticker_last_dates[ticker] = last
                if global_max_date is None or last > global_max_date:
                    global_max_date = last
        except Exception as e:
            logger.warning(f"Could not read {path} for delisting check: {e}")

    if global_max_date is None:
        return {}

    cutoff = global_max_date - timedelta(days=_DELISTING_GAP_DAYS)
    return {
        ticker: last_date
        for ticker, last_date in ticker_last_dates.items()
        if last_date < cutoff
    }


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
        run_id: Optional[str] = None,
    ) -> List[Path]:
        """Extract price data to raw lake."""
        symbols = tickers or self.universe
        start = start_date or date.fromisoformat(get_settings().default_start_date)
        end = end_date or date.fromisoformat(get_settings().default_end_date)

        files = self._extractor.extract_to_raw(
            output_dir=output_dir,
            tickers=symbols,
            start_date=start,
            end_date=end,
            run_id=run_id,
        )

        # Post-extraction: check for potential delistings
        prices_dir = Path(output_dir) / "prices"
        delisted = detect_delisted_symbols(prices_dir)
        if delisted:
            logger.warning(
                f"Potential delistings detected ({len(delisted)} symbols): "
                f"{', '.join(delisted.keys())}"
            )

        return files


def extract_prices(
    output_dir: Path,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None,
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
        run_id=run_id,
    )
