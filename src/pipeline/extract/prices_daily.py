"""Daily OHLCV price data extractor with multi-source fallback.

Supports Yahoo Finance (default), Alpaca, and Polygon as price data sources.
When a fallback source is configured, it is tried automatically if the primary
source fails for a given ticker.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# Number of consecutive missing trading days that signals delisting
_DELISTING_GAP_DAYS = 30


# ---------------------------------------------------------------------------
# Corporate actions adjustment
# ---------------------------------------------------------------------------

def adjust_for_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust OHLCV prices for stock splits and dividends.

    Applies adjustments backwards from the most recent date so that all
    historical prices are comparable.  Modifies open/high/low/close/volume
    in-place and stores the original close as ``unadjusted_close``.

    Split ratios are expected in "new:old" format (e.g. "4:1" means a 4-for-1
    split).  Dividends reduce the adjustment factor by ``(close - div) / close``
    on ex-dates.
    """
    if df.empty:
        return df

    df = df.copy()
    df["unadjusted_close"] = df["close"].copy()

    # Build cumulative adjustment factor (working backwards from newest)
    adj_factor = np.ones(len(df), dtype=float)

    for i in range(len(df) - 2, -1, -1):
        factor = adj_factor[i + 1]

        # Split adjustment
        split_raw = df.iloc[i + 1].get("split_ratio")
        if split_raw is not None and pd.notna(split_raw):
            ratio = _parse_split_ratio(split_raw)
            if ratio != 1.0:
                factor /= ratio

        # Dividend adjustment
        div_amt = df.iloc[i + 1].get("dividend", 0.0)
        if pd.notna(div_amt) and div_amt > 0:
            close_val = df.iloc[i + 1]["unadjusted_close"]
            if close_val > 0:
                factor *= (close_val - div_amt) / close_val

        adj_factor[i] = factor

    # Apply adjustment
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col] * adj_factor

    # Adjust volume inversely
    if "volume" in df.columns:
        vol_adjusted = pd.to_numeric(df["volume"], errors="coerce") / adj_factor
        df["volume"] = vol_adjusted.round().astype("Int64")

    # Update adj_close to match
    df["adj_close"] = df["close"]

    return df


def _parse_split_ratio(raw: str | float) -> float:
    """Parse split ratio from 'new:old' string or numeric value."""
    if isinstance(raw, (int, float)):
        return float(raw) if raw != 0 else 1.0
    if isinstance(raw, str) and ":" in raw:
        parts = raw.split(":")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return 1.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 1.0


# ---------------------------------------------------------------------------
# Yahoo Finance extractor
# ---------------------------------------------------------------------------

class YahooFinanceExtractor:
    """Extract price data from Yahoo Finance."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
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
        tickers: list[str],
        start_date: date,
        end_date: date,
        run_id: str | None = None,
    ) -> list[Path]:
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


# ---------------------------------------------------------------------------
# Alpaca extractor
# ---------------------------------------------------------------------------

class AlpacaPriceExtractor:
    """Extract historical price data from Alpaca Markets data API."""

    BASE_URL = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        settings = get_settings().prices
        self._api_key = (
            api_key
            or settings.alpaca_api_key
            or os.environ.get("ALPACA_API_KEY", "")
        )
        self._secret_key = (
            secret_key
            or settings.alpaca_secret_key
            or os.environ.get("ALPACA_SECRET_KEY", "")
        )
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or configure in settings."
            )
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
            },
        )
        self._circuit = get_circuit_breaker(
            "alpaca_data", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("alpaca_prices_extractor")

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_ticker_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Get historical daily bars from Alpaca."""

        def _do_request() -> list[dict]:
            url = f"{self.BASE_URL}/stocks/{ticker}/bars"
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "timeframe": "1Day",
                "limit": 10000,
                "adjustment": "all",  # split + dividend adjusted
            }
            all_bars: list[dict] = []
            page_token = None

            while True:
                if page_token:
                    params["page_token"] = page_token
                response = self.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                bars = data.get("bars", [])
                if bars:
                    all_bars.extend(bars)
                page_token = data.get("next_page_token")
                if not page_token:
                    break

            return all_bars

        bars = self._circuit.call(_do_request)

        if not bars:
            logger.warning(f"No Alpaca data for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df = df.rename(columns={
            "t": "date", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["adj_close"] = df["close"]
        df["ticker"] = ticker
        df["split_ratio"] = None
        df["dividend"] = 0.0

        # Drop extra Alpaca columns
        keep_cols = ["date", "open", "high", "low", "close", "volume",
                     "adj_close", "ticker", "split_ratio", "dividend"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.dropna(subset=["open", "high", "low", "close"])

        return df

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str],
        start_date: date,
        end_date: date,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract price data to raw lake."""
        output_dir = Path(output_dir) / "prices"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for ticker in tickers:
            logger.info(f"Extracting prices for {ticker} from Alpaca")
            try:
                with self._metrics.time_operation(f"extract_{ticker}"):
                    df = self.get_ticker_data(ticker, start_date, end_date)
                if df.empty:
                    continue

                df["extracted_at"] = datetime.now(timezone.utc)
                df["run_id"] = run_id

                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("prices", len(df))
                logger.info(f"Saved {len(df)} Alpaca price records for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to extract {ticker} from Alpaca: {e}")
                continue

        return saved_files


# ---------------------------------------------------------------------------
# Polygon extractor
# ---------------------------------------------------------------------------

class PolygonPriceExtractor:
    """Extract historical price data from Polygon.io."""

    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, api_key: str | None = None):
        settings = get_settings().prices
        self._api_key = (
            api_key
            or settings.polygon_api_key
            or os.environ.get("POLYGON_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY environment "
                "variable or configure polygon_api_key in settings."
            )
        self.client = httpx.Client(timeout=30.0)
        self._circuit = get_circuit_breaker(
            "polygon_data", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("polygon_prices_extractor")

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_ticker_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Get historical daily bars from Polygon."""

        def _do_request() -> list[dict]:
            url = (
                f"{self.BASE_URL}/aggs/ticker/{ticker}/range/1/day/"
                f"{start_date.isoformat()}/{end_date.isoformat()}"
            )
            params = {
                "apiKey": self._api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
            }
            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])

        results = self._circuit.call(_do_request)

        if not results:
            logger.warning(f"No Polygon data for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "t": "date_ts", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        })
        df["date"] = pd.to_datetime(df["date_ts"], unit="ms").dt.date
        df["adj_close"] = df["close"]
        df["ticker"] = ticker
        df["split_ratio"] = None
        df["dividend"] = 0.0

        keep_cols = ["date", "open", "high", "low", "close", "volume",
                     "adj_close", "ticker", "split_ratio", "dividend"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.dropna(subset=["open", "high", "low", "close"])

        return df

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str],
        start_date: date,
        end_date: date,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract price data to raw lake."""
        output_dir = Path(output_dir) / "prices"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for ticker in tickers:
            logger.info(f"Extracting prices for {ticker} from Polygon")
            try:
                with self._metrics.time_operation(f"extract_{ticker}"):
                    df = self.get_ticker_data(ticker, start_date, end_date)
                if df.empty:
                    continue

                df["extracted_at"] = datetime.now(timezone.utc)
                df["run_id"] = run_id

                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("prices", len(df))
                logger.info(f"Saved {len(df)} Polygon price records for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to extract {ticker} from Polygon: {e}")
                continue

        return saved_files


# ---------------------------------------------------------------------------
# Delisting detection
# ---------------------------------------------------------------------------

def detect_delisted_symbols(prices_dir: Path) -> dict[str, date]:
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

    ticker_last_dates: dict[str, date] = {}
    global_max_date: date | None = None

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
        ticker: last_date for ticker, last_date in ticker_last_dates.items() if last_date < cutoff
    }


# ---------------------------------------------------------------------------
# Extractor factory
# ---------------------------------------------------------------------------

_EXTRACTOR_CLASSES: dict[str, type] = {
    "yahoo": YahooFinanceExtractor,
    "alpaca": AlpacaPriceExtractor,
    "polygon": PolygonPriceExtractor,
}


def _create_extractor(source: str):
    """Create an extractor instance for the given source name."""
    cls = _EXTRACTOR_CLASSES.get(source)
    if cls is None:
        raise ValueError(
            f"Unsupported price source: {source!r}. "
            f"Supported: {', '.join(_EXTRACTOR_CLASSES)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Main PriceExtractor with fallback
# ---------------------------------------------------------------------------

class PriceExtractor:
    """Generic price extractor with automatic fallback and corporate-action adjustment."""

    def __init__(self):
        settings = get_settings().prices
        self.source = settings.source
        self.fallback_source = settings.fallback_source
        self.universe = settings.universe
        self._adjust = settings.adjust_corporate_actions

        self._extractor = _create_extractor(self.source)
        self._fallback_extractor = None
        if self.fallback_source:
            try:
                self._fallback_extractor = _create_extractor(self.fallback_source)
                logger.info(
                    f"Price fallback configured: {self.source} -> {self.fallback_source}"
                )
            except Exception as e:
                logger.warning(f"Could not initialise fallback source {self.fallback_source}: {e}")

    def _extract_single_ticker(
        self,
        ticker: str,
        output_dir: Path,
        start_date: date,
        end_date: date,
        run_id: str | None,
    ) -> Path | None:
        """Extract a single ticker, falling back if primary fails."""
        prices_dir = Path(output_dir) / "prices"
        prices_dir.mkdir(parents=True, exist_ok=True)

        for label, extractor in self._sources():
            try:
                df = extractor.get_ticker_data(ticker, start_date, end_date)
                if df.empty:
                    logger.warning(f"No data from {label} for {ticker}, trying next source")
                    continue

                # Apply corporate-action adjustments if enabled and source is Yahoo
                # (Alpaca/Polygon return pre-adjusted data when requested)
                if self._adjust and label == "yahoo":
                    df = adjust_for_corporate_actions(df)

                df["extracted_at"] = datetime.now(timezone.utc)
                df["run_id"] = run_id
                df["data_source"] = label

                file_path = prices_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                logger.info(f"Saved {len(df)} price records for {ticker} from {label}")
                return file_path

            except Exception as e:
                logger.warning(f"{label} failed for {ticker}: {e}")
                continue

        logger.error(f"All sources failed for {ticker}")
        return None

    def _sources(self):
        """Yield (label, extractor) pairs: primary first, then fallback."""
        yield self.source, self._extractor
        if self._fallback_extractor is not None:
            yield self.fallback_source, self._fallback_extractor

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract price data to raw lake with fallback and adjustment."""
        symbols = tickers or self.universe
        start = start_date or date.fromisoformat(get_settings().default_start_date)
        end = end_date or date.fromisoformat(get_settings().default_end_date)

        saved_files: list[Path] = []
        for ticker in symbols:
            logger.info(f"Extracting prices for {ticker}")
            path = self._extract_single_ticker(ticker, output_dir, start, end, run_id)
            if path is not None:
                saved_files.append(path)

        # Post-extraction: check for potential delistings
        prices_dir = Path(output_dir) / "prices"
        delisted = detect_delisted_symbols(prices_dir)
        if delisted:
            logger.warning(
                f"Potential delistings detected ({len(delisted)} symbols): "
                f"{', '.join(delisted.keys())}"
            )

        return saved_files


def extract_prices(
    output_dir: Path,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
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
