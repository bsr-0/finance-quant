"""FINRA short interest data extractor."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class ShortInterestExtractor:
    """Extract short interest data.

    FINRA publishes short interest data twice monthly (mid-month and end-of-month
    settlement dates). This extractor scrapes publicly available short interest
    summary data.
    """

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketDataWarehouse/1.0)",
            },
        )
        self._circuit = get_circuit_breaker(
            "short_interest", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("short_interest_extractor")

    def __del__(self) -> None:
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_short_interest(self, ticker: str) -> list[dict]:
        """Fetch short interest data for a ticker.

        Uses Yahoo Finance key statistics as a readily available source for
        current short interest. For historical data, FINRA's delayed files
        would be used.
        """

        def _do() -> list[dict]:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "defaultKeyStatistics"}
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return []

            stats = result[0].get("defaultKeyStatistics", {})
            short_interest = stats.get("sharesShort", {}).get("raw")
            short_date = stats.get("dateShortInterest", {}).get("raw")
            avg_volume = stats.get("averageDailyVolume10Day", {}).get("raw") or stats.get("averageVolume", {}).get("raw")
            float_shares = stats.get("floatShares", {}).get("raw")
            short_ratio = stats.get("shortRatio", {}).get("raw")
            short_pct = stats.get("shortPercentOfFloat", {}).get("raw")

            if short_interest is None:
                return []

            settlement = (
                datetime.fromtimestamp(short_date, tz=timezone.utc).date()
                if isinstance(short_date, (int, float))
                else date.today()
            )

            return [{
                "ticker": ticker,
                "settlement_date": settlement,
                "short_interest": int(short_interest),
                "avg_daily_volume": int(avg_volume) if avg_volume else None,
                "days_to_cover": float(short_ratio) if short_ratio else None,
                "short_pct_float": float(short_pct) if short_pct else None,
                "float_shares": int(float_shares) if float_shares else None,
            }]

        return self._circuit.call(_do)

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract short interest data for tickers."""
        settings = get_settings()
        tickers = tickers or settings.prices.universe
        output_dir = Path(output_dir) / "short_interest"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []
        today = date.today()

        for ticker in tickers:
            logger.info(f"Extracting short interest for {ticker}")
            try:
                with self._metrics.time_operation(f"extract_si_{ticker}"):
                    rows = self._fetch_short_interest(ticker)
                    all_rows.extend(rows)
            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed short interest for {ticker}: {e}")
            time.sleep(0.5)

        if not all_rows:
            return []

        df = pd.DataFrame(all_rows)
        df["extracted_at"] = datetime.now(timezone.utc)
        df["run_id"] = run_id

        file_path = output_dir / f"short_interest_{today}.parquet"
        df.to_parquet(file_path, index=False)
        self._metrics.record_extracted("short_interest", len(df))
        logger.info(f"Saved {len(df)} short interest records")

        return [file_path]


def extract_short_interest(
    output_dir: Path,
    tickers: list[str] | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = ShortInterestExtractor()
    return extractor.extract_to_raw(
        output_dir=output_dir, tickers=tickers, run_id=run_id,
    )
