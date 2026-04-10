"""FINRA short interest data extractor."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.extract._base import HttpClientMixin
from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class ShortInterestExtractor(HttpClientMixin):
    """Extract short interest data.

    FINRA publishes short interest data twice monthly (mid-month and end-of-month
    settlement dates). This extractor scrapes publicly available short interest
    summary data.
    """

    _BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": self._BROWSER_UA},
        )
        self._circuit = get_circuit_breaker(
            "short_interest", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("short_interest_extractor")
        self._crumb: str | None = None

    def _ensure_crumb(self) -> str:
        """Obtain a Yahoo Finance crumb+cookie pair for authenticated requests.

        Yahoo's v10 quoteSummary API requires a valid crumb parameter and
        matching session cookies.  We fetch these by hitting the Yahoo Finance
        landing page first (which sets cookies), then calling the crumb
        endpoint.
        """
        if self._crumb is not None:
            return self._crumb

        # Step 1: Visit Yahoo Finance to obtain session cookies
        self.client.get("https://finance.yahoo.com/quote/SPY", headers={"User-Agent": self._BROWSER_UA})

        # Step 2: Fetch the crumb using the session cookies
        crumb_resp = self.client.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            headers={"User-Agent": self._BROWSER_UA},
        )
        crumb_resp.raise_for_status()
        self._crumb = crumb_resp.text.strip()
        logger.info("Obtained Yahoo Finance crumb for authenticated requests")
        return self._crumb

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_short_interest(self, ticker: str) -> list[dict]:
        """Fetch short interest data for a ticker.

        Uses Yahoo Finance key statistics as a readily available source for
        current short interest. For historical data, FINRA's delayed files
        would be used.
        """

        def _do() -> list[dict]:
            crumb = self._ensure_crumb()
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "defaultKeyStatistics", "crumb": crumb}
            resp = self.client.get(url, params=params)
            if resp.status_code == 401:
                # Crumb expired — clear and re-fetch on retry
                self._crumb = None
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()

            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return []

            stats = result[0].get("defaultKeyStatistics", {})
            short_interest = stats.get("sharesShort", {}).get("raw")
            short_date = stats.get("dateShortInterest", {}).get("raw")
            avg_volume = stats.get("averageDailyVolume10Day", {}).get("raw") or stats.get(
                "averageVolume", {}
            ).get("raw")
            float_shares = stats.get("floatShares", {}).get("raw")
            short_ratio = stats.get("shortRatio", {}).get("raw")
            short_pct = stats.get("shortPercentOfFloat", {}).get("raw")

            if short_interest is None:
                return []

            settlement = (
                datetime.fromtimestamp(short_date, tz=UTC).date()
                if isinstance(short_date, (int, float))
                else date.today()
            )

            return [
                {
                    "ticker": ticker,
                    "settlement_date": settlement,
                    "short_interest": int(short_interest),
                    "avg_daily_volume": int(avg_volume) if avg_volume else None,
                    "days_to_cover": float(short_ratio) if short_ratio else None,
                    "short_pct_float": float(short_pct) if short_pct else None,
                    "float_shares": int(float_shares) if float_shares else None,
                }
            ]

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
        df["extracted_at"] = datetime.now(UTC)
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
        output_dir=output_dir,
        tickers=tickers,
        run_id=run_id,
    )
