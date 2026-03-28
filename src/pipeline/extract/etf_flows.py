"""ETF fund flows extractor."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics

logger = logging.getLogger(__name__)

# ETF tickers to track for fund flow signals (broad market + sector)
DEFAULT_ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "VTI",
    "VOO",  # Broad market
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",  # Sector
    "XLU",
    "XLP",
    "XLY",
    "XLB",
    "XLRE",  # Sector continued
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",  # Fixed income
    "GLD",
    "SLV",
    "USO",  # Commodities
    "EEM",
    "EFA",
    "VWO",  # International
]


class EtfFlowsExtractor:
    """Extract ETF fund flow estimates.

    Fund flows are estimated from changes in shares outstanding multiplied
    by NAV/price, using Yahoo Finance data. This provides a daily proxy
    for capital flows without requiring a paid data vendor.
    """

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketDataWarehouse/1.0)",
            },
        )
        self._circuit = get_circuit_breaker("etf_flows", failure_threshold=5, recovery_timeout=60.0)
        self._metrics = PipelineMetrics("etf_flows_extractor")

    def __del__(self) -> None:
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_etf_profile(self, ticker: str) -> dict | None:
        """Fetch ETF profile data including AUM and shares outstanding."""

        def _do() -> dict | None:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "defaultKeyStatistics,summaryDetail,price"}
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return None

            stats = result[0].get("defaultKeyStatistics", {})
            price_data = result[0].get("price", {})
            summary = result[0].get("summaryDetail", {})

            total_assets = stats.get("totalAssets", {}).get("raw")
            shares_out = stats.get("sharesOutstanding", {}).get("raw") or price_data.get(
                "sharesOutstanding", {}
            ).get("raw")
            market_price = price_data.get("regularMarketPrice", {}).get("raw")
            volume = summary.get("volume", {}).get("raw") or price_data.get(
                "regularMarketVolume", {}
            ).get("raw")

            return {
                "ticker": ticker,
                "date": date.today(),
                "aum": total_assets,
                "shares_outstanding": int(shares_out) if shares_out else None,
                "market_price": market_price,
                "volume": int(volume) if volume else None,
            }

        return self._circuit.call(_do)

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract ETF data for fund flow estimation."""
        tickers = tickers or DEFAULT_ETF_UNIVERSE
        output_dir = Path(output_dir) / "etf_flows"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []
        today = date.today()

        for ticker in tickers:
            logger.info(f"Extracting ETF data for {ticker}")
            try:
                with self._metrics.time_operation(f"extract_etf_{ticker}"):
                    profile = self._fetch_etf_profile(ticker)
                    if profile:
                        all_rows.append(profile)
            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed ETF data for {ticker}: {e}")
            time.sleep(0.5)

        if not all_rows:
            return []

        df = pd.DataFrame(all_rows)
        df["extracted_at"] = datetime.now(UTC)
        df["run_id"] = run_id

        file_path = output_dir / f"etf_flows_{today}.parquet"
        df.to_parquet(file_path, index=False)
        self._metrics.record_extracted("etf_flows", len(df))
        logger.info(f"Saved {len(df)} ETF flow records")

        return [file_path]


def extract_etf_flows(
    output_dir: Path,
    tickers: list[str] | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = EtfFlowsExtractor()
    return extractor.extract_to_raw(
        output_dir=output_dir,
        tickers=tickers,
        run_id=run_id,
    )
