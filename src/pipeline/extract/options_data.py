"""Options chain / implied volatility extractor (Yahoo Finance)."""

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
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class OptionsDataExtractor:
    """Extract options chain data from Yahoo Finance."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketDataWarehouse/1.0)",
            },
        )
        self._circuit = get_circuit_breaker(
            "yahoo_options", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("options_extractor")

    def __del__(self) -> None:
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_options_chain(self, ticker: str, expiration_ts: int | None = None) -> dict:
        """Fetch options chain for a ticker from Yahoo Finance."""

        def _do() -> dict:
            url = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
            params = {}
            if expiration_ts is not None:
                params["date"] = str(expiration_ts)
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return self._circuit.call(_do)

    def _get_expiration_dates(self, ticker: str) -> list[int]:
        """Get available expiration dates (as Unix timestamps)."""
        data = self._fetch_options_chain(ticker)
        result = data.get("optionChain", {}).get("result", [])
        if not result:
            return []
        return result[0].get("expirationDates", [])

    def _parse_chain(
        self, data: dict, ticker: str, quote_date: date
    ) -> list[dict]:
        """Parse options chain response into flat records."""
        result = data.get("optionChain", {}).get("result", [])
        if not result:
            return []

        options = result[0].get("options", [])
        rows: list[dict] = []

        for chain in options:
            expiration_ts = chain.get("expirationDate", 0)
            expiration = datetime.fromtimestamp(expiration_ts, tz=UTC).date()

            for opt_type, key in [("call", "calls"), ("put", "puts")]:
                for contract in chain.get(key, []):
                    iv = contract.get("impliedVolatility")
                    rows.append({
                        "ticker": ticker,
                        "quote_date": quote_date,
                        "expiration": expiration,
                        "strike": contract.get("strike"),
                        "option_type": opt_type,
                        "last_price": contract.get("lastPrice"),
                        "bid": contract.get("bid"),
                        "ask": contract.get("ask"),
                        "volume": contract.get("volume", 0) or 0,
                        "open_interest": contract.get("openInterest", 0) or 0,
                        "implied_volatility": iv,
                        "in_the_money": contract.get("inTheMoney", False),
                    })

        return rows

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        run_id: str | None = None,
        max_expirations: int = 6,
    ) -> list[Path]:
        """Extract options chain data for tickers."""
        settings = get_settings()
        tickers = tickers or settings.prices.universe
        output_dir = Path(output_dir) / "options"
        output_dir.mkdir(parents=True, exist_ok=True)

        today = date.today()
        saved_files: list[Path] = []

        for ticker in tickers:
            logger.info(f"Extracting options for {ticker}")
            try:
                # Get available expirations
                expirations = self._get_expiration_dates(ticker)
                # Take nearest N expirations for term structure
                expirations = expirations[:max_expirations]
                time.sleep(0.5)

                all_rows: list[dict] = []

                for exp_ts in expirations:
                    with self._metrics.time_operation(f"extract_options_{ticker}"):
                        data = self._fetch_options_chain(ticker, exp_ts)
                        rows = self._parse_chain(data, ticker, today)
                        all_rows.extend(rows)
                    time.sleep(0.5)

                if not all_rows:
                    continue

                df = pd.DataFrame(all_rows)
                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                file_path = output_dir / f"{ticker}_{today}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("options", len(df))
                logger.info(f"Saved {len(df)} options contracts for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed options for {ticker}: {e}")
                continue

        return saved_files


def extract_options(
    output_dir: Path,
    tickers: list[str] | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = OptionsDataExtractor()
    return extractor.extract_to_raw(
        output_dir=output_dir, tickers=tickers, run_id=run_id,
    )
