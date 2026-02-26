"""Earnings calendar and estimates extractor (Yahoo Finance)."""

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


class EarningsExtractor:
    """Extract earnings calendar and surprise data from Yahoo Finance."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketDataWarehouse/1.0)",
            },
        )
        self._circuit = get_circuit_breaker(
            "yahoo_earnings", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("earnings_extractor")

    def __del__(self) -> None:
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_earnings_history(self, ticker: str) -> list[dict]:
        """Fetch earnings history for a ticker from Yahoo Finance."""

        def _do() -> list[dict]:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "earningsHistory,earnings"}
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return []

            module = result[0]
            history = module.get("earningsHistory", {}).get("history", [])
            return history

        return self._circuit.call(_do)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_earnings_calendar(self, ticker: str) -> list[dict]:
        """Fetch upcoming earnings dates."""

        def _do() -> list[dict]:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {"modules": "calendarEvents"}
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return []

            events = result[0].get("calendarEvents", {}).get("earnings", {})
            return [events] if events else []

        return self._circuit.call(_do)

    @staticmethod
    def _parse_earnings_history(
        history: list[dict], ticker: str
    ) -> list[dict]:
        """Parse earnings history into flat records."""
        rows: list[dict] = []
        for entry in history:
            quarter = entry.get("quarter", {})
            eps_est = entry.get("epsEstimate", {}).get("raw")
            eps_act = entry.get("epsActual", {}).get("raw")
            eps_diff = entry.get("epsDifference", {}).get("raw")
            surprise_pct = entry.get("surprisePercent", {}).get("raw")

            period_val = entry.get("period", {})
            if isinstance(period_val, dict):
                period = period_val.get("fmt")
            else:
                period = str(period_val) if period_val else None

            # Get the quarter end date
            quarter_raw = quarter.get("raw") if isinstance(quarter, dict) else quarter
            quarter_fmt = quarter.get("fmt") if isinstance(quarter, dict) else None

            report_date = None
            if quarter_fmt:
                report_date = quarter_fmt
            elif period:
                report_date = period

            if report_date is None:
                continue

            rows.append({
                "ticker": ticker,
                "report_date": report_date,
                "fiscal_quarter_end": period,
                "eps_estimate": eps_est,
                "eps_actual": eps_act,
                "eps_surprise": eps_diff,
                "eps_surprise_pct": surprise_pct * 100 if surprise_pct is not None else None,
            })

        return rows

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract earnings data for tickers."""
        settings = get_settings()
        tickers = tickers or settings.prices.universe
        output_dir = Path(output_dir) / "earnings"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[Path] = []

        for ticker in tickers:
            logger.info(f"Extracting earnings for {ticker}")
            try:
                with self._metrics.time_operation(f"extract_earnings_{ticker}"):
                    history = self._fetch_earnings_history(ticker)
                    rows = self._parse_earnings_history(history, ticker)

                if not rows:
                    logger.warning(f"No earnings data for {ticker}")
                    time.sleep(0.5)
                    continue

                df = pd.DataFrame(rows)
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date

                if start_date:
                    df = df[df["report_date"] >= start_date]
                if end_date:
                    df = df[df["report_date"] <= end_date]

                if df.empty:
                    time.sleep(0.5)
                    continue

                df["extracted_at"] = datetime.now(timezone.utc)
                df["run_id"] = run_id

                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("earnings", len(df))
                logger.info(f"Saved {len(df)} earnings records for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed earnings for {ticker}: {e}")
                continue

            time.sleep(0.5)

        return saved_files


def extract_earnings(
    output_dir: Path,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = EarningsExtractor()
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return extractor.extract_to_raw(
        output_dir=output_dir, tickers=tickers,
        start_date=start, end_date=end, run_id=run_id,
    )
