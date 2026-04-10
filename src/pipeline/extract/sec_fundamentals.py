"""SEC EDGAR fundamentals extractor (XBRL Company Facts API)."""

from __future__ import annotations

import logging
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

# Core US GAAP metrics to extract from XBRL company facts.
DEFAULT_METRICS: list[str] = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "OperatingIncomeLoss",
    "GrossProfit",
    "NetCashProvidedByUsedInOperatingActivities",
    "CommonStockSharesOutstanding",
]

# SEC requires a descriptive User-Agent header.
SEC_USER_AGENT = "MarketDataWarehouse/1.0 (research; contact@example.com)"


class SecFundamentalsExtractor(HttpClientMixin):
    """Extract quarterly/annual financial data from SEC EDGAR XBRL API."""

    def __init__(self) -> None:
        get_settings()
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": SEC_USER_AGENT},
            follow_redirects=True,
        )
        self._circuit = get_circuit_breaker("sec_edgar", failure_threshold=5, recovery_timeout=60.0)
        self._metrics = PipelineMetrics("sec_fundamentals_extractor")
        self._rate_limit_delay = 0.12  # SEC asks for <=10 req/s

    # ------------------------------------------------------------------
    # Ticker → CIK mapping
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_ticker_to_cik(self) -> dict[str, int]:
        """Fetch the SEC ticker→CIK mapping."""

        def _do() -> dict[str, int]:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            mapping: dict[str, int] = {}
            for entry in data.values():
                ticker = str(entry.get("ticker", "")).upper()
                cik = int(entry.get("cik_str", 0))
                if ticker and cik:
                    mapping[ticker] = cik
            return mapping

        return self._circuit.call(_do)

    # ------------------------------------------------------------------
    # Company facts
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_company_facts(self, cik: int) -> dict | None:
        """Fetch XBRL company facts for a CIK.

        Returns None for entities without XBRL data (e.g. ETFs).
        """

        def _do() -> dict | None:
            padded = str(cik).zfill(10)
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded}.json"
            resp = self.client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

        return self._circuit.call(_do)

    # ------------------------------------------------------------------
    # Parse facts into flat rows
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_facts(
        facts_json: dict,
        ticker: str,
        cik: int,
        metrics: list[str] | None = None,
    ) -> list[dict]:
        """Parse XBRL company facts JSON into flat records.

        Each record represents one (metric, fiscal_period_end) observation.
        We keep only 10-Q and 10-K filings and use the ``filed`` date as
        ``available_time`` (when the data became publicly available on EDGAR).
        """
        target_metrics = set(metrics or DEFAULT_METRICS)
        us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
        rows: list[dict] = []

        for concept_name, concept_data in us_gaap.items():
            if concept_name not in target_metrics:
                continue

            label = concept_data.get("label", concept_name)
            units_dict = concept_data.get("units", {})

            for unit_key, entries in units_dict.items():
                for entry in entries:
                    form = entry.get("form", "")
                    # Detect amendments (e.g. 10-Q/A, 10-K/A)
                    is_amendment = form.endswith("/A")
                    base_form = form.replace("/A", "")
                    if base_form not in ("10-Q", "10-K"):
                        continue

                    filed = entry.get("filed")
                    end = entry.get("end")
                    val = entry.get("val")
                    if filed is None or end is None or val is None:
                        continue

                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "metric_name": concept_name,
                            "metric_label": label,
                            "metric_value": float(val),
                            "units": unit_key,
                            "fiscal_period_end": end,
                            "filing_date": filed,
                            "form_type": base_form,
                            "original_form_type": form,
                            "is_amendment": is_amendment,
                            "accession_number": entry.get("accn"),
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                        }
                    )

        return rows

    @staticmethod
    def _assign_filing_sequence(rows: list[dict]) -> list[dict]:
        """Assign chronological filing_sequence per (ticker, metric, period).

        Sequence 1 is the original filing; higher numbers are amendments
        filed later.  This enables point-in-time queries that return only
        the data available at a specific date.
        """
        from collections import defaultdict

        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            key = (row["ticker"], row["metric_name"], row["fiscal_period_end"])
            groups[key].append(i)

        for _key, indices in groups.items():
            # Sort by filing_date, then by is_amendment (originals first)
            indices.sort(key=lambda i: (rows[i]["filing_date"], rows[i]["is_amendment"]))
            for seq, idx in enumerate(indices, start=1):
                rows[idx]["filing_sequence"] = seq

        return rows

    # ------------------------------------------------------------------
    # Public extraction entry-point
    # ------------------------------------------------------------------

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
        metrics: list[str] | None = None,
    ) -> list[Path]:
        """Extract SEC fundamentals for a list of tickers to parquet files."""
        settings = get_settings()
        tickers = tickers or settings.prices.universe
        output_dir = Path(output_dir) / "sec_fundamentals"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch CIK mapping
        logger.info("Fetching SEC ticker→CIK mapping")
        ticker_to_cik = self._fetch_ticker_to_cik()

        saved_files: list[Path] = []
        import time

        for ticker in tickers:
            cik = ticker_to_cik.get(ticker.upper())
            if cik is None:
                logger.warning(f"No CIK found for {ticker}, skipping")
                continue

            logger.info(f"Extracting SEC fundamentals for {ticker} (CIK {cik})")
            try:
                with self._metrics.time_operation(f"extract_{ticker}"):
                    facts = self._fetch_company_facts(cik)
                    if facts is None:
                        logger.warning(
                            "No XBRL company facts for %s (CIK %s), likely an ETF/fund",
                            ticker, cik,
                        )
                        continue
                    rows = self._parse_facts(facts, ticker, cik, metrics)
                    rows = self._assign_filing_sequence(rows)

                if not rows:
                    logger.warning(f"No fundamentals data for {ticker}")
                    continue

                df = pd.DataFrame(rows)

                # Optionally filter by date range
                df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"]).dt.date
                df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
                if start_date:
                    df = df[df["fiscal_period_end"] >= start_date]
                if end_date:
                    df = df[df["fiscal_period_end"] <= end_date]

                if df.empty:
                    continue

                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("sec_fundamentals", len(df))
                logger.info(f"Saved {len(df)} fundamentals records for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to extract fundamentals for {ticker}: {e}")
                continue

            # Rate-limit to stay under SEC's 10 req/s guideline
            time.sleep(self._rate_limit_delay)

        return saved_files


def point_in_time_fundamentals(
    df: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """Return fundamentals as known at a specific date.

    For each ``(ticker, metric_name, fiscal_period_end)`` group, returns the
    latest filing whose ``filing_date <= as_of``.  This ensures backtests
    never use restated data that was not yet publicly available.

    Args:
        df: Raw fundamentals DataFrame with ``filing_date`` column (as
            ``datetime.date`` or convertible).
        as_of: The point-in-time cutoff date.

    Returns:
        Filtered DataFrame with one row per (ticker, metric, period).
    """
    if df.empty:
        return df
    if df["filing_date"].dtype != "object":
        filing_col = pd.to_datetime(df["filing_date"]).dt.date
    else:
        filing_col = df["filing_date"]
    available = df[filing_col <= as_of]
    if available.empty:
        return available
    idx = available.groupby(["ticker", "metric_name", "fiscal_period_end"])[
        "filing_date"
    ].transform("max")
    return available[available["filing_date"] == idx].drop_duplicates(
        subset=["ticker", "metric_name", "fiscal_period_end"],
        keep="last",
    )


def extract_sec_fundamentals(
    output_dir: Path,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper for SEC fundamentals extraction."""
    extractor = SecFundamentalsExtractor()
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return extractor.extract_to_raw(
        output_dir=output_dir,
        tickers=tickers,
        start_date=start,
        end_date=end,
        run_id=run_id,
    )
