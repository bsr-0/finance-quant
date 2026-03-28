"""CFTC Commitments of Traders (COT) data extractor."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# CFTC bulk CSV column name mapping (Futures + Options Combined report)
_COL_MAP = {
    "CFTC_Contract_Market_Code": "commodity_code",
    "Market_and_Exchange_Names": "commodity_name",
    "Report_Date_as_YYYY-MM-DD": "report_date",
    "Comm_Positions_Long_All": "commercial_long",
    "Comm_Positions_Short_All": "commercial_short",
    "NonComm_Positions_Long_All": "noncommercial_long",
    "NonComm_Positions_Short_All": "noncommercial_short",
    "NonComm_Positions_Spread_All": "noncommercial_spreading",
    "NonRept_Positions_Long_All": "nonreportable_long",
    "NonRept_Positions_Short_All": "nonreportable_short",
    "Open_Interest_All": "open_interest",
}


class CftcCotExtractor:
    """Extract CFTC Commitments of Traders data.

    The CFTC publishes weekly COT reports every Friday at 3:30 PM ET
    reflecting positions as of the preceding Tuesday.  Historical data
    is available as annual bulk CSV files (zipped) from the CFTC website.
    """

    def __init__(self) -> None:
        settings = get_settings().cftc_cot
        self.base_url = settings.base_url
        self.commodity_codes = set(settings.commodity_codes.keys())
        self.client = httpx.Client(
            timeout=60.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketDataWarehouse/1.0)",
            },
        )
        self._circuit = get_circuit_breaker("cftc_cot", failure_threshold=5, recovery_timeout=60.0)
        self._metrics = PipelineMetrics("cftc_cot_extractor")

    def __del__(self) -> None:
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_cot_year(self, year: int) -> pd.DataFrame:
        """Fetch COT bulk CSV for a given year.

        The CFTC publishes combined Futures+Options reports as zipped CSVs.
        Current year uses ``deacot_combo.zip``; historical years use
        ``HistoricalCompressed/deacot<YYYY>.zip`` with the ``_combo`` variant
        for futures+options combined.
        """

        def _do() -> pd.DataFrame:
            current_year = date.today().year
            if year == current_year:
                url = f"{self.base_url}/deacot_combo.zip"
            else:
                url = f"{self.base_url}/HistoricalCompressed/deacot{year}_combo.zip"

            logger.info(f"Fetching COT data for {year} from {url}")
            resp = self.client.get(url)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".txt") or n.endswith(".csv")]
                if not csv_names:
                    logger.warning(f"No CSV/TXT found in COT zip for {year}")
                    return pd.DataFrame()
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, low_memory=False)

            # Trim whitespace from column names (CFTC CSVs have trailing spaces)
            df.columns = df.columns.str.strip()

            available_cols = {c: _COL_MAP[c] for c in _COL_MAP if c in df.columns}
            if not available_cols:
                logger.warning(f"No expected columns found in COT data for {year}")
                return pd.DataFrame()

            df = df[list(available_cols.keys())].rename(columns=available_cols)
            df["commodity_code"] = df["commodity_code"].astype(str).str.strip()
            df = df[df["commodity_code"].isin(self.commodity_codes)]
            df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

            for col in [
                "commercial_long",
                "commercial_short",
                "noncommercial_long",
                "noncommercial_short",
                "noncommercial_spreading",
                "nonreportable_long",
                "nonreportable_short",
                "open_interest",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        return self._circuit.call(_do)

    def extract_to_raw(
        self,
        output_dir: Path,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract COT data to raw parquet files."""
        settings = get_settings()
        if start_date is None:
            start_date = date.fromisoformat(settings.default_start_date)
        if end_date is None:
            end_date = date.today()

        output_dir = Path(output_dir) / "cftc_cot"
        output_dir.mkdir(parents=True, exist_ok=True)

        years = range(start_date.year, end_date.year + 1)
        all_frames: list[pd.DataFrame] = []

        for year in years:
            try:
                with self._metrics.time_operation(f"extract_cot_{year}"):
                    df = self._fetch_cot_year(year)
                    if not df.empty:
                        all_frames.append(df)
            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to fetch COT data for {year}: {e}")

        if not all_frames:
            return []

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined[
            (combined["report_date"] >= start_date) & (combined["report_date"] <= end_date)
        ]

        if combined.empty:
            return []

        combined["extracted_at"] = datetime.now(UTC)
        combined["run_id"] = run_id

        file_path = output_dir / f"cftc_cot_{start_date}_{end_date}.parquet"
        combined.to_parquet(file_path, index=False)
        self._metrics.record_extracted("cftc_cot", len(combined))
        logger.info(f"Saved {len(combined)} COT records to {file_path}")

        return [file_path]


def extract_cftc_cot(
    output_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = CftcCotExtractor()
    return extractor.extract_to_raw(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
    )
