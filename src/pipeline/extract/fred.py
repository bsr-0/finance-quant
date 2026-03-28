"""FRED (Federal Reserve Economic Data) extractor."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class FredExtractor:
    """Extract economic data from FRED API."""

    def __init__(self, api_key: str | None = None):
        settings = get_settings().fred
        self.api_key = api_key or settings.api_key
        self.base_url = settings.base_url
        self.series_codes = settings.series_codes
        self.client = httpx.Client(timeout=30.0)
        self._circuit = get_circuit_breaker("fred", failure_threshold=5, recovery_timeout=60.0)
        self._metrics = PipelineMetrics("fred_extractor")

    def __del__(self):
        self.client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with retry logic and circuit breaker."""

        def _do_request() -> dict:
            url = f"{self.base_url}/{endpoint}"
            params["api_key"] = self.api_key
            params["file_type"] = "json"
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        return self._circuit.call(_do_request)

    def get_series_info(self, series_code: str) -> dict:
        """Get metadata for a series."""
        data = self._make_request("series", {"series_id": series_code})
        return data.get("seriess", [{}])[0]

    def get_observations(
        self, series_code: str, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        """Get observations for a series."""
        params = {"series_id": series_code}
        if start_date:
            params["observation_start"] = start_date.isoformat()
        if end_date:
            params["observation_end"] = end_date.isoformat()

        data = self._make_request("series/observations", params)
        observations = data.get("observations", [])

        if not observations:
            logger.warning(f"No observations found for series {series_code}")
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["series_code"] = series_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce")
        df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce")

        return df

    def extract_to_raw(
        self,
        output_dir: Path,
        series_codes: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract series data to raw lake."""
        codes = series_codes or self.series_codes
        output_dir = Path(output_dir) / "fred"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for code in codes:
            logger.info(f"Extracting FRED series: {code}")
            try:
                with self._metrics.time_operation(f"extract_{code}"):
                    df = self.get_observations(code, start_date, end_date)
                if df.empty:
                    continue

                # Add metadata
                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                # Save to parquet
                file_path = output_dir / f"{code}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("fred", len(df))
                logger.info(f"Saved {len(df)} observations to {file_path}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed to extract {code}: {e}")
                raise

        return saved_files


def extract_fred(
    output_dir: Path,
    series_codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper for FRED extraction."""

    extractor = FredExtractor()

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None

    return extractor.extract_to_raw(
        output_dir=output_dir,
        series_codes=series_codes,
        start_date=start,
        end_date=end,
        run_id=run_id,
    )
