"""GDELT (Global Database of Events, Language, and Tone) extractor."""

from __future__ import annotations

import logging
import zipfile
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.extract._base import HttpClientMixin

logger = logging.getLogger(__name__)


class GDELTExtractor(HttpClientMixin):
    """Extract event data from GDELT."""

    # GDELT column names for the events table
    EVENT_COLUMNS = [
        "GLOBALEVENTID",
        "SQLDATE",
        "MonthYear",
        "Year",
        "FractionDate",
        "Actor1Code",
        "Actor1Name",
        "Actor1CountryCode",
        "Actor1KnownGroupCode",
        "Actor1EthnicCode",
        "Actor1Religion1Code",
        "Actor1Religion2Code",
        "Actor1Type1Code",
        "Actor1Type2Code",
        "Actor1Type3Code",
        "Actor2Code",
        "Actor2Name",
        "Actor2CountryCode",
        "Actor2KnownGroupCode",
        "Actor2EthnicCode",
        "Actor2Religion1Code",
        "Actor2Religion2Code",
        "Actor2Type1Code",
        "Actor2Type2Code",
        "Actor2Type3Code",
        "IsRootEvent",
        "EventCode",
        "EventBaseCode",
        "EventRootCode",
        "QuadClass",
        "GoldsteinScale",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",
        "Actor1Geo_Type",
        "Actor1Geo_FullName",
        "Actor1Geo_CountryCode",
        "Actor1Geo_ADM1Code",
        "Actor1Geo_Lat",
        "Actor1Geo_Long",
        "Actor1Geo_FeatureID",
        "Actor2Geo_Type",
        "Actor2Geo_FullName",
        "Actor2Geo_CountryCode",
        "Actor2Geo_ADM1Code",
        "Actor2Geo_Lat",
        "Actor2Geo_Long",
        "Actor2Geo_FeatureID",
        "ActionGeo_Type",
        "ActionGeo_FullName",
        "ActionGeo_CountryCode",
        "ActionGeo_ADM1Code",
        "ActionGeo_Lat",
        "ActionGeo_Long",
        "ActionGeo_FeatureID",
        "DATEADDED",
        "SOURCEURL",
    ]

    def __init__(self):
        self.client = httpx.Client(timeout=60.0)
        self.base_url = "http://data.gdeltproject.org/events"

    def _get_export_url(self, target_date: date) -> str:
        """Get the export file URL for a specific date."""
        date_str = target_date.strftime("%Y%m%d")
        return f"{self.base_url}/{date_str}.export.CSV.zip"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_day(self, target_date: date) -> pd.DataFrame | None:
        """Download and parse GDELT data for a single day."""
        url = self._get_export_url(target_date)
        logger.info(f"Downloading GDELT data for {target_date}")

        try:
            response = self.client.get(url)
            if response.status_code == 404:
                logger.warning(f"No data available for {target_date}")
                return None
            response.raise_for_status()

            # Extract ZIP file
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                if not z.namelist():
                    logger.warning(f"Empty ZIP file for {target_date}")
                    return None
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        sep="\t",
                        header=None,
                        names=self.EVENT_COLUMNS,
                        dtype=str,
                        low_memory=False,
                    )

            # Parse dates
            df["SQLDATE"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
            df["DATEADDED"] = pd.to_datetime(
                df["DATEADDED"].astype(str).str.ljust(14, "0"),
                format="%Y%m%d%H%M%S",
            )

            # Parse numeric columns
            numeric_cols = ["GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parse geo coordinates
            geo_cols = [
                "Actor1Geo_Lat",
                "Actor1Geo_Long",
                "Actor2Geo_Lat",
                "Actor2Geo_Long",
                "ActionGeo_Lat",
                "ActionGeo_Long",
            ]
            for col in geo_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"Downloaded {len(df)} events for {target_date}")
            return df

        except Exception as e:
            logger.error(f"Failed to download GDELT for {target_date}: {e}")
            raise

    def extract_to_raw(
        self, output_dir: Path, start_date: date, end_date: date, run_id: str | None = None
    ) -> list[Path]:
        """Extract GDELT data to raw lake."""
        output_dir = Path(output_dir) / "gdelt"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        current_date = start_date

        while current_date <= end_date:
            try:
                df = self.download_day(current_date)
                if df is not None and not df.empty:
                    # Add metadata
                    df["extracted_at"] = datetime.now(UTC)
                    df["run_id"] = run_id

                    # Save to parquet
                    file_path = output_dir / f"gdelt_{current_date.isoformat()}.parquet"
                    df.to_parquet(file_path, index=False)
                    saved_files.append(file_path)
                    logger.info(f"Saved {len(df)} events to {file_path}")

            except Exception as e:
                logger.error(f"Error processing {current_date}: {e}")
                # Continue with next date

            current_date += timedelta(days=1)

        return saved_files


def extract_gdelt(
    output_dir: Path, start_date: str, end_date: str, run_id: str | None = None
) -> list[Path]:
    """CLI-friendly wrapper for GDELT extraction."""
    extractor = GDELTExtractor()

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    return extractor.extract_to_raw(
        output_dir=output_dir, start_date=start, end_date=end, run_id=run_id
    )
