"""FRED (Federal Reserve Economic Data) extractor."""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class FredExtractor:
    """Extract economic data from FRED API."""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings().fred
        self.api_key = api_key or settings.api_key
        self.base_url = settings.base_url
        self.series_codes = settings.series_codes
        self.client = httpx.Client(timeout=30.0)
    
    def __del__(self):
        self.client.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_series_info(self, series_code: str) -> dict:
        """Get metadata for a series."""
        data = self._make_request("series", {"series_id": series_code})
        return data.get("seriess", [{}])[0]
    
    def get_observations(
        self, 
        series_code: str, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
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
        series_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_id: Optional[str] = None
    ) -> List[Path]:
        """Extract series data to raw lake."""
        codes = series_codes or self.series_codes
        output_dir = Path(output_dir) / "fred"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for code in codes:
            logger.info(f"Extracting FRED series: {code}")
            try:
                df = self.get_observations(code, start_date, end_date)
                if df.empty:
                    continue
                
                # Add metadata
                df["extracted_at"] = datetime.utcnow()
                df["run_id"] = run_id
                
                # Save to parquet
                file_path = output_dir / f"{code}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                logger.info(f"Saved {len(df)} observations to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to extract {code}: {e}")
                raise
        
        return saved_files


def extract_fred(
    output_dir: Path,
    series_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None
) -> List[Path]:
    """CLI-friendly wrapper for FRED extraction."""
    settings = get_settings()
    
    extractor = FredExtractor()
    
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    
    return extractor.extract_to_raw(
        output_dir=output_dir,
        series_codes=series_codes,
        start_date=start,
        end_date=end,
        run_id=run_id
    )
