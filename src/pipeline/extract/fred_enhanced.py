"""Enhanced FRED extractor with async, batching, and validation."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.infrastructure.async_pool import get_async_pool
from pipeline.infrastructure.batch_processor import BatchConfig, BatchInserter
from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.checkpoint import CheckpointManager
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.infrastructure.validation import BatchValidator, FredObservationValidator
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class EnhancedFredExtractor:
    """Enhanced FRED extractor with scalability features."""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings().fred
        self.api_key = api_key or settings.api_key
        self.base_url = settings.base_url
        self.series_codes = settings.series_codes
        
        # Circuit breaker for API calls
        self.circuit = get_circuit_breaker(
            "fred_api",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # Async client
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        # Metrics
        self.metrics = PipelineMetrics("fred_extract")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
    )
    async def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make async API request with retry and circuit breaker."""
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        async with self.metrics.time_operation("api_request"):
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    
    async def get_series_info(self, series_code: str) -> dict:
        """Get metadata for a series."""
        data = await self._make_request("series", {"series_id": series_code})
        return data.get("seriess", [{}])[0]
    
    async def get_observations(
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
        
        data = await self._make_request("series/observations", params)
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
    
    async def extract_series_parallel(
        self,
        series_codes: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[tuple[str, pd.DataFrame]]:
        """Extract multiple series in parallel."""
        async def extract_one(code: str) -> tuple[str, pd.DataFrame]:
            try:
                df = await self.get_observations(code, start_date, end_date)
                return code, df
            except Exception as e:
                logger.error(f"Failed to extract {code}: {e}")
                return code, pd.DataFrame()
        
        tasks = [extract_one(code) for code in series_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Series extraction failed: {result}")
            else:
                successful.append(result)
        
        return successful
    
    async def extract_to_raw(
        self,
        output_dir: Path,
        series_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_id: Optional[str] = None,
        validate: bool = True
    ) -> List[Path]:
        """Extract series data to raw lake with validation."""
        codes = series_codes or self.series_codes
        output_dir = Path(output_dir) / "fred"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint for resumability
        checkpoint_mgr = CheckpointManager(output_dir / ".checkpoints")
        
        with checkpoint_mgr.checkpoint_context(
            f"fred_extract_{run_id}",
            resume=True
        ) as ctx:
            # Get already processed codes
            processed = set(ctx.state.get("processed_codes", []))
            remaining = [c for c in codes if c not in processed]
            
            if not remaining:
                logger.info("All series already processed")
                return []
            
            logger.info(f"Processing {len(remaining)} series ({len(processed)} already done)")
            
            # Extract in parallel
            results = await self.extract_series_parallel(
                remaining, start_date, end_date
            )
            
            saved_files = []
            validator = BatchValidator(FredObservationValidator, max_errors=100)
            
            for code, df in results:
                if df.empty:
                    continue
                
                # Validate
                if validate:
                    records = df.to_dict("records")
                    valid_records, val_result = validator.validate_batch(records)
                    
                    if not val_result.is_valid:
                        logger.warning(f"Validation issues for {code}: {len(val_result.errors)} errors")
                    
                    df = pd.DataFrame(valid_records)
                
                # Add metadata
                df["extracted_at"] = datetime.utcnow()
                df["run_id"] = run_id
                
                # Save to parquet with partitioning
                date_part = f"{start_date}_{end_date}" if start_date and end_date else "all"
                file_path = output_dir / f"{code}_{date_part}.parquet"
                
                # Use efficient compression
                df.to_parquet(
                    file_path,
                    index=False,
                    compression="zstd",
                    engine="pyarrow"
                )
                
                saved_files.append(file_path)
                self.metrics.record_extracted("fred", len(df))
                
                # Update checkpoint
                processed.add(code)
                ctx.update(processed_codes=list(processed), last_processed=code)
                ctx.save()
                
                logger.info(f"Saved {len(df)} observations for {code}")
            
            return saved_files
    
    def extract_to_raw_sync(
        self,
        output_dir: Path,
        series_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_id: Optional[str] = None,
        validate: bool = True
    ) -> List[Path]:
        """Synchronous wrapper for extract_to_raw."""
        return asyncio.run(self.extract_to_raw(
            output_dir, series_codes, start_date, end_date, run_id, validate
        ))


class FredBulkLoader:
    """Bulk loader for FRED data with batching."""
    
    def __init__(self):
        self.metrics = PipelineMetrics("fred_load")
    
    def load_parquet_to_raw(
        self,
        file_path: Path,
        run_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> int:
        """Load parquet file to raw table with batching."""
        import pandas as pd
        from pipeline.db import get_db_manager
        
        df = pd.read_parquet(file_path)
        if df.empty:
            return 0
        
        db = get_db_manager()
        total_loaded = 0
        
        config = BatchConfig(batch_size=batch_size)
        columns = [
            "series_code", "observation_date", "value",
            "raw_data", "extracted_at", "run_id"
        ]
        
        with BatchInserter("raw_fred_observations", columns, config) as inserter:
            for _, row in df.iterrows():
                record = {
                    "series_code": row.get("series_code"),
                    "observation_date": row.get("date"),
                    "value": row.get("value"),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", datetime.utcnow()),
                    "run_id": run_id
                }
                inserter.add(record)
                total_loaded += 1
        
        self.metrics.record_loaded("raw_fred_observations", total_loaded)
        logger.info(f"Loaded {total_loaded} records from {file_path}")
        
        return total_loaded


def extract_fred_enhanced(
    output_dir: Path,
    series_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None,
    validate: bool = True
) -> List[Path]:
    """CLI-friendly wrapper for enhanced FRED extraction."""
    settings = get_settings()
    
    extractor = EnhancedFredExtractor()
    
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    
    return extractor.extract_to_raw_sync(
        output_dir=output_dir,
        series_codes=series_codes,
        start_date=start,
        end_date=end,
        run_id=run_id,
        validate=validate
    )
