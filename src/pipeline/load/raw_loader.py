"""Load raw data files into raw tables."""

import logging
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager

logger = logging.getLogger(__name__)


class RawLoader:
    """Load raw parquet files into database raw tables."""

    def __init__(self):
        self.db = get_db_manager()
    
    def load_fred_observations(
        self,
        file_path: Path,
        run_id: Optional[UUID] = None,
    ) -> int:
        """Load FRED observations from parquet file.

        Persists ``realtime_start`` / ``realtime_end`` from the FRED API so
        that the curated layer can derive correct ``available_time`` (the date
        FRED first published the data point) rather than using the pipeline's
        ``extracted_at`` timestamp.
        """
        logger.info(f"Loading FRED data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            logger.warning(f"Empty file: {file_path}")
            return 0

        df["run_id"] = run_id

        insert_sql = text("""
            INSERT INTO raw_fred_observations
            (series_code, observation_date, value, realtime_start, realtime_end,
             raw_data, extracted_at, run_id)
            VALUES (:series_code, :date, :value, :realtime_start, :realtime_end,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT DO NOTHING
        """)

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                params = {
                    "series_code": row["series_code"],
                    "date": row["date"],
                    "value": row["value"],
                    "realtime_start": row.get("realtime_start"),
                    "realtime_end": row.get("realtime_end"),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }
                conn.execute(insert_sql, params)
                rows_loaded += 1
            conn.commit()

        logger.info(f"Loaded {rows_loaded} FRED observations")
        return rows_loaded

    def load_gdelt_events(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load GDELT events from parquet file."""
        logger.info(f"Loading GDELT data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            logger.warning(f"Empty file: {file_path}")
            return 0

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO raw_gdelt_events
                    (gdelt_event_id, event_date, raw_data, extracted_at, run_id)
                    VALUES (:event_id, :event_date, :raw_data, :extracted_at, :run_id)
                    ON CONFLICT DO NOTHING
                """)

                params = {
                    "event_id": row.get("GLOBALEVENTID"),
                    "event_date": row.get("SQLDATE"),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }

                conn.execute(insert_sql, params)
                rows_loaded += 1

            conn.commit()

        logger.info(f"Loaded {rows_loaded} GDELT events")
        return rows_loaded

    def load_polymarket_markets(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket market metadata from parquet file."""
        logger.info(f"Loading Polymarket markets from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO raw_polymarket_markets
                    (venue_market_id, raw_data, extracted_at, run_id)
                    VALUES (:market_id, :raw_data, :extracted_at, :run_id)
                    ON CONFLICT DO NOTHING
                """)

                market_id = row.get("id") or row.get("marketId")
                params = {
                    "market_id": market_id,
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }

                conn.execute(insert_sql, params)
                rows_loaded += 1

            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket markets")
        return rows_loaded

    def load_polymarket_prices(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket price data from parquet file."""
        logger.info(f"Loading Polymarket prices from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO raw_polymarket_prices
                    (venue_market_id, ts, outcome, price, raw_data, extracted_at, run_id)
                    VALUES (:market_id, :ts, :outcome, :price, :raw_data, :extracted_at, :run_id)
                    ON CONFLICT DO NOTHING
                """)

                params = {
                    "market_id": row.get("market_id"),
                    "ts": row.get("timestamp"),
                    "outcome": row.get("outcome"),
                    "price": row.get("price"),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }

                conn.execute(insert_sql, params)
                rows_loaded += 1

            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket prices")
        return rows_loaded

    def load_polymarket_trades(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket trades from parquet file."""
        logger.info(f"Loading Polymarket trades from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO raw_polymarket_trades
                    (venue_market_id, trade_id, ts, price, size,
                     side, raw_data, extracted_at, run_id)
                    VALUES (:market_id, :trade_id, :ts, :price,
                     :size, :side, :raw_data, :extracted_at, :run_id)
                    ON CONFLICT DO NOTHING
                """)

                params = {
                    "market_id": row.get("market_id"),
                    "trade_id": row.get("id") or row.get("trade_id"),
                    "ts": row.get("timestamp"),
                    "price": row.get("price"),
                    "size": row.get("size"),
                    "side": row.get("side"),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }

                conn.execute(insert_sql, params)
                rows_loaded += 1

            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket trades")
        return rows_loaded
      
    def load_prices_ohlcv(
        self,
        file_path: Path,
        run_id: Optional[UUID] = None,
    ) -> int:
        """Load OHLCV price data from parquet file.

        Also persists ``split_ratio`` and ``dividend`` columns so that the
        curated transform can populate ``cur_corporate_actions``.
        """
        logger.info(f"Loading OHLCV data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_prices_ohlcv
            (ticker, date, open, high, low, close, adj_close, volume,
             split_ratio, dividend, raw_data, extracted_at, run_id)
            VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume,
                    :split_ratio, :dividend, :raw_data, :extracted_at, :run_id)
            ON CONFLICT DO NOTHING
        """)

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            for _, row in df.iterrows():
                params = {
                    "ticker": row.get("ticker"),
                    "date": row.get("date"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "adj_close": row.get("adj_close"),
                    "volume": row.get("volume"),
                    "split_ratio": row.get("split_ratio"),
                    "dividend": row.get("dividend", 0),
                    "raw_data": row.to_json(),
                    "extracted_at": row.get("extracted_at", pd.Timestamp.now()),
                    "run_id": run_id,
                }
                conn.execute(insert_sql, params)
                rows_loaded += 1
            conn.commit()

        logger.info(f"Loaded {rows_loaded} OHLCV records")
        return rows_loaded

    def load_all_raw_files(self, raw_dir: Path, source: str, run_id: UUID | None = None) -> int:
        """Load all raw files for a source."""
        source_dir = raw_dir / source
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return 0

        total_rows = 0
        parquet_files = list(source_dir.rglob("*.parquet"))

        for file_path in parquet_files:
            try:
                if source == "fred":
                    total_rows += self.load_fred_observations(file_path, run_id)
                elif source == "gdelt":
                    total_rows += self.load_gdelt_events(file_path, run_id)
                elif source == "polymarket":
                    if "markets" in str(file_path):
                        total_rows += self.load_polymarket_markets(file_path, run_id)
                    elif "prices" in str(file_path):
                        total_rows += self.load_polymarket_prices(file_path, run_id)
                    elif "trades" in str(file_path):
                        total_rows += self.load_polymarket_trades(file_path, run_id)
                elif source == "prices":
                    total_rows += self.load_prices_ohlcv(file_path, run_id)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        return total_rows
