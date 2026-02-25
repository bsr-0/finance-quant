"""Load raw data files into raw tables."""

import json
import logging
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class RawLoader:
    """Load raw parquet files into database raw tables."""

    def __init__(self):
        self.db = get_db_manager()
        self._batch_size = get_settings().infrastructure.batch_size

    def _batch_insert(self, conn, insert_sql, records: list[dict]) -> int:
        if not records:
            return 0
        total = 0
        for i in range(0, len(records), self._batch_size):
            batch = records[i : i + self._batch_size]
            conn.execute(insert_sql, batch)
            total += len(batch)
        return total

    def load_fred_observations(
        self,
        file_path: Path,
        run_id: UUID | None = None,
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
            ON CONFLICT (series_code, observation_date, realtime_start) DO UPDATE SET
                value = EXCLUDED.value,
                realtime_end = EXCLUDED.realtime_end,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "series_code": row.get("series_code"),
                    "date": row.get("date"),
                    "value": row.get("value"),
                    "realtime_start": row.get("realtime_start"),
                    "realtime_end": row.get("realtime_end"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
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

        insert_sql = text("""
            INSERT INTO raw_gdelt_events
            (gdelt_event_id, event_date, raw_data, extracted_at, run_id)
            VALUES (:event_id, :event_date, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (gdelt_event_id) DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "event_id": row.get("GLOBALEVENTID"),
                    "event_date": row.get("SQLDATE"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} GDELT events")
        return rows_loaded

    def load_polymarket_markets(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket market metadata from parquet file."""
        logger.info(f"Loading Polymarket markets from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_markets
            (venue_market_id, raw_data, extracted_at, run_id)
            VALUES (:market_id, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id) DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            market_id = row.get("id") or row.get("marketId")
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": market_id,
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket markets")
        return rows_loaded

    def load_polymarket_prices(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket price data from parquet file."""
        logger.info(f"Loading Polymarket prices from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_prices
            (venue_market_id, ts, outcome, price, raw_data, extracted_at, run_id)
            VALUES (:market_id, :ts, :outcome, :price, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id, ts, outcome) DO UPDATE SET
                price = EXCLUDED.price,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "ts": row.get("timestamp"),
                    "outcome": row.get("outcome") or "YES",
                    "price": row.get("price"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket prices")
        return rows_loaded

    def load_polymarket_trades(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket trades from parquet file."""
        logger.info(f"Loading Polymarket trades from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_trades
            (venue_market_id, trade_id, ts, price, size,
             side, raw_data, extracted_at, run_id)
            VALUES (:market_id, :trade_id, :ts, :price,
             :size, :side, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (trade_id) DO UPDATE SET
                price = EXCLUDED.price,
                size = EXCLUDED.size,
                side = EXCLUDED.side,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "trade_id": row.get("id") or row.get("trade_id"),
                    "ts": row.get("timestamp"),
                    "price": row.get("price"),
                    "size": row.get("size"),
                    "side": row.get("side"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket trades")
        return rows_loaded

    def load_polymarket_orderbooks(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket orderbook snapshots from parquet file."""
        logger.info(f"Loading Polymarket orderbooks from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_orderbook_snapshots
            (venue_market_id, ts, best_bid, best_ask, spread, bids, asks,
             raw_data, extracted_at, run_id)
            VALUES (:market_id, :ts, :best_bid, :best_ask, :spread, :bids, :asks,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id, ts) DO UPDATE SET
                best_bid = EXCLUDED.best_bid,
                best_ask = EXCLUDED.best_ask,
                spread = EXCLUDED.spread,
                bids = EXCLUDED.bids,
                asks = EXCLUDED.asks,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "ts": row.get("ts"),
                    "best_bid": row.get("best_bid"),
                    "best_ask": row.get("best_ask"),
                    "spread": row.get("spread"),
                    "bids": row.get("bids"),
                    "asks": row.get("asks"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket orderbook snapshots")
        return rows_loaded

    def load_prices_ohlcv(
        self,
        file_path: Path,
        run_id: UUID | None = None,
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
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume,
                split_ratio = EXCLUDED.split_ratio,
                dividend = EXCLUDED.dividend,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
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
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} OHLCV records")
        return rows_loaded

    def load_factor_returns(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load factor returns from parquet file."""
        logger.info(f"Loading factor data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_factor_returns
            (date, mkt_rf, smb, hml, rmw, cma, mom, rf, raw_data, extracted_at, run_id)
            VALUES (:date, :mkt_rf, :smb, :hml, :rmw, :cma, :mom, :rf,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (date) DO UPDATE SET
                mkt_rf = EXCLUDED.mkt_rf,
                smb = EXCLUDED.smb,
                hml = EXCLUDED.hml,
                rmw = EXCLUDED.rmw,
                cma = EXCLUDED.cma,
                mom = EXCLUDED.mom,
                rf = EXCLUDED.rf,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "date": row.get("date"),
                    "mkt_rf": row.get("mkt_rf"),
                    "smb": row.get("smb"),
                    "hml": row.get("hml"),
                    "rmw": row.get("rmw"),
                    "cma": row.get("cma"),
                    "mom": row.get("mom"),
                    "rf": row.get("rf"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} factor rows")
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
                    elif "orderbooks" in str(file_path):
                        total_rows += self.load_polymarket_orderbooks(file_path, run_id)
                elif source == "prices":
                    total_rows += self.load_prices_ohlcv(file_path, run_id)
                elif source == "factors":
                    total_rows += self.load_factor_returns(file_path, run_id)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        return total_rows
