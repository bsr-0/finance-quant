"""Fixed curated transformations with proper survivor bias handling and point-in-time adjustments."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class CuratedTransformerFixed:
    """Fixed transformer with proper quant handling."""
    
    def __init__(self):
        self.db = get_db_manager()
    
    def _get_source_id(self, source_name: str) -> Optional[UUID]:
        """Get or create source ID."""
        result = self.db.run_query(
            "SELECT source_id FROM dim_source WHERE name = :name",
            {"name": source_name}
        )
        if result:
            return UUID(result[0]["source_id"])
        
        with self.db.engine.connect() as conn:
            insert = text("""
                INSERT INTO dim_source (name, type, base_url)
                VALUES (:name, :type, :base_url)
                RETURNING source_id
            """)
            result = conn.execute(insert, {
                "name": source_name,
                "type": "api",
                "base_url": self._get_source_url(source_name)
            })
            source_id = result.scalar()
            conn.commit()
            return source_id
    
    def _get_source_url(self, source_name: str) -> str:
        """Get base URL for a source."""
        settings = get_settings()
        urls = {
            "fred": settings.fred.base_url,
            "gdelt": settings.gdelt.base_url,
            "polymarket": settings.polymarket.base_url,
            "prices": "https://finance.yahoo.com"
        }
        return urls.get(source_name, "")
    
    def transform_ticker_info(self) -> int:
        """Transform ticker info with proper delisting tracking."""
        logger.info("Transforming ticker info with survivor bias tracking...")
        
        # First, check if we have ticker info data
        has_ticker_info = self.db.run_query("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'raw_ticker_info'
            ) as exists
        """)[0]["exists"]
        
        if not has_ticker_info:
            logger.warning("No ticker info table found - using basic transform")
            return self._transform_ticker_info_basic()
        
        # Full transform with delisting info
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO dim_symbol 
                (ticker, exchange, asset_class, currency, start_date, end_date, is_delisted, external_ids)
                SELECT 
                    r.ticker,
                    r.exchange,
                    r.asset_class,
                    'USD' as currency,
                    r.first_trade_date as start_date,
                    r.last_trade_date as end_date,
                    r.is_delisted,
                    jsonb_build_object(
                        'delisted_date', r.delisted_date,
                        'extracted_at', r.extracted_at
                    ) as external_ids
                FROM raw_ticker_info r
                LEFT JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE s.symbol_id IS NULL
                ON CONFLICT (ticker, exchange) DO UPDATE SET
                    end_date = EXCLUDED.end_date,
                    is_delisted = EXCLUDED.is_delisted,
                    external_ids = EXCLUDED.external_ids,
                    updated_at = NOW()
            """))
            conn.commit()
            rows = result.rowcount
        
        logger.info(f"Transformed {rows} ticker info records")
        
        # Log survivor bias metrics
        bias_stats = self.db.run_query("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_delisted THEN 1 ELSE 0 END) as delisted,
                SUM(CASE WHEN NOT is_delisted THEN 1 ELSE 0 END) as active
            FROM dim_symbol
        """)
        
        if bias_stats:
            stats = bias_stats[0]
            total = stats["total"] or 0
            delisted = stats["delisted"] or 0
            survivor_bias_pct = (delisted / total * 100) if total > 0 else 0
            logger.info(f"Survivor bias: {delisted}/{total} tickers delisted ({survivor_bias_pct:.1f}%)")
        
        return rows
    
    def _transform_ticker_info_basic(self) -> int:
        """Basic ticker info transform without delisting data."""
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO dim_symbol 
                (ticker, exchange, asset_class, currency, start_date, is_delisted)
                SELECT DISTINCT 
                    r.ticker,
                    COALESCE(r.exchange, 'NYSE') as exchange,
                    'equity' as asset_class,
                    'USD' as currency,
                    MIN(r.date) as start_date,
                    false as is_delisted
                FROM raw_prices_ohlcv r
                LEFT JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE s.symbol_id IS NULL
                GROUP BY r.ticker, r.exchange
                ON CONFLICT (ticker, exchange) DO NOTHING
            """))
            conn.commit()
            return result.rowcount
    
    def transform_corporate_actions(self) -> int:
        """Transform corporate actions with proper point-in-time handling."""
        logger.info("Transforming corporate actions...")
        
        # Check if we have corporate actions data
        has_actions = self.db.run_query("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'raw_corporate_actions'
            ) as exists
        """)[0]["exists"]
        
        if not has_actions:
            logger.warning("No corporate actions table found")
            return 0
        
        with self.db.engine.connect() as conn:
            # Transform splits
            result_splits = conn.execute(text("""
                INSERT INTO cur_corporate_actions 
                (symbol_id, action_type, action_date, ratio, event_time, available_time, time_quality)
                SELECT 
                    s.symbol_id,
                    'split' as action_type,
                    r.action_date,
                    r.ratio,
                    r.action_date::timestamptz as event_time,
                    r.extracted_at as available_time,
                    'confirmed' as time_quality
                FROM raw_corporate_actions r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.action_type = 'split'
                ON CONFLICT (symbol_id, action_type, action_date) DO UPDATE SET
                    ratio = EXCLUDED.ratio,
                    available_time = EXCLUDED.available_time
            """))
            
            # Transform dividends
            result_divs = conn.execute(text("""
                INSERT INTO cur_corporate_actions 
                (symbol_id, action_type, action_date, amount, event_time, available_time, time_quality)
                SELECT 
                    s.symbol_id,
                    'dividend' as action_type,
                    r.action_date,
                    r.amount,
                    r.action_date::timestamptz as event_time,
                    r.extracted_at as available_time,
                    'confirmed' as time_quality
                FROM raw_corporate_actions r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.action_type = 'dividend'
                ON CONFLICT (symbol_id, action_type, action_date) DO UPDATE SET
                    amount = EXCLUDED.amount,
                    available_time = EXCLUDED.available_time
            """))
            
            conn.commit()
            total = (result_splits.rowcount or 0) + (result_divs.rowcount or 0)
        
        logger.info(f"Transformed {total} corporate actions")
        return total
    
    def transform_prices_ohlcv_fixed(self) -> Dict[str, int]:
        """Transform OHLCV with proper adjustments and quality flags."""
        logger.info("Transforming OHLCV prices with quality checks...")
        
        # First ensure symbols exist
        self.transform_ticker_info()
        
        results = {"inserted": 0, "updated": 0, "quality_issues": 0}
        
        with self.db.engine.connect() as conn:
            # Get data quality stats before transform
            quality_stats = conn.execute(text("""
                SELECT 
                    data_quality_flag,
                    COUNT(*) as cnt
                FROM raw_prices_ohlcv
                GROUP BY data_quality_flag
            """)).fetchall()
            
            for flag, cnt in quality_stats:
                if flag != "ok":
                    logger.warning(f"Data quality issue '{flag}': {cnt} records")
                    results["quality_issues"] += cnt
            
            # Transform prices - only quality-flagged 'ok' records
            result = conn.execute(text("""
                INSERT INTO cur_prices_ohlcv_daily 
                (symbol_id, date, open, high, low, close, adj_close, volume, 
                 event_time, available_time, time_quality, data_quality_flag)
                SELECT 
                    s.symbol_id,
                    r.date,
                    r.open,
                    r.high,
                    r.low,
                    r.close,
                    r.adj_close,
                    r.volume,
                    (r.date || ' 16:00:00 America/New_York')::timestamptz as event_time,
                    r.extracted_at as available_time,
                    CASE 
                        WHEN r.has_adjustment THEN 'adjusted'
                        ELSE 'unadjusted'
                    END as time_quality,
                    r.data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.data_quality_flag = 'ok'
                ON CONFLICT (symbol_id, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume,
                    available_time = EXCLUDED.available_time,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at = NOW()
            """))
            conn.commit()
            
            results["inserted"] = result.rowcount or 0
        
        logger.info(f"Transformed {results['inserted']} OHLCV records")
        return results
    
    def apply_point_in_time_adjustments(
        self,
        symbol_id: UUID,
        asof_date: date
    ) -> pd.DataFrame:
        """Apply point-in-time price adjustments for backtesting.
        
        This is critical - uses only adjustments known at asof_date.
        """
        # Get prices
        prices = self.db.run_query("""
            SELECT date, open, high, low, close, volume
            FROM cur_prices_ohlcv_daily
            WHERE symbol_id = :symbol_id
            ORDER BY date
        """, {"symbol_id": str(symbol_id)})
        
        if not prices:
            return pd.DataFrame()
        
        df = pd.DataFrame(prices)
        
        # Get corporate actions up to asof_date
        actions = self.db.run_query("""
            SELECT action_type, action_date, ratio, amount
            FROM cur_corporate_actions
            WHERE symbol_id = :symbol_id
              AND action_date <= :asof_date
              AND available_time <= :asof_date::timestamptz
            ORDER BY action_date DESC
        """, {"symbol_id": str(symbol_id), "asof_date": asof_date})
        
        if not actions:
            return df
        
        # Calculate cumulative adjustment factor
        adj_factor = 1.0
        for action in actions:
            if action["action_type"] == "split" and action["ratio"]:
                adj_factor *= action["ratio"]
            # Dividends would require more complex handling
        
        # Apply adjustment
        df["open"] = df["open"] * adj_factor
        df["high"] = df["high"] * adj_factor
        df["low"] = df["low"] * adj_factor
        df["close"] = df["close"] * adj_factor
        
        return df
    
    def detect_and_flag_anomalies(self) -> Dict[str, int]:
        """Detect and flag price anomalies in curated data."""
        logger.info("Detecting price anomalies...")
        
        anomalies = {"price_spikes": 0, "volume_spikes": 0, "ohlc_errors": 0}
        
        with self.db.engine.connect() as conn:
            # Detect price spikes (>50% daily change)
            price_spikes = conn.execute(text("""
                UPDATE cur_prices_ohlcv_daily
                SET data_quality_flag = 'price_spike'
                WHERE ABS(close / NULLIF(
                    LAG(close) OVER (PARTITION BY symbol_id ORDER BY date), 0
                ) - 1) > 0.50
                AND data_quality_flag = 'ok'
                RETURNING symbol_id, date
            """)).fetchall()
            anomalies["price_spikes"] = len(price_spikes)
            
            # Detect volume spikes (>10x average)
            volume_spikes = conn.execute(text("""
                WITH volume_stats AS (
                    SELECT 
                        symbol_id,
                        AVG(volume) as avg_volume,
                        STDDEV(volume) as std_volume
                    FROM cur_prices_ohlcv_daily
                    GROUP BY symbol_id
                )
                UPDATE cur_prices_ohlcv_daily p
                SET data_quality_flag = 'volume_spike'
                FROM volume_stats v
                WHERE p.symbol_id = v.symbol_id
                  AND p.volume > v.avg_volume + 10 * v.std_volume
                  AND p.data_quality_flag = 'ok'
                RETURNING p.symbol_id, p.date
            """)).fetchall()
            anomalies["volume_spikes"] = len(volume_spikes)
            
            # Detect OHLC logic errors
            ohlc_errors = conn.execute(text("""
                UPDATE cur_prices_ohlcv_daily
                SET data_quality_flag = 'ohlc_error'
                WHERE (high < low OR high < open OR high < close OR 
                       low > open OR low > close)
                AND data_quality_flag = 'ok'
                RETURNING symbol_id, date
            """)).fetchall()
            anomalies["ohlc_errors"] = len(ohlc_errors)
            
            conn.commit()
        
        total = sum(anomalies.values())
        logger.info(f"Flagged {total} anomalies: {anomalies}")
        return anomalies
    
    def build_survivor_unbiased_universe(
        self,
        asof_date: date
    ) -> List[UUID]:
        """Build survivor-unbiased universe as of a specific date.
        
        Includes tickers that were active at asof_date, even if later delisted.
        """
        result = self.db.run_query("""
            SELECT symbol_id
            FROM dim_symbol
            WHERE start_date <= :asof_date
              AND (end_date IS NULL OR end_date >= :asof_date)
        """, {"asof_date": asof_date})
        
        return [UUID(r["symbol_id"]) for r in result]
    
    def transform_macro_observations_fixed(self) -> int:
        """Transform macro with vintage tracking."""
        logger.info("Transforming macro observations with vintage tracking...")
        
        source_id = self._get_source_id("fred")
        
        with self.db.engine.connect() as conn:
            # Get series info
            conn.execute(text("""
                INSERT INTO dim_macro_series (provider_series_code, name, frequency, source_id)
                SELECT DISTINCT 
                    r.series_code,
                    r.series_code as name,
                    'monthly' as frequency,
                    :source_id
                FROM raw_fred_observations r
                LEFT JOIN dim_macro_series d ON r.series_code = d.provider_series_code
                WHERE d.series_id IS NULL
            """), {"source_id": source_id})
            conn.commit()
            
            # Transform with vintage info
            result = conn.execute(text("""
                INSERT INTO cur_macro_observations 
                (series_id, period_end, value, revision_id, event_time, available_time, time_quality)
                SELECT 
                    s.series_id,
                    r.observation_date as period_end,
                    r.value,
                    NULL as revision_id,
                    r.observation_date::timestamptz as event_time,
                    COALESCE(r.realtime_start, r.extracted_at) as available_time,
                    CASE 
                        WHEN r.realtime_start IS NOT NULL THEN 'confirmed'
                        ELSE 'assumed'
                    END as time_quality
                FROM raw_fred_observations r
                JOIN dim_macro_series s ON r.series_code = s.provider_series_code
                ON CONFLICT (series_id, period_end, revision_id) DO UPDATE SET
                    value = EXCLUDED.value,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    updated_at = NOW()
            """))
            conn.commit()
            rows = result.rowcount
        
        logger.info(f"Transformed {rows} macro observations")
        return rows
    
    def transform_all_fixed(self) -> Dict[str, any]:
        """Run all fixed transformations."""
        results = {}
        results["ticker_info"] = self.transform_ticker_info()
        results["corporate_actions"] = self.transform_corporate_actions()
        results["prices_ohlcv"] = self.transform_prices_ohlcv_fixed()
        results["macro_observations"] = self.transform_macro_observations_fixed()
        results["anomalies"] = self.detect_and_flag_anomalies()
        return results


class PointInTimeJoin:
    """Utilities for point-in-time joins - critical for backtesting."""
    
    def __init__(self):
        self.db = get_db_manager()
    
    def get_price_asof(
        self,
        symbol_id: UUID,
        asof_ts: datetime
    ) -> Optional[Dict]:
        """Get price as of a specific timestamp with proper lookback."""
        result = self.db.run_query("""
            SELECT date, open, high, low, close, volume
            FROM cur_prices_ohlcv_daily
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
            ORDER BY date DESC
            LIMIT 1
        """, {"symbol_id": str(symbol_id), "asof_ts": asof_ts})
        
        return dict(result[0]) if result else None
    
    def get_macro_asof(
        self,
        series_code: str,
        asof_ts: datetime
    ) -> Optional[Dict]:
        """Get macro observation as of timestamp with vintage handling."""
        result = self.db.run_query("""
            SELECT o.value, o.period_end, o.time_quality
            FROM cur_macro_observations o
            JOIN dim_macro_series s ON o.series_id = s.series_id
            WHERE s.provider_series_code = :series_code
              AND o.available_time <= :asof_ts
            ORDER BY o.period_end DESC
            LIMIT 1
        """, {"series_code": series_code, "asof_ts": asof_ts})
        
        return dict(result[0]) if result else None
    
    def get_staleness_days(
        self,
        table: str,
        date_col: str,
        asof_ts: datetime
    ) -> int:
        """Calculate how stale the data is."""
        result = self.db.run_query(f"""
            SELECT EXTRACT(DAY FROM (:asof_ts - MAX({date_col}))) as days_stale
            FROM {table}
        """, {"asof_ts": asof_ts})
        
        return int(result[0]["days_stale"]) if result and result[0]["days_stale"] else 999
