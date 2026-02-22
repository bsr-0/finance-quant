"""Fixed contract snapshot builder with proper quant handling."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from pipeline.db import get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality report for a snapshot."""
    price_staleness_hours: float
    macro_staleness_days: float
    has_outliers: bool
    outlier_score: float
    data_completeness_pct: float


class ContractSnapshotBuilderFixed:
    """Fixed snapshot builder with proper quant handling."""
    
    def __init__(self):
        self.db = get_db_manager()
        self._price_cache: Dict[UUID, pd.DataFrame] = {}
        self._macro_cache: Optional[pd.DataFrame] = None
        self._event_cache: Optional[pd.DataFrame] = None
    
    def _get_contract_prices_cached(
        self,
        contract_id: UUID,
        start_ts: datetime,
        end_ts: datetime
    ) -> pd.DataFrame:
        """Get prices with caching."""
        cache_key = (contract_id, start_ts.date(), end_ts.date())
        
        if contract_id not in self._price_cache:
            query = """
                SELECT ts, price_normalized, available_time
                FROM cur_contract_prices
                WHERE contract_id = :contract_id
                  AND ts BETWEEN :start_ts AND :end_ts
                ORDER BY ts
            """
            result = self.db.run_query(query, {
                "contract_id": str(contract_id),
                "start_ts": start_ts,
                "end_ts": end_ts
            })
            
            if result:
                df = pd.DataFrame(result)
                df["ts"] = pd.to_datetime(df["ts"])
                df["available_time"] = pd.to_datetime(df["available_time"])
                self._price_cache[contract_id] = df
            else:
                return pd.DataFrame()
        
        return self._price_cache.get(contract_id, pd.DataFrame())
    
    def _detect_price_outliers(
        self,
        prices: pd.Series,
        method: str = "iqr"
    ) -> Tuple[pd.Series, float]:
        """Detect price outliers using robust methods.
        
        Returns:
            Tuple of (outlier_mask, outlier_score)
        """
        if len(prices) < 10:
            return pd.Series([False] * len(prices)), 0.0
        
        if method == "iqr":
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR  # 3x IQR for outliers
            upper = Q3 + 3 * IQR
            outlier_mask = (prices < lower) | (prices > upper)
            outlier_score = outlier_mask.sum() / len(prices)
            
        elif method == "zscore":
            zscores = np.abs(stats.zscore(prices, nan_policy='omit'))
            outlier_mask = zscores > 4.0  # 4 sigma
            outlier_score = (zscores > 4.0).sum() / len(prices)
            
        elif method == "mad":  # Median Absolute Deviation - more robust
            median = prices.median()
            mad = np.median(np.abs(prices - median))
            modified_z = 0.6745 * (prices - median) / mad if mad > 0 else 0
            outlier_mask = np.abs(modified_z) > 3.5
            outlier_score = outlier_mask.sum() / len(prices)
            
        else:
            outlier_mask = pd.Series([False] * len(prices))
            outlier_score = 0.0
        
        return outlier_mask, outlier_score
    
    def _calculate_microstructure_features(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate market microstructure features."""
        if trades_df.empty or len(trades_df) < 5:
            return {
                "trade_imbalance": 0.0,
                "buy_sell_ratio": 1.0,
                "avg_trade_size": 0.0,
                "trade_size_variance": 0.0,
                "price_impact": 0.0
            }
        
        # Trade imbalance (buy vs sell volume)
        buy_volume = trades_df[trades_df["side"] == "buy"]["size"].sum()
        sell_volume = trades_df[trades_df["side"] == "sell"]["size"].sum()
        total_volume = buy_volume + sell_volume
        
        trade_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        
        # Trade size statistics
        avg_trade_size = trades_df["size"].mean()
        trade_size_variance = trades_df["size"].var()
        
        # Price impact (correlation between trade size and price change)
        if len(trades_df) > 1:
            trades_df = trades_df.sort_values("ts")
            trades_df["price_change"] = trades_df["price"].diff()
            price_impact = trades_df["size"].corr(trades_df["price_change"].abs())
            price_impact = price_impact if not np.isnan(price_impact) else 0.0
        else:
            price_impact = 0.0
        
        return {
            "trade_imbalance": trade_imbalance,
            "buy_sell_ratio": buy_sell_ratio,
            "avg_trade_size": avg_trade_size,
            "trade_size_variance": trade_size_variance,
            "price_impact": price_impact
        }
    
    def _get_staleness_metrics(
        self,
        contract_id: UUID,
        asof_ts: datetime
    ) -> DataQualityReport:
        """Calculate data staleness metrics."""
        # Price staleness
        latest_price = self.db.run_query("""
            SELECT ts, available_time
            FROM cur_contract_prices
            WHERE contract_id = :contract_id
            ORDER BY ts DESC
            LIMIT 1
        """, {"contract_id": str(contract_id)})
        
        if latest_price:
            price_staleness = (asof_ts - latest_price[0]["ts"]).total_seconds() / 3600
        else:
            price_staleness = 9999
        
        # Macro staleness
        latest_macro = self.db.run_query("""
            SELECT MAX(period_end) as latest_date
            FROM cur_macro_observations
        """)
        
        if latest_macro and latest_macro[0]["latest_date"]:
            macro_staleness = (asof_ts.date() - latest_macro[0]["latest_date"]).days
        else:
            macro_staleness = 9999
        
        return DataQualityReport(
            price_staleness_hours=price_staleness,
            macro_staleness_days=macro_staleness,
            has_outliers=False,  # Will be set later
            outlier_score=0.0,
            data_completeness_pct=100.0
        )
    
    def build_contract_snapshot_fixed(
        self,
        contract_id: UUID,
        asof_ts: datetime,
        lookback_windows: Optional[Dict[str, timedelta]] = None
    ) -> Optional[Dict]:
        """Build snapshot with proper quant handling."""
        if lookback_windows is None:
            lookback_windows = {
                "1h": timedelta(hours=1),
                "6h": timedelta(hours=6),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7)
            }
        
        snapshot = {
            "contract_id": contract_id,
            "asof_ts": asof_ts,
            "event_time": asof_ts,
            "available_time": asof_ts
        }
        
        # Get price staleness metrics
        quality_report = self._get_staleness_metrics(contract_id, asof_ts)
        snapshot["price_staleness_hours"] = quality_report.price_staleness_hours
        snapshot["macro_staleness_days"] = quality_report.macro_staleness_days
        
        # Warn if data is stale
        if quality_report.price_staleness_hours > 24:
            logger.warning(f"Price data is {quality_report.price_staleness_hours:.1f} hours stale for {contract_id}")
        
        # Get latest price with proper filtering
        prices_df = self._get_contract_prices_cached(
            contract_id,
            asof_ts - timedelta(days=7),
            asof_ts
        )
        
        if prices_df.empty:
            return None
        
        # Filter to available data
        available_prices = prices_df[prices_df["available_time"] <= asof_ts]
        
        if available_prices.empty:
            return None
        
        latest = available_prices.iloc[-1]
        snapshot["implied_p_yes"] = latest["price_normalized"]
        snapshot["last_price_ts"] = latest["ts"]
        
        # Calculate features for each lookback window
        for window_name, window_delta in lookback_windows.items():
            window_start = asof_ts - window_delta
            window_prices = available_prices[
                (available_prices["ts"] >= window_start) &
                (available_prices["ts"] <= asof_ts)
            ]
            
            if not window_prices.empty:
                prices_series = window_prices["price_normalized"]
                
                # Basic stats
                snapshot[f"price_mean_{window_name}"] = prices_series.mean()
                snapshot[f"price_std_{window_name}"] = prices_series.std()
                snapshot[f"price_min_{window_name}"] = prices_series.min()
                snapshot[f"price_max_{window_name}"] = prices_series.max()
                
                # Returns
                if len(prices_series) > 1:
                    snapshot[f"price_return_{window_name}"] = (
                        prices_series.iloc[-1] / prices_series.iloc[0] - 1
                    )
                
                # Outlier detection for 24h window
                if window_name == "24h":
                    outlier_mask, outlier_score = self._detect_price_outliers(
                        prices_series, method="mad"
                    )
                    snapshot["has_price_outliers"] = outlier_mask.any()
                    snapshot["outlier_score"] = outlier_score
                    quality_report.has_outliers = outlier_mask.any()
                    quality_report.outlier_score = outlier_score
        
        # Get trade statistics
        trades_df = self._get_trades_in_window(contract_id, asof_ts - timedelta(hours=24), asof_ts)
        
        if not trades_df.empty:
            snapshot["volume_24h"] = trades_df["size"].sum()
            snapshot["trade_count_24h"] = len(trades_df)
            snapshot["price_volatility_24h"] = trades_df["price"].std()
            
            # Microstructure features
            micro_features = self._calculate_microstructure_features(trades_df)
            snapshot.update({f"micro_{k}": v for k, v in micro_features.items()})
        else:
            snapshot["volume_24h"] = 0
            snapshot["trade_count_24h"] = 0
        
        # Get macro panel
        macro_panel = self._get_macro_panel_fixed(asof_ts)
        snapshot["macro_panel"] = macro_panel
        
        # Get event stats
        event_stats = self._get_event_stats_fixed(asof_ts - timedelta(hours=24), asof_ts)
        snapshot["event_counts_24h"] = event_stats.get("count", 0)
        snapshot["event_tone_avg"] = event_stats.get("avg_tone", 0)
        
        # Data quality flag
        snapshot["data_quality_score"] = self._calculate_quality_score(quality_report)
        
        return snapshot
    
    def _get_trades_in_window(
        self,
        contract_id: UUID,
        start_ts: datetime,
        end_ts: datetime
    ) -> pd.DataFrame:
        """Get trades in time window."""
        query = """
            SELECT ts, price, size, side, available_time
            FROM cur_contract_trades
            WHERE contract_id = :contract_id
              AND available_time <= :end_ts
              AND ts BETWEEN :start_ts AND :end_ts
            ORDER BY ts
        """
        result = self.db.run_query(query, {
            "contract_id": str(contract_id),
            "start_ts": start_ts,
            "end_ts": end_ts
        })
        
        if result:
            df = pd.DataFrame(result)
            df["ts"] = pd.to_datetime(df["ts"])
            return df
        return pd.DataFrame()
    
    def _get_macro_panel_fixed(self, asof_ts: datetime) -> Dict[str, float]:
        """Get macro panel with vintage handling."""
        query = """
            SELECT DISTINCT ON (s.provider_series_code)
                s.provider_series_code as series_code,
                o.value,
                o.period_end,
                o.time_quality
            FROM cur_macro_observations o
            JOIN dim_macro_series s ON o.series_id = s.series_id
            WHERE o.available_time <= :asof_ts
            ORDER BY s.provider_series_code, o.period_end DESC
        """
        results = self.db.run_query(query, {"asof_ts": asof_ts})
        
        panel = {}
        for r in results:
            panel[r["series_code"]] = {
                "value": r["value"],
                "period_end": r["period_end"].isoformat() if r["period_end"] else None,
                "quality": r["time_quality"]
            }
        
        return panel
    
    def _get_event_stats_fixed(
        self,
        start_ts: datetime,
        end_ts: datetime
    ) -> Dict[str, float]:
        """Get event stats with proper filtering."""
        query = """
            SELECT 
                COUNT(*) as count,
                COALESCE(AVG(tone_score), 0) as avg_tone,
                COALESCE(STDDEV(tone_score), 0) as tone_std,
                COALESCE(AVG(sentiment_positive), 0) as avg_pos,
                COALESCE(AVG(sentiment_negative), 0) as avg_neg
            FROM cur_world_events
            WHERE available_time <= :end_ts
              AND event_time BETWEEN :start_ts AND :end_ts
        """
        result = self.db.run_query(query, {
            "start_ts": start_ts,
            "end_ts": end_ts
        })
        
        return dict(result[0]) if result else {"count": 0, "avg_tone": 0}
    
    def _calculate_quality_score(self, report: DataQualityReport) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct for staleness
        if report.price_staleness_hours > 1:
            score -= min(30, report.price_staleness_hours / 2)
        
        if report.macro_staleness_days > 1:
            score -= min(20, report.macro_staleness_days)
        
        # Deduct for outliers
        score -= report.outlier_score * 20
        
        return max(0, min(100, score))
    
    def build_snapshots_for_range_fixed(
        self,
        contract_ids: Optional[List[UUID]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        frequency: str = "1h"
    ) -> Dict[str, any]:
        """Build snapshots with reporting."""
        if contract_ids is None:
            result = self.db.run_query(
                "SELECT contract_id FROM dim_contract WHERE status = 'active'"
            )
            contract_ids = [UUID(r["contract_id"]) for r in result]
        
        if not start_ts:
            start_ts = datetime.utcnow() - timedelta(days=30)
        if not end_ts:
            end_ts = datetime.utcnow()
        
        freq_map = {"1h": "H", "1d": "D", "15min": "15min"}
        pandas_freq = freq_map.get(frequency, "H")
        timestamps = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq).tolist()
        
        total_snapshots = 0
        quality_issues = 0
        stale_data_count = 0
        
        for contract_id in contract_ids:
            for ts in timestamps:
                snapshot = self.build_contract_snapshot_fixed(contract_id, ts)
                
                if snapshot:
                    self._save_snapshot_fixed(snapshot)
                    total_snapshots += 1
                    
                    if snapshot.get("price_staleness_hours", 0) > 24:
                        stale_data_count += 1
                    
                    if snapshot.get("data_quality_score", 100) < 70:
                        quality_issues += 1
        
        report = {
            "total_snapshots": total_snapshots,
            "contracts_processed": len(contract_ids),
            "timestamps_per_contract": len(timestamps),
            "stale_data_count": stale_data_count,
            "quality_issues": quality_issues
        }
        
        logger.info(f"Snapshot build complete: {report}")
        return report
    
    def _save_snapshot_fixed(self, snapshot: Dict) -> None:
        """Save snapshot with extended schema."""
        import json
        
        with self.db.engine.connect() as conn:
            # Check if extended columns exist, if not use basic insert
            columns = self.db.run_query("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'snap_contract_features'
            """)
            column_names = [c["column_name"] for c in columns]
            
            # Build dynamic insert
            base_fields = [
                "contract_id", "asof_ts", "implied_p_yes", "spread",
                "volume_24h", "trade_count_24h", "price_volatility_24h",
                "macro_panel", "event_counts_24h", "event_tone_avg",
                "event_time", "available_time"
            ]
            
            # Add extended fields if they exist
            extended_fields = [
                "price_staleness_hours", "macro_staleness_days",
                "has_price_outliers", "outlier_score", "data_quality_score"
            ]
            
            fields = base_fields + [f for f in extended_fields if f in column_names]
            
            placeholders = ", ".join([f":{f}" for f in fields])
            columns_str = ", ".join(fields)
            
            insert = text(f"""
                INSERT INTO snap_contract_features ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (contract_id, asof_ts) DO UPDATE SET
                    implied_p_yes = EXCLUDED.implied_p_yes,
                    volume_24h = EXCLUDED.volume_24h,
                    updated_at = NOW()
            """)
            
            params = {f: snapshot.get(f) for f in fields}
            params["macro_panel"] = json.dumps(params.get("macro_panel", {}))
            
            conn.execute(insert, params)
            conn.commit()


def build_snapshots_fixed(
    contracts: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "1h"
) -> Dict[str, any]:
    """CLI-friendly wrapper for fixed snapshot building."""
    builder = ContractSnapshotBuilderFixed()
    
    contract_ids = [UUID(c) for c in contracts] if contracts else None
    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None
    
    return builder.build_snapshots_for_range_fixed(
        contract_ids=contract_ids,
        start_ts=start_ts,
        end_ts=end_ts,
        frequency=freq
    )
