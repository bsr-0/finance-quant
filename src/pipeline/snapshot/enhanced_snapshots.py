"""Enhanced snapshot builder with batching and parallel processing."""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.features.technical_indicators import ContractFeatureEngineer
from pipeline.infrastructure.batch_processor import BatchConfig, BatchInserter
from pipeline.infrastructure.checkpoint import CheckpointManager
from pipeline.infrastructure.lineage import LineageContext
from pipeline.infrastructure.metrics import PipelineMetrics

logger = logging.getLogger(__name__)


class EnhancedSnapshotBuilder:
    """High-performance snapshot builder with batching and caching."""
    
    def __init__(self, max_workers: int = 4):
        self.db = get_db_manager()
        self.max_workers = max_workers
        self.metrics = PipelineMetrics("snapshot_build")
        self._price_cache: Dict[UUID, pd.DataFrame] = {}
        self._macro_cache: Optional[pd.DataFrame] = None
        self._event_cache: Optional[pd.DataFrame] = None
    
    def _get_contract_prices(
        self,
        contract_id: UUID,
        start_ts: datetime,
        end_ts: datetime
    ) -> pd.DataFrame:
        """Get prices for a contract with caching."""
        cache_key = (contract_id, start_ts, end_ts)
        
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
                self._price_cache[contract_id] = df
            else:
                return pd.DataFrame()
        
        return self._price_cache.get(contract_id, pd.DataFrame())
    
    def _get_macro_panel(self, asof_ts: datetime) -> Dict[str, float]:
        """Get macro panel with caching."""
        if self._macro_cache is None:
            query = """
                SELECT 
                    s.provider_series_code as series_code,
                    o.value,
                    o.period_end,
                    o.available_time
                FROM cur_macro_observations o
                JOIN dim_macro_series s ON o.series_id = s.series_id
                ORDER BY s.provider_series_code, o.period_end DESC
            """
            result = self.db.run_query(query)
            
            if result:
                self._macro_cache = pd.DataFrame(result)
                self._macro_cache["available_time"] = pd.to_datetime(
                    self._macro_cache["available_time"]
                )
            else:
                return {}
        
        # Filter to available data at asof_ts
        if not self._macro_cache.empty:
            available = self._macro_cache[
                self._macro_cache["available_time"] <= asof_ts
            ]
            # Get most recent per series
            latest = available.groupby("series_code").first()
            return latest["value"].to_dict()
        
        return {}
    
    def _get_event_stats(
        self,
        start_ts: datetime,
        end_ts: datetime
    ) -> Dict[str, float]:
        """Get event statistics."""
        if self._event_cache is None:
            query = """
                SELECT event_time, tone_score, available_time
                FROM cur_world_events
                WHERE event_time > NOW() - INTERVAL '30 days'
            """
            result = self.db.run_query(query)
            
            if result:
                self._event_cache = pd.DataFrame(result)
                self._event_cache["event_time"] = pd.to_datetime(
                    self._event_cache["event_time"]
                )
                self._event_cache["available_time"] = pd.to_datetime(
                    self._event_cache["available_time"]
                )
            else:
                return {"count": 0, "avg_tone": 0}
        
        if self._event_cache.empty:
            return {"count": 0, "avg_tone": 0}
        
        # Filter to time window and availability
        mask = (
            (self._event_cache["event_time"] >= start_ts) &
            (self._event_cache["event_time"] <= end_ts) &
            (self._event_cache["available_time"] <= end_ts)
        )
        filtered = self._event_cache[mask]
        
        return {
            "count": len(filtered),
            "avg_tone": filtered["tone_score"].mean() if len(filtered) > 0 else 0
        }
    
    def build_contract_snapshot_vectorized(
        self,
        contract_id: UUID,
        timestamps: List[datetime]
    ) -> List[Dict]:
        """Build multiple snapshots for a contract efficiently."""
        if not timestamps:
            return []
        
        # Get all prices at once
        min_ts = min(timestamps) - timedelta(days=7)  # Lookback
        max_ts = max(timestamps)
        
        prices_df = self._get_contract_prices(contract_id, min_ts, max_ts)
        
        if prices_df.empty:
            return []
        
        # Ensure timestamps are sorted
        timestamps = sorted(timestamps)
        
        snapshots = []
        for ts in timestamps:
            # Find latest price at or before ts
            available_prices = prices_df[prices_df["available_time"] <= ts]
            latest = available_prices[available_prices["ts"] <= ts].tail(1)
            
            if latest.empty:
                continue
            
            snapshot = {
                "contract_id": contract_id,
                "asof_ts": ts,
                "event_time": ts,
                "available_time": ts,
                "implied_p_yes": latest["price_normalized"].values[0],
            }
            
            # Calculate price features
            window_start = ts - timedelta(hours=24)
            window_prices = available_prices[
                (available_prices["ts"] >= window_start) &
                (available_prices["ts"] <= ts)
            ]
            
            if not window_prices.empty:
                snapshot["price_volatility_24h"] = window_prices["price_normalized"].std()
                snapshot["price_change_24h"] = (
                    window_prices["price_normalized"].iloc[-1] -
                    window_prices["price_normalized"].iloc[0]
                ) if len(window_prices) > 1 else 0
            
            # Get macro panel
            macro_panel = self._get_macro_panel(ts)
            snapshot["macro_panel"] = macro_panel
            
            # Get event stats
            event_stats = self._get_event_stats(window_start, ts)
            snapshot["event_counts_24h"] = event_stats["count"]
            snapshot["event_tone_avg"] = event_stats["avg_tone"]
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def build_snapshots_for_range(
        self,
        contract_ids: Optional[List[UUID]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        frequency: str = "1h",
        checkpoint_dir: Optional[Path] = None
    ) -> int:
        """Build snapshots for multiple contracts with checkpointing."""
        
        if contract_ids is None:
            result = self.db.run_query(
                "SELECT contract_id FROM dim_contract WHERE status = 'active'"
            )
            contract_ids = [UUID(r["contract_id"]) for r in result]
        
        if not start_ts:
            start_ts = datetime.utcnow() - timedelta(days=30)
        if not end_ts:
            end_ts = datetime.utcnow()
        
        # Generate timestamp series
        freq_map = {"1h": "H", "1d": "D", "15min": "15min", "4h": "4H"}
        pandas_freq = freq_map.get(frequency, "H")
        timestamps = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq).tolist()
        
        logger.info(f"Building {len(timestamps)} snapshots for {len(contract_ids)} contracts")
        
        total_snapshots = 0
        
        # Checkpoint setup
        checkpoint_mgr = None
        if checkpoint_dir:
            checkpoint_mgr = CheckpointManager(checkpoint_dir)
            ctx = checkpoint_mgr.checkpoint_context("snapshot_build", resume=True)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()
        
        with ctx:
            if checkpoint_mgr:
                processed = set(ctx.state.get("processed_contracts", []))
                contract_ids = [c for c in contract_ids if str(c) not in processed]
            
            # Process contracts in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.build_contract_snapshot_vectorized,
                        contract_id,
                        timestamps
                    ): contract_id
                    for contract_id in contract_ids
                }
                
                for future in futures:
                    contract_id = futures[future]
                    try:
                        snapshots = future.result()
                        
                        if snapshots:
                            # Batch insert
                            self._save_snapshots_batch(snapshots)
                            total_snapshots += len(snapshots)
                            
                            self.metrics.record_loaded(
                                "snap_contract_features",
                                len(snapshots)
                            )
                        
                        # Update checkpoint
                        if checkpoint_mgr:
                            processed.add(str(contract_id))
                            ctx.update(processed_contracts=list(processed))
                            ctx.save()
                        
                    except Exception as e:
                        logger.error(f"Failed to build snapshots for {contract_id}: {e}")
                        self.metrics.record_error("snapshot_build_failed")
        
        logger.info(f"Built {total_snapshots} total snapshots")
        return total_snapshots
    
    def _save_snapshots_batch(self, snapshots: List[Dict]) -> None:
        """Save snapshots in batch."""
        import json
        
        config = BatchConfig(batch_size=100)
        columns = [
            "contract_id", "asof_ts", "implied_p_yes", "spread",
            "volume_24h", "trade_count_24h", "price_volatility_24h",
            "price_change_24h", "macro_panel", "event_counts_24h",
            "event_tone_avg", "event_time", "available_time"
        ]
        
        with BatchInserter("snap_contract_features", columns, config) as inserter:
            for snapshot in snapshots:
                record = {
                    "contract_id": str(snapshot["contract_id"]),
                    "asof_ts": snapshot["asof_ts"],
                    "implied_p_yes": snapshot.get("implied_p_yes"),
                    "spread": snapshot.get("spread"),
                    "volume_24h": snapshot.get("volume_24h"),
                    "trade_count_24h": snapshot.get("trade_count_24h"),
                    "price_volatility_24h": snapshot.get("price_volatility_24h"),
                    "price_change_24h": snapshot.get("price_change_24h"),
                    "macro_panel": json.dumps(snapshot.get("macro_panel", {})),
                    "event_counts_24h": snapshot.get("event_counts_24h"),
                    "event_tone_avg": snapshot.get("event_tone_avg"),
                    "event_time": snapshot["event_time"],
                    "available_time": snapshot["available_time"]
                }
                inserter.add(record)
    
    def build_feature_enriched_snapshots(
        self,
        contract_id: UUID,
        asof_ts: datetime
    ) -> Optional[Dict]:
        """Build snapshot with advanced features."""
        # Get price history
        prices_df = self._get_contract_prices(
            contract_id,
            asof_ts - timedelta(days=30),
            asof_ts
        )
        
        if prices_df.empty:
            return None
        
        # Calculate technical features
        price_series = prices_df.set_index("ts")["price_normalized"]
        
        features = ContractFeatureEngineer.calculate_price_features(price_series)
        
        # Get latest feature values
        latest = features.iloc[-1].to_dict()
        
        snapshot = {
            "contract_id": contract_id,
            "asof_ts": asof_ts,
            "event_time": asof_ts,
            "available_time": asof_ts,
            **latest
        }
        
        return snapshot


def build_snapshots_enhanced(
    contracts: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "1h",
    max_workers: int = 4
) -> int:
    """CLI-friendly wrapper for enhanced snapshot building."""
    builder = EnhancedSnapshotBuilder(max_workers=max_workers)
    
    contract_ids = [UUID(c) for c in contracts] if contracts else None
    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None
    
    return builder.build_snapshots_for_range(
        contract_ids=contract_ids,
        start_ts=start_ts,
        end_ts=end_ts,
        frequency=freq
    )
