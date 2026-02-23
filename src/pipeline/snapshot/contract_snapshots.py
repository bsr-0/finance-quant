"""Build contract-centric snapshots for training data."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager

logger = logging.getLogger(__name__)


class ContractSnapshotBuilder:
    """Build point-in-time snapshots for prediction market contracts."""
    
    def __init__(self):
        self.db = get_db_manager()
    
    def build_contract_snapshot(
        self,
        contract_id: UUID,
        asof_ts: datetime,
        lookback_windows: Optional[Dict[str, timedelta]] = None
    ) -> Optional[dict]:
        """Build a snapshot for a contract at a specific point in time.
        
        Args:
            contract_id: The contract UUID
            asof_ts: Snapshot timestamp (point-in-time)
            lookback_windows: Dict of window names to timedelta for aggregations
        
        Returns:
            Dictionary with snapshot features or None if no data
        """
        if lookback_windows is None:
            lookback_windows = {
                "1h": timedelta(hours=1),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7)
            }
        
        snapshot = {
            "contract_id": contract_id,
            "asof_ts": asof_ts,
            "event_time": asof_ts,
            "available_time": asof_ts
        }
        
        # 1. Get latest price at or before asof_ts
        price_data = self._get_latest_price(contract_id, asof_ts)
        if price_data:
            snapshot["implied_p_yes"] = price_data.get("price_normalized")
        else:
            snapshot["implied_p_yes"] = None
        
        # 2. Get orderbook snapshot
        ob_data = self._get_orderbook_snapshot(contract_id, asof_ts)
        if ob_data:
            snapshot["spread"] = ob_data.get("spread")
            snapshot["depth_best_bid"] = self._calculate_depth(ob_data.get("bids", []))
            snapshot["depth_best_ask"] = self._calculate_depth(ob_data.get("asks", []))
        else:
            snapshot["spread"] = None
            snapshot["depth_best_bid"] = None
            snapshot["depth_best_ask"] = None
        
        # 3. Aggregate trades in lookback windows
        for window_name, window_delta in lookback_windows.items():
            window_start = asof_ts - window_delta
            trade_stats = self._get_trade_stats(contract_id, window_start, asof_ts)
            
            if window_name == "24h":
                snapshot["volume_24h"] = trade_stats.get("total_volume")
                snapshot["trade_count_24h"] = trade_stats.get("trade_count")
                snapshot["price_volatility_24h"] = trade_stats.get("price_std")
        
        # 4. Get macro panel (most recent values at asof_ts)
        macro_panel = self._get_macro_panel(asof_ts)
        snapshot["macro_panel"] = macro_panel
        
        # 5. Aggregate world events/news in lookback windows
        event_stats = self._get_event_stats(asof_ts - lookback_windows["24h"], asof_ts)
        snapshot["event_counts_24h"] = event_stats.get("count")
        snapshot["event_tone_avg"] = event_stats.get("avg_tone")
        
        # News counts by window
        news_counts = {}
        for window_name, window_delta in lookback_windows.items():
            window_start = asof_ts - window_delta
            news_count = self._get_news_count(window_start, asof_ts)
            news_counts[window_name] = news_count
        snapshot["news_counts"] = news_counts
        
        return snapshot
    
    def _get_latest_price(self, contract_id: UUID, asof_ts: datetime) -> Optional[dict]:
        """Get the most recent price at or before asof_ts."""
        query = """
            SELECT price_normalized, price_raw, ts
            FROM cur_contract_prices
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
            ORDER BY ts DESC
            LIMIT 1
        """
        result = self.db.run_query(query, {
            "contract_id": str(contract_id),
            "asof_ts": asof_ts
        })
        return result[0] if result else None
    
    def _get_orderbook_snapshot(self, contract_id: UUID, asof_ts: datetime) -> Optional[dict]:
        """Get the most recent orderbook at or before asof_ts."""
        query = """
            SELECT spread, bids, asks
            FROM cur_contract_orderbook_snapshots
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
            ORDER BY ts DESC
            LIMIT 1
        """
        result = self.db.run_query(query, {
            "contract_id": str(contract_id),
            "asof_ts": asof_ts
        })
        return result[0] if result else None
    
    def _calculate_depth(self, orders: List) -> Optional[float]:
        """Calculate total depth from order book entries."""
        if not orders:
            return None
        try:
            return sum(float(order[1]) for order in orders if len(order) >= 2)
        except (TypeError, IndexError):
            return None
    
    def _get_trade_stats(
        self, 
        contract_id: UUID, 
        start_ts: datetime, 
        end_ts: datetime
    ) -> dict:
        """Get trade statistics for a time window."""
        query = """
            SELECT 
                COUNT(*) as trade_count,
                COALESCE(SUM(size), 0) as total_volume,
                COALESCE(STDDEV(price), 0) as price_std
            FROM cur_contract_trades
            WHERE contract_id = :contract_id
              AND available_time <= :end_ts
              AND ts BETWEEN :start_ts AND :end_ts
        """
        result = self.db.run_query(query, {
            "contract_id": str(contract_id),
            "start_ts": start_ts,
            "end_ts": end_ts
        })
        return result[0] if result else {"trade_count": 0, "total_volume": 0, "price_std": 0}
    
    def _get_macro_panel(self, asof_ts: datetime) -> dict:
        """Get most recent macro observations at asof_ts."""
        query = """
            SELECT DISTINCT ON (s.provider_series_code)
                s.provider_series_code as series_code,
                o.value,
                o.period_end
            FROM cur_macro_observations o
            JOIN dim_macro_series s ON o.series_id = s.series_id
            WHERE o.available_time <= :asof_ts
            ORDER BY s.provider_series_code, o.period_end DESC
        """
        results = self.db.run_query(query, {"asof_ts": asof_ts})
        return {r["series_code"]: r["value"] for r in results}
    
    def _get_event_stats(self, start_ts: datetime, end_ts: datetime) -> dict:
        """Get event statistics for a time window."""
        query = """
            SELECT 
                COUNT(*) as count,
                COALESCE(AVG(tone_score), 0) as avg_tone
            FROM cur_world_events
            WHERE available_time <= :end_ts
              AND event_time BETWEEN :start_ts AND :end_ts
        """
        result = self.db.run_query(query, {
            "start_ts": start_ts,
            "end_ts": end_ts
        })
        return result[0] if result else {"count": 0, "avg_tone": 0}
    
    def _get_news_count(self, start_ts: datetime, end_ts: datetime) -> int:
        """Get news count for a time window."""
        query = """
            SELECT COUNT(*) as cnt
            FROM cur_news_items
            WHERE available_time <= :end_ts
              AND event_time BETWEEN :start_ts AND :end_ts
        """
        result = self.db.run_query(query, {
            "start_ts": start_ts,
            "end_ts": end_ts
        })
        return result[0]["cnt"] if result else 0
    
    def build_snapshots_for_range(
        self,
        contract_ids: Optional[List[UUID]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        frequency: str = "1h"
    ) -> int:
        """Build snapshots for multiple contracts over a time range.
        
        Args:
            contract_ids: List of contract IDs (None = all active contracts)
            start_ts: Start of snapshot range
            end_ts: End of snapshot range
            frequency: Snapshot frequency (e.g., '1h', '1d')
        
        Returns:
            Number of snapshots created
        """
        if contract_ids is None:
            # Get all active contracts
            result = self.db.run_query(
                "SELECT contract_id FROM dim_contract WHERE status = 'active'"
            )
            contract_ids = [UUID(r["contract_id"]) for r in result]
        
        if not start_ts:
            start_ts = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_ts:
            end_ts = datetime.now(timezone.utc)
        
        # Generate timestamp series
        freq_map = {"1h": "H", "1d": "D", "15min": "15min"}
        pandas_freq = freq_map.get(frequency, "H")
        
        timestamps = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)
        
        total_snapshots = 0
        
        for contract_id in contract_ids:
            logger.info(f"Building snapshots for contract {contract_id}")
            
            for ts in timestamps:
                snapshot = self.build_contract_snapshot(contract_id, ts)
                if snapshot and snapshot["implied_p_yes"] is not None:
                    self._save_snapshot(snapshot)
                    total_snapshots += 1
        
        logger.info(f"Built {total_snapshots} total snapshots")
        return total_snapshots
    
    def _save_snapshot(self, snapshot: dict) -> None:
        """Save snapshot to database."""
        with self.db.engine.connect() as conn:
            insert = text("""
                INSERT INTO snap_contract_features 
                (contract_id, asof_ts, implied_p_yes, spread, depth_best_bid, depth_best_ask,
                 volume_24h, trade_count_24h, price_volatility_24h, macro_panel, news_counts,
                 event_counts_24h, event_tone_avg, event_time, available_time)
                VALUES (:contract_id, :asof_ts, :implied_p_yes, :spread, :depth_best_bid,
                        :depth_best_ask, :volume_24h, :trade_count_24h, :price_volatility_24h,
                        :macro_panel, :news_counts, :event_counts_24h, :event_tone_avg,
                        :event_time, :available_time)
                ON CONFLICT (contract_id, asof_ts) DO UPDATE SET
                    implied_p_yes = EXCLUDED.implied_p_yes,
                    spread = EXCLUDED.spread,
                    volume_24h = EXCLUDED.volume_24h,
                    updated_at = NOW()
            """)
            
            import json
            conn.execute(insert, {
                "contract_id": str(snapshot["contract_id"]),
                "asof_ts": snapshot["asof_ts"],
                "implied_p_yes": snapshot["implied_p_yes"],
                "spread": snapshot["spread"],
                "depth_best_bid": snapshot["depth_best_bid"],
                "depth_best_ask": snapshot["depth_best_ask"],
                "volume_24h": snapshot["volume_24h"],
                "trade_count_24h": snapshot["trade_count_24h"],
                "price_volatility_24h": snapshot["price_volatility_24h"],
                "macro_panel": json.dumps(snapshot["macro_panel"]),
                "news_counts": json.dumps(snapshot["news_counts"]),
                "event_counts_24h": snapshot["event_counts_24h"],
                "event_tone_avg": snapshot["event_tone_avg"],
                "event_time": snapshot["event_time"],
                "available_time": snapshot["available_time"]
            })
            conn.commit()


def build_snapshots(
    contracts: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "1h"
) -> int:
    """CLI-friendly wrapper for building snapshots."""
    builder = ContractSnapshotBuilder()
    
    contract_ids = [UUID(c) for c in contracts] if contracts else None
    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None
    
    return builder.build_snapshots_for_range(
        contract_ids=contract_ids,
        start_ts=start_ts,
        end_ts=end_ts,
        frequency=freq
    )
