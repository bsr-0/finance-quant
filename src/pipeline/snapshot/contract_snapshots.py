"""Build contract-centric snapshots for training data."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.features import robust_stats
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class ContractSnapshotBuilder:
    """Build point-in-time snapshots for prediction market contracts."""

    def __init__(self):
        self.db = get_db_manager()
        self.settings = get_settings()

    def build_contract_snapshot(
        self,
        contract_id: UUID,
        asof_ts: datetime,
        lookback_windows: dict[str, timedelta] | None = None,
    ) -> dict | None:
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
                "7d": timedelta(days=7),
            }

        snapshot = {
            "contract_id": contract_id,
            "asof_ts": asof_ts,
            "event_time": asof_ts,
            "available_time": asof_ts,
            "volume_24h": None,
            "trade_count_24h": None,
            "price_volatility_24h": None,
            "price_volatility_24h_robust": None,
            "trade_outlier_pct": None,
            "trade_imbalance": None,
            "avg_trade_size": None,
            "trade_size_std": None,
            "price_staleness_hours": None,
            "macro_staleness_days": None,
            "data_quality_score": None,
        }

        # 1. Get latest price at or before asof_ts
        price_data = self._get_latest_price(contract_id, asof_ts)
        if price_data:
            snapshot["implied_p_yes"] = price_data.get("price_normalized")
            snapshot["price_staleness_hours"] = self._staleness_hours(asof_ts, price_data.get("ts"))
        else:
            snapshot["implied_p_yes"] = None
            snapshot["price_staleness_hours"] = None

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
                snapshot["price_volatility_24h_robust"] = trade_stats.get("price_std_robust")
                snapshot["trade_outlier_pct"] = trade_stats.get("outlier_pct")
                snapshot["trade_imbalance"] = trade_stats.get("trade_imbalance")
                snapshot["avg_trade_size"] = trade_stats.get("avg_trade_size")
                snapshot["trade_size_std"] = trade_stats.get("trade_size_std")

        # 4. Get macro panel (most recent values at asof_ts)
        macro_panel, macro_staleness_days = self._get_macro_panel(asof_ts)
        snapshot["macro_panel"] = macro_panel
        snapshot["macro_staleness_days"] = macro_staleness_days

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

        # 6. Staleness metrics
        staleness = self._get_staleness_metrics(contract_id, asof_ts)
        snapshot["price_staleness_hours"] = staleness["price_staleness_hours"]
        snapshot["macro_staleness_days"] = staleness["macro_staleness_days"]
        snapshot["last_price_ts"] = staleness["last_price_ts"]
        if (
            staleness["price_staleness_hours"] is not None
            and staleness["price_staleness_hours"] > 24
        ):
            logger.warning(
                f"Price data for {contract_id} is "
                f"{staleness['price_staleness_hours']:.1f} hours stale"
            )

        # 7. Outlier detection
        outliers = self._detect_price_outliers(contract_id, asof_ts)
        snapshot["has_price_outliers"] = outliers["has_price_outliers"]
        snapshot["outlier_score"] = outliers["outlier_score"]

        # 8. Microstructure features (24h window)
        micro = self._calculate_microstructure_features(
            contract_id,
            asof_ts - lookback_windows["24h"],
            asof_ts,
        )
        snapshot["micro_trade_imbalance"] = micro["trade_imbalance"]
        snapshot["micro_buy_sell_ratio"] = micro["buy_sell_ratio"]
        snapshot["micro_avg_trade_size"] = micro["avg_trade_size"]

        # 9. Data quality score
        snapshot["data_quality_score"] = self._calculate_data_quality_score(
            price_staleness_hours=staleness["price_staleness_hours"],
            macro_staleness_days=staleness["macro_staleness_days"],
            has_price_outliers=outliers["has_price_outliers"],
            microstructure=micro,
        )

        return snapshot

    def _get_latest_price(self, contract_id: UUID, asof_ts: datetime) -> dict | None:
        """Get the most recent price at or before asof_ts."""
        query = """
            SELECT price_normalized, price_raw, ts
            FROM cur_contract_prices
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
            ORDER BY ts DESC
            LIMIT 1
        """
        result = self.db.run_query(query, {"contract_id": str(contract_id), "asof_ts": asof_ts})
        return result[0] if result else None

    def _get_orderbook_snapshot(self, contract_id: UUID, asof_ts: datetime) -> dict | None:
        """Get the most recent orderbook at or before asof_ts."""
        query = """
            SELECT spread, bids, asks
            FROM cur_contract_orderbook_snapshots
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
            ORDER BY ts DESC
            LIMIT 1
        """
        result = self.db.run_query(query, {"contract_id": str(contract_id), "asof_ts": asof_ts})
        return result[0] if result else None

    def _calculate_depth(self, orders: list) -> float | None:
        """Calculate total depth from order book entries."""
        if not orders:
            return None
        try:
            return sum(float(order[1]) for order in orders if len(order) >= 2)
        except (TypeError, IndexError):
            return None

    def _get_trade_stats(self, contract_id: UUID, start_ts: datetime, end_ts: datetime) -> dict:
        """Get trade statistics for a time window."""
        query = """
            SELECT price, size, side
            FROM cur_contract_trades
            WHERE contract_id = :contract_id
              AND available_time <= :end_ts
              AND ts BETWEEN :start_ts AND :end_ts
        """
        rows = self.db.run_query(
            query, {"contract_id": str(contract_id), "start_ts": start_ts, "end_ts": end_ts}
        )
        if not rows:
            return {
                "trade_count": 0,
                "total_volume": 0,
                "price_std": 0,
                "price_std_robust": 0,
                "outlier_pct": 0,
                "trade_imbalance": None,
                "avg_trade_size": None,
                "trade_size_std": None,
            }

        df = pd.DataFrame(rows)
        prices = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce")
        sizes = pd.to_numeric(df.get("size", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        trade_count = int(prices.notna().sum())
        total_volume = float(sizes.sum())
        price_std = float(prices.std()) if trade_count > 1 else 0.0

        robust_mad = robust_stats.mad(prices.dropna()) if trade_count > 0 else 0.0
        price_std_robust = float(1.4826 * robust_mad) if robust_mad else 0.0

        outlier_pct = 0.0
        if robust_mad and trade_count > 0:
            z = 0.6745 * (prices - prices.median()) / robust_mad
            threshold = self.settings.historical_fixes.trade_outlier_mad_threshold
            outlier_pct = float((z.abs() > threshold).mean())

        trade_imbalance = None
        if "side" in df.columns:
            side = df["side"].astype(str).str.lower()
            buy_volume = float(sizes[side == "buy"].sum())
            sell_volume = float(sizes[side == "sell"].sum())
            denom = buy_volume + sell_volume
            if denom > 0:
                trade_imbalance = (buy_volume - sell_volume) / denom

        avg_trade_size = float(total_volume / trade_count) if trade_count > 0 else None
        trade_size_std = float(sizes.std()) if trade_count > 1 else None

        return {
            "trade_count": trade_count,
            "total_volume": total_volume,
            "price_std": price_std,
            "price_std_robust": price_std_robust,
            "outlier_pct": outlier_pct,
            "trade_imbalance": trade_imbalance,
            "avg_trade_size": avg_trade_size,
            "trade_size_std": trade_size_std,
        }

    def _get_macro_panel(self, asof_ts: datetime) -> tuple[dict, float | None]:
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
        panel: dict[str, float | None] = {}
        staleness_days: float | None = None
        for row in results:
            panel[row["series_code"]] = float(row["value"]) if row["value"] is not None else None
            period_end = row.get("period_end")
            if period_end:
                period_dt = pd.to_datetime(period_end).date()
                delta_days = (asof_ts.date() - period_dt).days
                if staleness_days is None or delta_days > staleness_days:
                    staleness_days = float(delta_days)
        return panel, staleness_days

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
        result = self.db.run_query(query, {"start_ts": start_ts, "end_ts": end_ts})
        return result[0] if result else {"count": 0, "avg_tone": 0}

    def _get_news_count(self, start_ts: datetime, end_ts: datetime) -> int:
        """Get news count for a time window."""
        query = """
            SELECT COUNT(*) as cnt
            FROM cur_news_items
            WHERE available_time <= :end_ts
              AND event_time BETWEEN :start_ts AND :end_ts
        """
        result = self.db.run_query(query, {"start_ts": start_ts, "end_ts": end_ts})
        return result[0]["cnt"] if result else 0

    def _to_datetime(self, value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            parsed = pd.to_datetime(value)
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
        except Exception:
            return None

    def _staleness_hours(self, asof_ts: datetime, value: object) -> float | None:
        dt = self._to_datetime(value)
        if dt is None:
            return None
        return max(0.0, (asof_ts - dt).total_seconds() / 3600.0)

    def _get_staleness_metrics(self, contract_id: UUID, asof_ts: datetime) -> dict:
        """Return staleness information for price and macro data."""
        # Price staleness
        price_result = self.db.run_query(
            """
            SELECT ts
            FROM cur_contract_prices
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
            ORDER BY ts DESC
            LIMIT 1
            """,
            {"contract_id": str(contract_id), "asof_ts": asof_ts},
        )

        price_staleness_hours: float | None = None
        last_price_ts: datetime | None = None
        if price_result:
            last_price_ts = price_result[0]["ts"]
            if isinstance(last_price_ts, str):
                last_price_ts = datetime.fromisoformat(last_price_ts.replace("Z", "+00:00"))
            asof_aware = asof_ts.replace(tzinfo=UTC) if asof_ts.tzinfo is None else asof_ts
            lp_aware = (
                last_price_ts.replace(tzinfo=UTC) if last_price_ts.tzinfo is None else last_price_ts
            )
            price_staleness_hours = (asof_aware - lp_aware).total_seconds() / 3600

        # Macro staleness
        macro_result = self.db.run_query(
            """
            SELECT MAX(period_end) as latest_period
            FROM cur_macro_observations
            WHERE available_time <= :asof_ts
            """,
            {"asof_ts": asof_ts},
        )

        macro_staleness_days: float | None = None
        if macro_result and macro_result[0]["latest_period"]:
            latest_period = macro_result[0]["latest_period"]
            if hasattr(latest_period, "date"):
                latest_period = latest_period.date()
            asof_date = asof_ts.date() if hasattr(asof_ts, "date") else asof_ts
            macro_staleness_days = float((asof_date - latest_period).days)

        return {
            "price_staleness_hours": price_staleness_hours,
            "macro_staleness_days": macro_staleness_days,
            "last_price_ts": last_price_ts,
        }

    def _detect_price_outliers(
        self, contract_id: UUID, asof_ts: datetime, window_days: int = 30
    ) -> dict:
        """Detect whether the latest price is an outlier using the MAD method.

        Uses the Modified Z-score (Iglewicz & Hoaglin, 1993).  A score above
        3.5 is considered an outlier.
        """
        window_start = asof_ts - timedelta(days=window_days)
        result = self.db.run_query(
            """
            SELECT price_normalized
            FROM cur_contract_prices
            WHERE contract_id = :contract_id
              AND available_time <= :asof_ts
              AND ts >= :window_start
            ORDER BY ts
            """,
            {
                "contract_id": str(contract_id),
                "asof_ts": asof_ts,
                "window_start": window_start,
            },
        )

        if not result or len(result) < 10:
            return {"has_price_outliers": False, "outlier_score": 0.0}

        prices = [r["price_normalized"] for r in result if r["price_normalized"] is not None]
        if not prices:
            return {"has_price_outliers": False, "outlier_score": 0.0}

        prices_arr = np.array(prices, dtype=float)
        median = float(np.median(prices_arr))
        mad = float(np.median(np.abs(prices_arr - median)))

        if mad == 0:
            return {"has_price_outliers": False, "outlier_score": 0.0}

        modified_z_scores = 0.6745 * np.abs(prices_arr - median) / mad
        latest_score = float(modified_z_scores[-1])
        has_outliers = latest_score > 3.5

        return {"has_price_outliers": has_outliers, "outlier_score": latest_score}

    def _calculate_microstructure_features(
        self, contract_id: UUID, start_ts: datetime, end_ts: datetime
    ) -> dict:
        """Calculate microstructure features from trade data.

        Returns trade imbalance, buy/sell ratio, and average trade size.
        """
        result = self.db.run_query(
            """
            SELECT
                SUM(CASE WHEN side = 'buy'  THEN size ELSE 0 END) AS buy_volume,
                SUM(CASE WHEN side = 'sell' THEN size ELSE 0 END) AS sell_volume,
                AVG(size)                                          AS avg_trade_size,
                COUNT(*)                                           AS trade_count
            FROM cur_contract_trades
            WHERE contract_id = :contract_id
              AND available_time <= :end_ts
              AND ts BETWEEN :start_ts AND :end_ts
            """,
            {
                "contract_id": str(contract_id),
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        )

        if not result or not result[0]["trade_count"]:
            return {"trade_imbalance": None, "buy_sell_ratio": None, "avg_trade_size": None}

        row = result[0]
        buy_vol = float(row["buy_volume"] or 0)
        sell_vol = float(row["sell_volume"] or 0)
        total_vol = buy_vol + sell_vol

        trade_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else None
        buy_sell_ratio = buy_vol / sell_vol if sell_vol > 0 else None

        return {
            "trade_imbalance": trade_imbalance,
            "buy_sell_ratio": buy_sell_ratio,
            "avg_trade_size": float(row["avg_trade_size"]) if row["avg_trade_size"] else None,
        }

    def _calculate_data_quality_score(
        self,
        price_staleness_hours: float | None,
        macro_staleness_days: float | None,
        has_price_outliers: bool,
        microstructure: dict,
    ) -> float:
        """Calculate a composite data quality score (0-100).

        Deductions:
        - Up to -30 for price staleness > 1 hour
        - Up to -20 for macro staleness > 1 day
        - -20 if price outliers detected
        - -10 if microstructure features are unavailable
        """
        score = 100.0

        if price_staleness_hours is not None and price_staleness_hours > 1:
            score -= min(30.0, (price_staleness_hours - 1) * 5)

        if macro_staleness_days is not None and macro_staleness_days > 1:
            score -= min(20.0, (macro_staleness_days - 1) * 2)

        if has_price_outliers:
            score -= 20.0

        if microstructure.get("trade_imbalance") is None:
            score -= 10.0

        return max(0.0, score)

    def build_snapshots_for_range(
        self,
        contract_ids: list[UUID] | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        frequency: str = "1h",
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
            start_ts = datetime.now(UTC) - timedelta(days=30)
        if not end_ts:
            end_ts = datetime.now(UTC)

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
                 volume_24h, trade_count_24h, price_volatility_24h, price_volatility_24h_robust,
                 trade_outlier_pct, trade_imbalance, avg_trade_size, trade_size_std,
                 price_staleness_hours, macro_staleness_days, data_quality_score,
                 macro_panel, news_counts, event_counts_24h,
                 event_tone_avg, event_time, available_time,
                 last_price_ts, has_price_outliers, outlier_score,
                 micro_trade_imbalance, micro_buy_sell_ratio)
                VALUES (:contract_id, :asof_ts, :implied_p_yes, :spread, :depth_best_bid,
                        :depth_best_ask, :volume_24h, :trade_count_24h, :price_volatility_24h,
                        :price_volatility_24h_robust, :trade_outlier_pct, :trade_imbalance,
                        :avg_trade_size, :trade_size_std, :price_staleness_hours,
                        :macro_staleness_days, :data_quality_score, :macro_panel, :news_counts,
                        :event_counts_24h, :event_tone_avg, :event_time, :available_time,
                        :last_price_ts, :has_price_outliers, :outlier_score,
                        :micro_trade_imbalance, :micro_buy_sell_ratio)
                ON CONFLICT (contract_id, asof_ts) DO UPDATE SET
                    implied_p_yes = EXCLUDED.implied_p_yes,
                    spread = EXCLUDED.spread,
                    volume_24h = EXCLUDED.volume_24h,
                    price_volatility_24h = EXCLUDED.price_volatility_24h,
                    price_volatility_24h_robust = EXCLUDED.price_volatility_24h_robust,
                    trade_outlier_pct = EXCLUDED.trade_outlier_pct,
                    trade_imbalance = EXCLUDED.trade_imbalance,
                    avg_trade_size = EXCLUDED.avg_trade_size,
                    trade_size_std = EXCLUDED.trade_size_std,
                    price_staleness_hours = EXCLUDED.price_staleness_hours,
                    macro_staleness_days = EXCLUDED.macro_staleness_days,
                    data_quality_score = EXCLUDED.data_quality_score,
                    has_price_outliers = EXCLUDED.has_price_outliers,
                    outlier_score = EXCLUDED.outlier_score,
                    micro_trade_imbalance = EXCLUDED.micro_trade_imbalance,
                    micro_buy_sell_ratio = EXCLUDED.micro_buy_sell_ratio,
                    updated_at = NOW()
            """)

            conn.execute(
                insert,
                {
                    "contract_id": str(snapshot["contract_id"]),
                    "asof_ts": snapshot["asof_ts"],
                    "implied_p_yes": snapshot["implied_p_yes"],
                    "spread": snapshot["spread"],
                    "depth_best_bid": snapshot["depth_best_bid"],
                    "depth_best_ask": snapshot["depth_best_ask"],
                    "volume_24h": snapshot["volume_24h"],
                    "trade_count_24h": snapshot["trade_count_24h"],
                    "price_volatility_24h": snapshot["price_volatility_24h"],
                    "price_volatility_24h_robust": snapshot.get("price_volatility_24h_robust"),
                    "trade_outlier_pct": snapshot.get("trade_outlier_pct"),
                    "trade_imbalance": snapshot.get("trade_imbalance"),
                    "avg_trade_size": snapshot.get("avg_trade_size"),
                    "trade_size_std": snapshot.get("trade_size_std"),
                    "price_staleness_hours": snapshot.get("price_staleness_hours"),
                    "macro_staleness_days": snapshot.get("macro_staleness_days"),
                    "data_quality_score": snapshot.get("data_quality_score", 100.0),
                    "macro_panel": json.dumps(snapshot["macro_panel"], default=str),
                    "news_counts": json.dumps(snapshot["news_counts"], default=str),
                    "event_counts_24h": snapshot["event_counts_24h"],
                    "event_tone_avg": snapshot["event_tone_avg"],
                    "event_time": snapshot["event_time"],
                    "available_time": snapshot["available_time"],
                    "last_price_ts": snapshot.get("last_price_ts"),
                    "has_price_outliers": snapshot.get("has_price_outliers", False),
                    "outlier_score": snapshot.get("outlier_score"),
                    "micro_trade_imbalance": snapshot.get("micro_trade_imbalance"),
                    "micro_buy_sell_ratio": snapshot.get("micro_buy_sell_ratio"),
                },
            )
            conn.commit()


def build_snapshots(
    contracts: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str = "1h",
) -> int:
    """CLI-friendly wrapper for building snapshots."""
    builder = ContractSnapshotBuilder()

    contract_ids = [UUID(c) for c in contracts] if contracts else None
    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None

    return builder.build_snapshots_for_range(
        contract_ids=contract_ids, start_ts=start_ts, end_ts=end_ts, frequency=freq
    )
