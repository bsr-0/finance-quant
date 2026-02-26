"""Build symbol-centric snapshots for equity training data."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.features import robust_stats

logger = logging.getLogger(__name__)


class SymbolSnapshotBuilder:
    """Build point-in-time snapshots for equity symbols."""

    def __init__(self):
        self.db = get_db_manager()

    def build_symbol_snapshot(
        self,
        symbol_id: UUID,
        asof_ts: datetime,
        lookback_days: int = 60,
    ) -> dict | None:
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.replace(tzinfo=UTC)
        else:
            asof_ts = asof_ts.astimezone(UTC)
        snapshot = {
            "symbol_id": symbol_id,
            "asof_ts": asof_ts,
            "event_time": asof_ts,
            "available_time": asof_ts,
            "price_latest": None,
            "price_change_1d": None,
            "price_change_7d": None,
            "volume_avg_20d": None,
            "volatility_20d": None,
            "macro_panel": {},
            "news_counts": {},
        }

        price_df = self._get_price_series(symbol_id, asof_ts, lookback_days)
        if price_df.empty:
            return snapshot

        price_df = price_df.sort_values("date")
        prices = price_df["price"].astype(float)
        volumes = price_df["volume"].astype(float)

        snapshot["price_latest"] = float(prices.iloc[-1])

        returns = prices.pct_change().dropna()
        if not returns.empty:
            winsor_returns = robust_stats.winsorize(returns, 0.01, 0.99)
            snapshot["price_change_1d"] = float(winsor_returns.iloc[-1])

            if len(prices) >= 8:
                raw_7d = (prices.iloc[-1] / prices.iloc[-8]) - 1
                lo = winsor_returns.quantile(0.01)
                hi = winsor_returns.quantile(0.99)
                snapshot["price_change_7d"] = float(min(max(raw_7d, lo), hi))

            recent_returns = winsor_returns.tail(20)
            if not recent_returns.empty:
                mad = robust_stats.mad(recent_returns)
                snapshot["volatility_20d"] = float(1.4826 * mad) if mad else 0.0

        if not volumes.empty:
            recent_volume = volumes.tail(20)
            winsor_vol = robust_stats.winsorize(recent_volume, 0.01, 0.99)
            snapshot["volume_avg_20d"] = float(winsor_vol.mean())

        snapshot["macro_panel"] = self._get_macro_panel(asof_ts)
        snapshot["news_counts"] = self._get_news_counts(asof_ts)

        return snapshot

    def _price_table(self) -> str:
        if self.db.table_exists("cur_prices_adjusted_daily"):
            return "cur_prices_adjusted_daily"
        return "cur_prices_ohlcv_daily"

    def _get_price_series(
        self, symbol_id: UUID, asof_ts: datetime, lookback_days: int
    ) -> pd.DataFrame:
        table = self._price_table()
        asof_date = asof_ts.date()
        min_date = asof_date - timedelta(days=lookback_days)
        price_col = "adj_close" if table == "cur_prices_adjusted_daily" else "close"
        volume_col = "adj_volume" if table == "cur_prices_adjusted_daily" else "volume"

        query = f"""
            SELECT date, {price_col} AS price, {volume_col} AS volume
            FROM {table}
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
              AND date BETWEEN :min_date AND :asof_date
            ORDER BY date DESC
        """
        rows = self.db.run_query(
            query,
            {
                "symbol_id": str(symbol_id),
                "asof_ts": asof_ts,
                "min_date": min_date,
                "asof_date": asof_date,
            },
        )
        return pd.DataFrame(rows)

    def _get_macro_panel(self, asof_ts: datetime) -> dict:
        if not self.db.table_exists("cur_macro_observations"):
            return {}
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

    def _get_news_counts(self, asof_ts: datetime) -> dict:
        windows = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}
        counts = {}
        if not self.db.table_exists("cur_news_items"):
            return dict.fromkeys(windows, 0)
        for name, delta in windows.items():
            start_ts = asof_ts - delta
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_news_items
                WHERE available_time <= :end_ts
                  AND event_time BETWEEN :start_ts AND :end_ts
            """
            result = self.db.run_query(query, {"start_ts": start_ts, "end_ts": asof_ts})
            counts[name] = result[0]["cnt"] if result else 0
        return counts

    def build_snapshots_for_range(
        self,
        symbol_ids: list[UUID] | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        frequency: str = "1d",
    ) -> int:
        if symbol_ids is None:
            result = self.db.run_query("SELECT symbol_id FROM dim_symbol")
            symbol_ids = [UUID(r["symbol_id"]) for r in result]

        if not start_ts:
            start_ts = datetime.now(UTC) - timedelta(days=90)
        if not end_ts:
            end_ts = datetime.now(UTC)

        freq_map = {"1h": "H", "1d": "D", "15min": "15min"}
        pandas_freq = freq_map.get(frequency, "D")
        timestamps = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)

        total = 0
        for symbol_id in symbol_ids:
            logger.info(f"Building symbol snapshots for {symbol_id}")
            for ts in timestamps:
                snapshot = self.build_symbol_snapshot(symbol_id, ts.to_pydatetime())
                if snapshot and snapshot["price_latest"] is not None:
                    self._save_snapshot(snapshot)
                    total += 1
        logger.info(f"Built {total} symbol snapshots")
        return total

    def _save_snapshot(self, snapshot: dict) -> None:
        with self.db.engine.connect() as conn:
            insert = text(
                """
                INSERT INTO snap_symbol_features
                    (symbol_id, asof_ts, price_latest, price_change_1d, price_change_7d,
                     volume_avg_20d, volatility_20d, macro_panel, news_counts,
                     event_time, available_time)
                VALUES
                    (:symbol_id, :asof_ts, :price_latest, :price_change_1d, :price_change_7d,
                     :volume_avg_20d, :volatility_20d, :macro_panel, :news_counts,
                     :event_time, :available_time)
                ON CONFLICT (symbol_id, asof_ts) DO UPDATE SET
                    price_latest = EXCLUDED.price_latest,
                    price_change_1d = EXCLUDED.price_change_1d,
                    price_change_7d = EXCLUDED.price_change_7d,
                    volume_avg_20d = EXCLUDED.volume_avg_20d,
                    volatility_20d = EXCLUDED.volatility_20d,
                    macro_panel = EXCLUDED.macro_panel,
                    news_counts = EXCLUDED.news_counts,
                    updated_at = NOW()
            """
            )
            import json

            conn.execute(
                insert,
                {
                    "symbol_id": str(snapshot["symbol_id"]),
                    "asof_ts": snapshot["asof_ts"],
                    "price_latest": snapshot["price_latest"],
                    "price_change_1d": snapshot["price_change_1d"],
                    "price_change_7d": snapshot["price_change_7d"],
                    "volume_avg_20d": snapshot["volume_avg_20d"],
                    "volatility_20d": snapshot["volatility_20d"],
                    "macro_panel": json.dumps(snapshot["macro_panel"]),
                    "news_counts": json.dumps(snapshot["news_counts"]),
                    "event_time": snapshot["event_time"],
                    "available_time": snapshot["available_time"],
                },
            )
            conn.commit()


def build_symbol_snapshots(
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str = "1d",
) -> int:
    builder = SymbolSnapshotBuilder()
    symbol_ids = [UUID(s) for s in symbols] if symbols else None
    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None
    return builder.build_snapshots_for_range(
        symbol_ids=symbol_ids, start_ts=start_ts, end_ts=end_ts, frequency=freq
    )
