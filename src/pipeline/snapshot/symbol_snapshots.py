"""Build symbol-centric snapshots for equity training data."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
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
        asof_ts = asof_ts.replace(tzinfo=UTC) if asof_ts.tzinfo is None else asof_ts.astimezone(UTC)
        snapshot: dict[str, Any] = {
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
            # New data source features
            "pe_ratio": None,
            "pb_ratio": None,
            "debt_to_equity": None,
            "roe": None,
            "insider_net_shares_90d": None,
            "insider_buy_count_90d": None,
            "institutional_holders_count": None,
            "iv_30d": None,
            "put_call_volume_ratio": None,
            "skew_25d": None,
            "days_to_next_earnings": None,
            "last_eps_surprise_pct": None,
            "short_interest_ratio": None,
            "cot_noncommercial_net": None,
            "cot_commercial_net": None,
            "cot_noncommercial_pct_oi": None,
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

        # New data source features
        price_latest: float | None = snapshot.get("price_latest")  # type: ignore[assignment]
        fundamentals = self._get_fundamentals(symbol_id, asof_ts, price_latest)
        snapshot.update(fundamentals)

        insider = self._get_insider_activity(symbol_id, asof_ts)
        snapshot.update(insider)

        inst = self._get_institutional_holdings(symbol_id, asof_ts)
        snapshot.update(inst)

        options = self._get_options_iv(symbol_id, asof_ts)
        snapshot.update(options)

        earnings = self._get_earnings_features(symbol_id, asof_ts)
        snapshot.update(earnings)

        short = self._get_short_interest(symbol_id, asof_ts)
        snapshot.update(short)

        cot = self._get_cot_positioning(symbol_id, asof_ts)
        snapshot.update(cot)

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
        return {
            r["series_code"]: float(r["value"]) if r["value"] is not None else None for r in results
        }

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

    def _get_fundamentals(self, symbol_id: UUID, asof_ts: datetime, price: float | None) -> dict:
        """Get latest fundamental ratios as of asof_ts."""
        result: dict[str, float | None] = {
            "pe_ratio": None,
            "pb_ratio": None,
            "debt_to_equity": None,
            "roe": None,
        }
        if not self.db.table_exists("cur_fundamentals_quarterly"):
            return result

        rows = self.db.run_query(
            """
            SELECT metric_name, metric_value
            FROM cur_fundamentals_quarterly
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
              AND fiscal_period_end = (
                  SELECT MAX(fiscal_period_end) FROM cur_fundamentals_quarterly
                  WHERE symbol_id = :symbol_id AND available_time <= :asof_ts
              )
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts},
        )

        if not rows:
            return result

        metrics = {
            r["metric_name"]: float(r["metric_value"])
            for r in rows
            if r["metric_value"] is not None
        }

        eps = metrics.get("EarningsPerShareDiluted") or metrics.get("EarningsPerShareBasic")
        if price and eps and eps != 0:
            result["pe_ratio"] = price / eps

        equity = metrics.get("StockholdersEquity")
        shares = metrics.get("CommonStockSharesOutstanding")
        if price and equity and shares and shares > 0:
            book_per_share = equity / shares
            if book_per_share != 0:
                result["pb_ratio"] = price / book_per_share

        debt = metrics.get("LongTermDebt")
        if debt is not None and equity and equity != 0:
            result["debt_to_equity"] = debt / equity

        net_income = metrics.get("NetIncomeLoss")
        if net_income is not None and equity and equity != 0:
            result["roe"] = net_income / equity

        return result

    def _get_insider_activity(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get insider trading signals in a 90-day window."""
        result: dict[str, float | int | None] = {
            "insider_net_shares_90d": None,
            "insider_buy_count_90d": None,
        }
        if not self.db.table_exists("cur_insider_trades"):
            return result

        start_ts = asof_ts - timedelta(days=90)
        rows = self.db.run_query(
            """
            SELECT
                COALESCE(SUM(CASE WHEN transaction_type = 'P' THEN shares
                                  WHEN transaction_type = 'S' THEN -shares
                                  ELSE 0 END), 0) AS net_shares,
                COALESCE(SUM(CASE WHEN transaction_type = 'P' THEN 1 ELSE 0 END), 0) AS buy_count
            FROM cur_insider_trades
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
              AND event_time >= :start_ts
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts, "start_ts": start_ts},
        )

        if rows and rows[0]["net_shares"] is not None:
            result["insider_net_shares_90d"] = float(rows[0]["net_shares"])
            result["insider_buy_count_90d"] = int(rows[0]["buy_count"])

        return result

    def _get_institutional_holdings(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get institutional holder count from latest 13F report."""
        result: dict[str, int | None] = {"institutional_holders_count": None}
        if not self.db.table_exists("cur_institutional_holdings"):
            return result

        rows = self.db.run_query(
            """
            SELECT COUNT(DISTINCT filer_name) AS holder_count
            FROM cur_institutional_holdings
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
              AND report_date = (
                  SELECT MAX(report_date) FROM cur_institutional_holdings
                  WHERE symbol_id = :symbol_id AND available_time <= :asof_ts
              )
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts},
        )

        if rows and rows[0]["holder_count"]:
            result["institutional_holders_count"] = int(rows[0]["holder_count"])

        return result

    def _get_options_iv(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get latest options IV metrics."""
        result: dict[str, float | None] = {
            "iv_30d": None,
            "put_call_volume_ratio": None,
            "skew_25d": None,
        }
        if not self.db.table_exists("cur_options_summary_daily"):
            return result

        rows = self.db.run_query(
            """
            SELECT iv_30d, put_call_volume_ratio, skew_25d
            FROM cur_options_summary_daily
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
            ORDER BY date DESC
            LIMIT 1
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts},
        )

        if rows:
            row = rows[0]
            if row["iv_30d"] is not None:
                result["iv_30d"] = float(row["iv_30d"])
            if row["put_call_volume_ratio"] is not None:
                result["put_call_volume_ratio"] = float(row["put_call_volume_ratio"])
            if row["skew_25d"] is not None:
                result["skew_25d"] = float(row["skew_25d"])

        return result

    def _get_earnings_features(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get earnings surprise and next earnings date."""
        result: dict[str, float | int | None] = {
            "days_to_next_earnings": None,
            "last_eps_surprise_pct": None,
        }
        if not self.db.table_exists("cur_earnings_events"):
            return result

        # Last earnings surprise
        rows = self.db.run_query(
            """
            SELECT eps_surprise_pct
            FROM cur_earnings_events
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
            ORDER BY report_date DESC
            LIMIT 1
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts},
        )

        if rows and rows[0]["eps_surprise_pct"] is not None:
            result["last_eps_surprise_pct"] = float(rows[0]["eps_surprise_pct"])

        # Days to next earnings (future event we know about)
        asof_date = asof_ts.date()
        rows = self.db.run_query(
            """
            SELECT report_date
            FROM cur_earnings_events
            WHERE symbol_id = :symbol_id
              AND report_date > :asof_date
            ORDER BY report_date ASC
            LIMIT 1
        """,
            {"symbol_id": str(symbol_id), "asof_date": asof_date},
        )

        if rows and rows[0]["report_date"]:
            delta = rows[0]["report_date"] - asof_date
            result["days_to_next_earnings"] = delta.days if hasattr(delta, "days") else int(delta)

        return result

    def _get_short_interest(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get latest short interest ratio (days to cover)."""
        result: dict[str, float | None] = {"short_interest_ratio": None}
        if not self.db.table_exists("cur_short_interest"):
            return result

        rows = self.db.run_query(
            """
            SELECT days_to_cover
            FROM cur_short_interest
            WHERE symbol_id = :symbol_id
              AND available_time <= :asof_ts
            ORDER BY settlement_date DESC
            LIMIT 1
        """,
            {"symbol_id": str(symbol_id), "asof_ts": asof_ts},
        )

        if rows and rows[0]["days_to_cover"] is not None:
            result["short_interest_ratio"] = float(rows[0]["days_to_cover"])

        return result

    # Mapping from equity tickers to CFTC COT commodity codes
    _TICKER_TO_COT_CODE: dict[str, str] = {
        "SPY": "13874A",
        "VOO": "13874A",
        "VTI": "13874A",
        "IVV": "13874A",
        "QQQ": "209742",
        "IWM": "239742",
        "TLT": "043602",
        "IEF": "043602",
        "SHY": "043602",
        "GLD": "098662",
        "SLV": "098662",
        "USO": "023651",
        "DIA": "13874A",
    }

    def _get_cot_positioning(self, symbol_id: UUID, asof_ts: datetime) -> dict:
        """Get latest CFTC COT positioning for the symbol's underlying futures market."""
        result: dict[str, float | None] = {
            "cot_noncommercial_net": None,
            "cot_commercial_net": None,
            "cot_noncommercial_pct_oi": None,
        }
        if not self.db.table_exists("cur_cftc_cot"):
            return result

        # Resolve ticker for this symbol_id
        ticker_rows = self.db.run_query(
            "SELECT ticker FROM dim_symbol WHERE symbol_id = :symbol_id",
            {"symbol_id": str(symbol_id)},
        )
        if not ticker_rows:
            return result

        ticker = ticker_rows[0]["ticker"]
        commodity_code = self._TICKER_TO_COT_CODE.get(ticker)
        if commodity_code is None:
            return result

        rows = self.db.run_query(
            """
            SELECT noncommercial_net, commercial_net, noncommercial_pct_oi
            FROM cur_cftc_cot
            WHERE commodity_code = :commodity_code
              AND available_time <= :asof_ts
            ORDER BY report_date DESC
            LIMIT 1
        """,
            {"commodity_code": commodity_code, "asof_ts": asof_ts},
        )

        if rows:
            row = rows[0]
            if row["noncommercial_net"] is not None:
                result["cot_noncommercial_net"] = float(row["noncommercial_net"])
            if row["commercial_net"] is not None:
                result["cot_commercial_net"] = float(row["commercial_net"])
            if row["noncommercial_pct_oi"] is not None:
                result["cot_noncommercial_pct_oi"] = float(row["noncommercial_pct_oi"])

        return result

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
            insert = text("""
                INSERT INTO snap_symbol_features
                    (symbol_id, asof_ts, price_latest, price_change_1d, price_change_7d,
                     volume_avg_20d, volatility_20d, macro_panel, news_counts,
                     pe_ratio, pb_ratio, debt_to_equity, roe,
                     insider_net_shares_90d, insider_buy_count_90d,
                     institutional_holders_count,
                     iv_30d, put_call_volume_ratio, skew_25d,
                     days_to_next_earnings, last_eps_surprise_pct,
                     short_interest_ratio,
                     event_time, available_time)
                VALUES
                    (:symbol_id, :asof_ts, :price_latest, :price_change_1d, :price_change_7d,
                     :volume_avg_20d, :volatility_20d, :macro_panel, :news_counts,
                     :pe_ratio, :pb_ratio, :debt_to_equity, :roe,
                     :insider_net_shares_90d, :insider_buy_count_90d,
                     :institutional_holders_count,
                     :iv_30d, :put_call_volume_ratio, :skew_25d,
                     :days_to_next_earnings, :last_eps_surprise_pct,
                     :short_interest_ratio,
                     :event_time, :available_time)
                ON CONFLICT (symbol_id, asof_ts) DO UPDATE SET
                    price_latest = EXCLUDED.price_latest,
                    price_change_1d = EXCLUDED.price_change_1d,
                    price_change_7d = EXCLUDED.price_change_7d,
                    volume_avg_20d = EXCLUDED.volume_avg_20d,
                    volatility_20d = EXCLUDED.volatility_20d,
                    macro_panel = EXCLUDED.macro_panel,
                    news_counts = EXCLUDED.news_counts,
                    pe_ratio = EXCLUDED.pe_ratio,
                    pb_ratio = EXCLUDED.pb_ratio,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    roe = EXCLUDED.roe,
                    insider_net_shares_90d = EXCLUDED.insider_net_shares_90d,
                    insider_buy_count_90d = EXCLUDED.insider_buy_count_90d,
                    institutional_holders_count = EXCLUDED.institutional_holders_count,
                    iv_30d = EXCLUDED.iv_30d,
                    put_call_volume_ratio = EXCLUDED.put_call_volume_ratio,
                    skew_25d = EXCLUDED.skew_25d,
                    days_to_next_earnings = EXCLUDED.days_to_next_earnings,
                    last_eps_surprise_pct = EXCLUDED.last_eps_surprise_pct,
                    short_interest_ratio = EXCLUDED.short_interest_ratio,
                    updated_at = NOW()
            """)

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
                    "macro_panel": json.dumps(snapshot["macro_panel"], default=str),
                    "news_counts": json.dumps(snapshot["news_counts"], default=str),
                    "pe_ratio": snapshot.get("pe_ratio"),
                    "pb_ratio": snapshot.get("pb_ratio"),
                    "debt_to_equity": snapshot.get("debt_to_equity"),
                    "roe": snapshot.get("roe"),
                    "insider_net_shares_90d": snapshot.get("insider_net_shares_90d"),
                    "insider_buy_count_90d": snapshot.get("insider_buy_count_90d"),
                    "institutional_holders_count": snapshot.get("institutional_holders_count"),
                    "iv_30d": snapshot.get("iv_30d"),
                    "put_call_volume_ratio": snapshot.get("put_call_volume_ratio"),
                    "skew_25d": snapshot.get("skew_25d"),
                    "days_to_next_earnings": snapshot.get("days_to_next_earnings"),
                    "last_eps_surprise_pct": snapshot.get("last_eps_surprise_pct"),
                    "short_interest_ratio": snapshot.get("short_interest_ratio"),
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
