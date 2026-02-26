"""Estimate and store source latency distributions for conservative availability."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


def _percentile_metric_name(percentile: float) -> str:
    pct = int(round(percentile * 100))
    return f"p{pct}"


def _store_latency_metrics(
    source_name: str,
    window_start: date,
    window_end: date,
    metrics: dict[str, float | None],
    sample_size: int,
) -> None:
    if sample_size <= 0:
        return

    db = get_db_manager()
    with db.engine.connect() as conn:
        for metric_name, value in metrics.items():
            if value is None:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO meta_latency_stats
                        (source_name, metric_name, metric_value, sample_size, window_start, window_end)
                    VALUES
                        (:source_name, :metric_name, :metric_value, :sample_size, :window_start, :window_end)
                    ON CONFLICT (source_name, metric_name, window_start, window_end) DO UPDATE SET
                        metric_value = EXCLUDED.metric_value,
                        sample_size = EXCLUDED.sample_size,
                        computed_at = NOW()
                """
                ),
                {
                    "source_name": source_name,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "sample_size": sample_size,
                    "window_start": window_start,
                    "window_end": window_end,
                },
            )
        conn.commit()


def _compute_latency_stats(sql: str, params: dict) -> dict[str, float | None]:
    db = get_db_manager()
    with db.engine.connect() as conn:
        result = conn.execute(text(sql), params).mappings().first()
        if not result:
            return {}
        return dict(result)


def compute_gdelt_latency_stats(window_days: int) -> dict[str, float | None]:
    """Compute GDELT latency from DATEADDED vs SQLDATE."""
    db = get_db_manager()
    if not db.table_exists("raw_gdelt_events"):
        return {}
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)

    sql = """
        WITH base AS (
            SELECT
                (to_timestamp(NULLIF(r.raw_data::json->>'DATEADDED', ''), 'YYYYMMDDHH24MISS') AT TIME ZONE 'UTC') AS date_added,
                (r.raw_data::json->>'SQLDATE')::date::timestamptz AS event_time
            FROM raw_gdelt_events r
            WHERE r.raw_data ? 'DATEADDED'
              AND r.raw_data ? 'SQLDATE'
              AND (r.raw_data::json->>'SQLDATE')::date >= :start_date
              AND (r.raw_data::json->>'SQLDATE')::date <= :end_date
        ),
        lags AS (
            SELECT EXTRACT(EPOCH FROM (date_added - event_time)) / 60.0 AS lag_min
            FROM base
            WHERE date_added IS NOT NULL
              AND event_time IS NOT NULL
              AND date_added >= event_time
        )
        SELECT
            COUNT(*)::int AS sample_size,
            AVG(lag_min) AS mean_min,
            STDDEV(lag_min) AS std_min,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lag_min) AS p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY lag_min) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lag_min) AS p95
        FROM lags
    """
    stats = _compute_latency_stats(sql, {"start_date": start_date, "end_date": end_date})
    if not stats:
        return {}

    sample_size = int(stats.get("sample_size") or 0)
    metrics = {
        "mean": stats.get("mean_min"),
        "std": stats.get("std_min"),
        "p50": stats.get("p50"),
        "p90": stats.get("p90"),
        "p95": stats.get("p95"),
    }
    _store_latency_metrics("gdelt", start_date, end_date, metrics, sample_size)
    return metrics | {"sample_size": sample_size}


def compute_polymarket_latency_stats(window_days: int) -> dict[str, float | None]:
    """Compute Polymarket latency from extracted_at vs trade timestamp."""
    db = get_db_manager()
    if not db.table_exists("raw_polymarket_trades"):
        return {}
    end_ts = datetime.now(UTC)
    start_ts = end_ts - timedelta(days=window_days)

    sql = """
        WITH lags AS (
            SELECT EXTRACT(EPOCH FROM (r.extracted_at - r.ts)) / 60.0 AS lag_min
            FROM raw_polymarket_trades r
            WHERE r.ts >= :start_ts
              AND r.extracted_at >= r.ts
        )
        SELECT
            COUNT(*)::int AS sample_size,
            AVG(lag_min) AS mean_min,
            STDDEV(lag_min) AS std_min,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lag_min) AS p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY lag_min) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lag_min) AS p95
        FROM lags
    """
    stats = _compute_latency_stats(sql, {"start_ts": start_ts})
    if not stats:
        return {}

    sample_size = int(stats.get("sample_size") or 0)
    metrics = {
        "mean": stats.get("mean_min"),
        "std": stats.get("std_min"),
        "p50": stats.get("p50"),
        "p90": stats.get("p90"),
        "p95": stats.get("p95"),
    }
    _store_latency_metrics("polymarket", start_ts.date(), end_ts.date(), metrics, sample_size)
    return metrics | {"sample_size": sample_size}


def compute_prices_latency_stats(window_days: int) -> dict[str, float | None]:
    """Compute price feed latency from extracted_at vs market close."""
    db = get_db_manager()
    if not db.table_exists("raw_prices_ohlcv"):
        return {}
    settings = get_settings()
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)

    sql = """
        WITH base AS (
            SELECT
                r.extracted_at,
                (r.date + :close_time::time) AT TIME ZONE :exchange_tz AS event_time
            FROM raw_prices_ohlcv r
            WHERE r.date >= :start_date
              AND r.date <= :end_date
        ),
        lags AS (
            SELECT EXTRACT(EPOCH FROM (extracted_at - event_time)) / 60.0 AS lag_min
            FROM base
            WHERE extracted_at >= event_time
        )
        SELECT
            COUNT(*)::int AS sample_size,
            AVG(lag_min) AS mean_min,
            STDDEV(lag_min) AS std_min,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lag_min) AS p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY lag_min) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lag_min) AS p95
        FROM lags
    """
    stats = _compute_latency_stats(
        sql,
        {
            "start_date": start_date,
            "end_date": end_date,
            "close_time": settings.prices.market_close_time,
            "exchange_tz": settings.prices.exchange_timezone,
        },
    )
    if not stats:
        return {}

    sample_size = int(stats.get("sample_size") or 0)
    metrics = {
        "mean": stats.get("mean_min"),
        "std": stats.get("std_min"),
        "p50": stats.get("p50"),
        "p90": stats.get("p90"),
        "p95": stats.get("p95"),
    }
    _store_latency_metrics("prices", start_date, end_date, metrics, sample_size)
    return metrics | {"sample_size": sample_size}


def refresh_latency_stats(window_days: int | None = None) -> dict[str, dict[str, float | None]]:
    """Compute and persist latency stats for all sources."""
    settings = get_settings()
    window_days = window_days or settings.historical_fixes.max_backfill_days

    results: dict[str, dict[str, float | None]] = {}

    try:
        results["gdelt"] = compute_gdelt_latency_stats(window_days)
    except Exception as exc:
        logger.warning(f"Failed to compute GDELT latency stats: {exc}")
        results["gdelt"] = {}

    try:
        results["polymarket"] = compute_polymarket_latency_stats(window_days)
    except Exception as exc:
        logger.warning(f"Failed to compute Polymarket latency stats: {exc}")
        results["polymarket"] = {}

    try:
        results["prices"] = compute_prices_latency_stats(window_days)
    except Exception as exc:
        logger.warning(f"Failed to compute price latency stats: {exc}")
        results["prices"] = {}

    return results


def get_latency_minutes(
    source_name: str,
    percentile: float,
    fallback_minutes: float,
    max_age_hours: int,
    min_samples: int,
) -> float:
    """Fetch latency minutes from meta table; fallback if stale or missing."""
    metric_name = _percentile_metric_name(percentile)
    db = get_db_manager()
    if not db.table_exists("meta_latency_stats"):
        return float(fallback_minutes)
    with db.engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT metric_value, sample_size, computed_at
                FROM meta_latency_stats
                WHERE source_name = :source_name
                  AND metric_name = :metric_name
                ORDER BY computed_at DESC
                LIMIT 1
            """
            ),
            {"source_name": source_name, "metric_name": metric_name},
        ).mappings().first()

    if not row:
        return float(fallback_minutes)

    computed_at = row.get("computed_at")
    if isinstance(computed_at, str):
        computed_at = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
    age_hours = None
    if isinstance(computed_at, datetime):
        age_hours = (datetime.now(UTC) - computed_at).total_seconds() / 3600.0

    sample_size = int(row.get("sample_size") or 0)
    metric_value = row.get("metric_value")

    if sample_size < min_samples:
        return float(fallback_minutes)
    if age_hours is not None and age_hours > max_age_hours:
        return float(fallback_minutes)
    if metric_value is None:
        return float(fallback_minutes)

    return float(metric_value)
