"""Dataset coverage and selection bias audits."""

from __future__ import annotations

import logging

from sqlalchemy import text

from pipeline.db import get_db_manager

logger = logging.getLogger(__name__)


def _extract_volume(market: dict) -> float | None:
    candidates = [
        "volume",
        "volume24hr",
        "volume_24h",
        "volume_1d",
        "volumeUSD",
        "volumeUsd",
        "volume_usd",
    ]
    for key in candidates:
        if key in market and market[key] is not None:
            try:
                return float(market[key])
            except (TypeError, ValueError):
                continue
    return None


def compute_polymarket_coverage(
    selected_markets: list[dict],
    all_markets: list[dict],
) -> dict[str, float]:
    total_active = len(all_markets)
    selected = len(selected_markets)
    coverage_pct = (selected / total_active * 100.0) if total_active > 0 else 0.0

    total_volume = 0.0
    total_volume_count = 0
    for market in all_markets:
        vol = _extract_volume(market)
        if vol is None:
            continue
        total_volume += vol
        total_volume_count += 1

    selected_volume = 0.0
    selected_volume_count = 0
    for market in selected_markets:
        vol = _extract_volume(market)
        if vol is None:
            continue
        selected_volume += vol
        selected_volume_count += 1

    volume_share = (selected_volume / total_volume * 100.0) if total_volume > 0 else 0.0

    return {
        "selected_count": float(selected),
        "total_active_count": float(total_active),
        "selection_coverage_pct": coverage_pct,
        "selected_volume": float(selected_volume),
        "total_volume": float(total_volume),
        "volume_share_pct": volume_share,
        "selected_volume_count": float(selected_volume_count),
        "total_volume_count": float(total_volume_count),
    }


def record_coverage_metrics(source_name: str, metrics: dict[str, float]) -> None:
    db = get_db_manager()
    if not db.table_exists("meta_dataset_coverage"):
        logger.warning("meta_dataset_coverage table missing; skipping coverage metrics")
        return

    with db.engine.connect() as conn:
        for metric_name, value in metrics.items():
            conn.execute(
                text(
                    """
                    INSERT INTO meta_dataset_coverage
                        (source_name, metric_name, metric_value)
                    VALUES
                        (:source_name, :metric_name, :metric_value)
                """
                ),
                {
                    "source_name": source_name,
                    "metric_name": metric_name,
                    "metric_value": value,
                },
            )
        conn.commit()
