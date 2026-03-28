"""Periodic orderbook snapshot runner with retention policy."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.extract.polymarket import PolymarketExtractor
from pipeline.settings import get_settings
from pipeline.transform.curated import CuratedTransformer

logger = logging.getLogger(__name__)


def parse_interval_to_seconds(interval: str) -> int:
    interval = interval.strip().lower()
    if interval in {"off", "0", "none"}:
        return 0
    if interval.endswith("s"):
        return int(interval[:-1])
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    return int(interval)


class OrderbookSnapshotRunner:
    """Capture orderbook snapshots at a fixed cadence and enforce retention."""

    def __init__(self, run_id: str | None = None):
        self.db = get_db_manager()
        self.settings = get_settings()
        self.extractor = PolymarketExtractor()
        self.run_id = run_id

    def capture_once(self, max_markets: int | None = None) -> int:
        markets = self.extractor.select_markets(max_markets=max_markets)
        if not markets:
            logger.warning("No markets found for orderbook snapshot")
            return 0

        ts = datetime.now(UTC)
        records = []

        for market in markets:
            market_id = market.get("id") or market.get("marketId")
            if not market_id:
                continue
            try:
                orderbook = self.extractor.get_orderbook(market_id)
                if not orderbook:
                    continue
                bids = orderbook.get("bids") or []
                asks = orderbook.get("asks") or []
                best_bid = max((float(b[0]) for b in bids if len(b) >= 2), default=None)
                best_ask = min((float(a[0]) for a in asks if len(a) >= 2), default=None)
                spread = None
                if best_bid is not None and best_ask is not None:
                    spread = float(best_ask) - float(best_bid)

                records.append(
                    {
                        "market_id": market_id,
                        "ts": ts,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "bids": bids,
                        "asks": asks,
                        "raw_data": json.dumps(orderbook, default=str),
                        "extracted_at": ts,
                        "run_id": self.run_id,
                    }
                )
            except Exception as exc:
                logger.warning(f"Orderbook snapshot failed for {market_id}: {exc}")
                continue

        if not records:
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

        with self.db.engine.connect() as conn:
            conn.execute(insert_sql, records)
            conn.commit()

        return len(records)

    def apply_retention(self, retention_days: int) -> None:
        if retention_days <= 0:
            return
        with self.db.engine.connect() as conn:
            conn.execute(
                text("""
                    DELETE FROM raw_polymarket_orderbook_snapshots
                    WHERE ts < NOW() - (:days || ' days')::interval
                """),
                {"days": retention_days},
            )
            conn.execute(
                text("""
                    DELETE FROM cur_contract_orderbook_snapshots
                    WHERE ts < NOW() - (:days || ' days')::interval
                """),
                {"days": retention_days},
            )
            conn.commit()

    def run(
        self,
        interval: str,
        iterations: int = 1,
        retention_days: int = 30,
        transform: bool = True,
        max_markets: int | None = None,
    ) -> int:
        interval_seconds = parse_interval_to_seconds(interval)
        total = 0
        remaining = iterations
        while True:
            captured = self.capture_once(max_markets=max_markets)
            total += captured
            if transform:
                CuratedTransformer(run_id=self.run_id).transform_contract_orderbooks()
            if retention_days:
                self.apply_retention(retention_days)

            if remaining == 1:
                break
            if remaining > 1:
                remaining -= 1
            if interval_seconds <= 0:
                break
            time.sleep(interval_seconds)

        return total
