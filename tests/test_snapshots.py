"""Tests for snapshot builder."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError

from pipeline.db import get_db_manager, reset_db_manager
from pipeline.snapshot.contract_snapshots import ContractSnapshotBuilder


@pytest.fixture
def db():
    """Provide database manager for tests."""
    reset_db_manager()
    db_manager = get_db_manager()
    try:
        with db_manager.engine.connect():
            pass
    except OperationalError as exc:
        pytest.skip(f"Database not available: {exc}")
    return db_manager


@pytest.fixture
def sample_contract(db):
    """Create a sample contract for testing."""
    contract_id = uuid4()

    with db.engine.connect() as conn:
        from sqlalchemy import text

        # Insert source
        conn.execute(
            text("""
            INSERT INTO dim_source (name, type)
            VALUES ('test_source', 'api')
            ON CONFLICT DO NOTHING
        """)
        )

        # Get source_id
        result = conn.execute(text("SELECT source_id FROM dim_source WHERE name = 'test_source'"))
        source_id = result.scalar()

        # Insert contract
        conn.execute(
            text("""
            INSERT INTO dim_contract
            (contract_id, venue, venue_market_id, title,
             outcome_type, outcomes, status, created_time,
             available_time, source_id)
            VALUES (:contract_id, 'polymarket', 'test_market_1',
             'Test Market', 'binary', '["YES", "NO"]', 'active',
             NOW(), NOW(), :source_id)
        """),
            {"contract_id": str(contract_id), "source_id": str(source_id)},
        )

        conn.commit()

    return contract_id


class TestContractSnapshotBuilder:
    """Test the contract snapshot builder."""

    def test_build_snapshot_basic(self, db, sample_contract):
        """Test building a basic snapshot."""
        # Insert some price data
        with db.engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(
                text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw,
                 price_normalized, event_time, available_time)
                VALUES (:contract_id, NOW(), 'YES', 55.0,
                 0.55, NOW(), NOW())
            """),
                {"contract_id": str(sample_contract)},
            )

            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(
            contract_id=sample_contract, asof_ts=datetime.now(timezone.utc)
        )

        assert snapshot is not None
        assert snapshot["contract_id"] == sample_contract
        assert "implied_p_yes" in snapshot

    def test_snapshot_no_lookahead(self, db, sample_contract):
        """Test that snapshots don't include future data."""
        now = datetime.now(timezone.utc)

        # Insert price at now
        with db.engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(
                text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw,
                 price_normalized, event_time, available_time)
                VALUES (:contract_id, :ts, 'YES', 55.0,
                 0.55, :ts, :ts)
            """),
                {"contract_id": str(sample_contract), "ts": now},
            )

            conn.commit()

        builder = ContractSnapshotBuilder()

        # Build snapshot for 1 hour ago
        past_ts = now - timedelta(hours=1)
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=past_ts)

        # Should not have price since it was recorded at 'now'
        assert snapshot["implied_p_yes"] is None

    def test_snapshot_uses_latest_available(self, db, sample_contract):
        """Test that snapshots use the latest available data."""
        now = datetime.now(timezone.utc)

        # Insert multiple prices
        with db.engine.connect() as conn:
            from sqlalchemy import text

            for i, minutes_ago in enumerate([60, 30, 10]):
                ts = now - timedelta(minutes=minutes_ago)
                price = 50.0 + i * 5  # 50, 55, 60

                conn.execute(
                    text("""
                    INSERT INTO cur_contract_prices
                    (contract_id, ts, outcome, price_raw,
                     price_normalized, event_time, available_time)
                    VALUES (:contract_id, :ts, 'YES', :price,
                     :price_norm, :ts, :ts)
                """),
                    {
                        "contract_id": str(sample_contract),
                        "ts": ts,
                        "price": price,
                        "price_norm": price / 100.0,
                    },
                )

            conn.commit()

        builder = ContractSnapshotBuilder()

        # Build snapshot for 5 minutes ago - should get price from 10 minutes ago
        snapshot_ts = now - timedelta(minutes=5)
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=snapshot_ts)

        assert snapshot["implied_p_yes"] == 0.60  # Latest available at that time

    def test_snapshot_trade_aggregation(self, db, sample_contract):
        """Test that trade statistics are correctly aggregated."""
        now = datetime.now(timezone.utc)

        # Insert some trades
        with db.engine.connect() as conn:
            from sqlalchemy import text

            for i in range(5):
                ts = now - timedelta(minutes=i * 10)
                conn.execute(
                    text("""
                    INSERT INTO cur_contract_trades
                    (contract_id, trade_id, ts, price, size,
                     side, event_time, available_time)
                    VALUES (:contract_id, :trade_id, :ts, :price,
                     :size, 'buy', :ts, :ts)
                """),
                    {
                        "contract_id": str(sample_contract),
                        "trade_id": f"trade_{i}",
                        "ts": ts,
                        "price": 50.0 + i,
                        "size": 100.0,
                    },
                )

            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=now)

        assert snapshot["trade_count_24h"] == 5
        assert snapshot["volume_24h"] == 500.0

    def test_snapshot_includes_staleness_metrics(self, db, sample_contract):
        """Test that snapshots include price and macro staleness fields."""
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(
                text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw,
                 price_normalized, event_time, available_time)
                VALUES (:contract_id, :ts, 'YES', 55.0, 0.55, :ts, :ts)
            """),
                {"contract_id": str(sample_contract), "ts": now - timedelta(hours=2)},
            )
            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=now)

        assert "price_staleness_hours" in snapshot
        assert "macro_staleness_days" in snapshot
        assert "data_quality_score" in snapshot
        # Price was inserted 2 hours ago; staleness should be ~2 hours
        assert snapshot["price_staleness_hours"] is not None
        assert snapshot["price_staleness_hours"] >= 1.9

    def test_snapshot_quality_score_deducted_for_staleness(self, db, sample_contract):
        """Test that data_quality_score is reduced for stale price data."""
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            from sqlalchemy import text

            # Insert price that is 12 hours stale
            conn.execute(
                text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw,
                 price_normalized, event_time, available_time)
                VALUES (:contract_id, :ts, 'YES', 55.0, 0.55, :ts, :ts)
            """),
                {"contract_id": str(sample_contract), "ts": now - timedelta(hours=12)},
            )
            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=now)

        # Price was inserted 12 hours ago → price_staleness deduction capped at 30 points.
        # No microstructure data → additional -10 points. Score should be below 100.
        assert snapshot["data_quality_score"] < 100.0

    def test_snapshot_microstructure_features(self, db, sample_contract):
        """Test microstructure features are computed from buy/sell trades."""
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            from sqlalchemy import text

            # Insert price data so implied_p_yes is populated
            conn.execute(
                text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw,
                 price_normalized, event_time, available_time)
                VALUES (:contract_id, :ts, 'YES', 55.0, 0.55, :ts, :ts)
            """),
                {"contract_id": str(sample_contract), "ts": now},
            )

            # 6 buy trades (size=100) and 4 sell trades (size=100)
            for i in range(6):
                conn.execute(
                    text("""
                    INSERT INTO cur_contract_trades
                    (contract_id, trade_id, ts, price, size,
                     side, event_time, available_time)
                    VALUES (:contract_id, :trade_id, :ts, 0.55, 100.0, 'buy', :ts, :ts)
                """),
                    {
                        "contract_id": str(sample_contract),
                        "trade_id": f"buy_{i}",
                        "ts": now - timedelta(minutes=i),
                    },
                )
            for i in range(4):
                conn.execute(
                    text("""
                    INSERT INTO cur_contract_trades
                    (contract_id, trade_id, ts, price, size,
                     side, event_time, available_time)
                    VALUES (:contract_id, :trade_id, :ts, 0.45, 100.0, 'sell', :ts, :ts)
                """),
                    {
                        "contract_id": str(sample_contract),
                        "trade_id": f"sell_{i}",
                        "ts": now - timedelta(minutes=i + 10),
                    },
                )
            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=now)

        # buy_vol=600, sell_vol=400, total=1000
        # trade_imbalance = (600-400)/1000 = 0.2
        assert snapshot["micro_trade_imbalance"] is not None
        assert abs(snapshot["micro_trade_imbalance"] - 0.2) < 0.01
        assert snapshot["micro_buy_sell_ratio"] is not None
        assert abs(snapshot["micro_buy_sell_ratio"] - 1.5) < 0.01

    def test_snapshot_outlier_detection(self, db, sample_contract):
        """Test that price outlier detection is included in snapshots."""
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            from sqlalchemy import text

            # Insert 15 normal prices around 0.5 and one extreme outlier
            for i in range(15):
                ts = now - timedelta(days=30 - i)
                price = 0.50 + (i % 3) * 0.01  # prices between 0.50 and 0.52
                conn.execute(
                    text("""
                    INSERT INTO cur_contract_prices
                    (contract_id, ts, outcome, price_raw,
                     price_normalized, event_time, available_time)
                    VALUES (:contract_id, :ts, 'YES', :p, :p, :ts, :ts)
                """),
                    {"contract_id": str(sample_contract), "ts": ts, "p": price},
                )
            conn.commit()

        builder = ContractSnapshotBuilder()
        snapshot = builder.build_contract_snapshot(contract_id=sample_contract, asof_ts=now)

        assert "has_price_outliers" in snapshot
        assert "outlier_score" in snapshot

