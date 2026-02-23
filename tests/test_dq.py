"""Data quality tests."""

import pytest
from sqlalchemy.exc import OperationalError

from pipeline.db import get_db_manager, reset_db_manager
from pipeline.dq.tests_sql import DataQualityTests


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


class TestTimeMonotonicity:
    """Test that available_time >= event_time."""

    def test_macro_observations_time_order(self, db):
        """Macro observations should have available_time >= event_time."""
        if not db.table_exists("cur_macro_observations"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT COUNT(*) as cnt
            FROM cur_macro_observations
            WHERE available_time < event_time
        """)

        assert result[0]["cnt"] == 0, "Found records where available_time < event_time"

    def test_contract_prices_time_order(self, db):
        """Contract prices should have available_time >= event_time."""
        if not db.table_exists("cur_contract_prices"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT COUNT(*) as cnt
            FROM cur_contract_prices
            WHERE available_time < event_time
        """)

        assert result[0]["cnt"] == 0, "Found records where available_time < event_time"


class TestPrimaryKeyUniqueness:
    """Test that primary keys are unique in curated tables."""

    def test_prices_ohlcv_pk_unique(self, db):
        """OHLCV prices should have unique (symbol_id, date)."""
        if not db.table_exists("cur_prices_ohlcv_daily"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT symbol_id, date, COUNT(*) as cnt
            FROM cur_prices_ohlcv_daily
            GROUP BY symbol_id, date
            HAVING COUNT(*) > 1
        """)

        assert len(result) == 0, f"Found duplicate PKs: {result}"

    def test_contract_prices_pk_unique(self, db):
        """Contract prices should have unique (contract_id, ts, outcome)."""
        if not db.table_exists("cur_contract_prices"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT contract_id, ts, outcome, COUNT(*) as cnt
            FROM cur_contract_prices
            GROUP BY contract_id, ts, outcome
            HAVING COUNT(*) > 1
        """)

        assert len(result) == 0, f"Found duplicate PKs: {result}"


class TestReferentialIntegrity:
    """Test referential integrity across tables."""

    def test_contract_prices_reference_valid_contracts(self, db):
        """All contract prices should reference valid contracts."""
        if not db.table_exists("cur_contract_prices") or not db.table_exists("dim_contract"):
            pytest.skip("Tables not created")

        result = db.run_query("""
            SELECT COUNT(DISTINCT cp.contract_id) as orphan_count
            FROM cur_contract_prices cp
            LEFT JOIN dim_contract c ON cp.contract_id = c.contract_id
            WHERE c.contract_id IS NULL
        """)

        assert result[0]["orphan_count"] == 0, "Found orphan contract prices"

    def test_prices_reference_valid_symbols(self, db):
        """All prices should reference valid symbols."""
        if not db.table_exists("cur_prices_ohlcv_daily") or not db.table_exists("dim_symbol"):
            pytest.skip("Tables not created")

        result = db.run_query("""
            SELECT COUNT(DISTINCT p.symbol_id) as orphan_count
            FROM cur_prices_ohlcv_daily p
            LEFT JOIN dim_symbol s ON p.symbol_id = s.symbol_id
            WHERE s.symbol_id IS NULL
        """)

        assert result[0]["orphan_count"] == 0, "Found orphan prices"


class TestCoverageSanity:
    """Test data coverage and sanity checks."""

    def test_no_negative_volumes(self, db):
        """Volume should never be negative."""
        if not db.table_exists("cur_prices_ohlcv_daily"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT COUNT(*) as cnt
            FROM cur_prices_ohlcv_daily
            WHERE volume < 0
        """)

        assert result[0]["cnt"] == 0, "Found negative volumes"

    def test_no_negative_prices(self, db):
        """Prices should never be negative."""
        if not db.table_exists("cur_prices_ohlcv_daily"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT COUNT(*) as cnt
            FROM cur_prices_ohlcv_daily
            WHERE open < 0 OR high < 0 OR low < 0 OR close < 0
        """)

        assert result[0]["cnt"] == 0, "Found negative prices"

    def test_contract_prices_in_range(self, db):
        """Normalized contract prices should be in [0, 1]."""
        if not db.table_exists("cur_contract_prices"):
            pytest.skip("Table not created")

        result = db.run_query("""
            SELECT COUNT(*) as cnt
            FROM cur_contract_prices
            WHERE price_normalized < 0 OR price_normalized > 1
        """)

        assert result[0]["cnt"] == 0, "Found prices outside [0, 1]"


class TestSnapshotCorrectness:
    """Test snapshot anti-look-ahead."""

    def test_snapshots_no_lookahead(self, db):
        """Snapshots should not contain data with available_time > asof_ts."""
        if not db.table_exists("snap_contract_features"):
            pytest.skip("Table not created")

        # Sample some snapshots and verify
        result = db.run_query("""
            SELECT contract_id, asof_ts
            FROM snap_contract_features
            ORDER BY RANDOM()
            LIMIT 10
        """)

        for row in result:
            contract_id = row["contract_id"]
            asof_ts = row["asof_ts"]

            # Check for look-ahead in prices
            lookahead = db.run_query(
                """
                SELECT COUNT(*) as cnt
                FROM cur_contract_prices
                WHERE contract_id = :contract_id
                  AND available_time > :asof_ts
                  AND ts <= :asof_ts
            """,
                {"contract_id": contract_id, "asof_ts": asof_ts},
            )

            assert lookahead[0]["cnt"] == 0, (
                f"Look-ahead detected in snapshot {contract_id} at {asof_ts}"
            )


class TestDataQualitySuite:
    """Run the full DQ test suite."""

    def test_full_dq_suite(self, db):
        """Run all DQ tests."""
        tester = DataQualityTests()
        results = tester.run_all_tests()

        failed_tests = [name for name, (passed, _) in results.items() if not passed]

        if failed_tests:
            pytest.fail(f"DQ tests failed: {failed_tests}")
