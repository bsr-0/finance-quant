"""Data quality tests using SQL assertions."""

import logging
from typing import Dict, List, Tuple

from pipeline.db import _validate_identifier, get_db_manager

logger = logging.getLogger(__name__)


class DataQualityTests:
    """Run data quality tests against the warehouse."""
    
    def __init__(self):
        self.db = get_db_manager()
        self.results: Dict[str, Tuple[bool, str]] = {}
    
    def run_all_tests(self) -> Dict[str, Tuple[bool, str]]:
        """Run all data quality tests."""
        logger.info("Running data quality tests...")
        
        self.test_time_monotonicity()
        self.test_no_duplicate_pks()
        self.test_referential_integrity()
        self.test_coverage_sanity()
        self.test_snapshot_anti_lookahead()
        
        return self.results
    
    def test_time_monotonicity(self) -> bool:
        """DQ1: available_time >= event_time for all tables."""
        logger.info("Testing time monotonicity...")
        
        tables = [
            ("cur_prices_ohlcv_daily", "event_time", "available_time"),
            ("cur_macro_observations", "event_time", "available_time"),
            ("cur_contract_prices", "event_time", "available_time"),
            ("cur_contract_trades", "event_time", "available_time"),
            ("cur_world_events", "event_time", "available_time"),
        ]
        
        violations = []
        for table, event_col, avail_col in tables:
            if not self.db.table_exists(table):
                continue

            table = _validate_identifier(table)
            event_col = _validate_identifier(event_col)
            avail_col = _validate_identifier(avail_col)
            query = f"""
                SELECT COUNT(*) as cnt
                FROM {table}
                WHERE {avail_col} < {event_col}
            """
            result = self.db.run_query(query)
            count = result[0]["cnt"] if result else 0
            
            if count > 0:
                violations.append(f"{table}: {count} rows")
        
        passed = len(violations) == 0
        message = "PASSED" if passed else f"Violations: {', '.join(violations)}"
        self.results["time_monotonicity"] = (passed, message)
        logger.info(f"Time monotonicity: {message}")
        return passed
    
    def test_no_duplicate_pks(self) -> bool:
        """DQ2: No duplicate primary keys in curated tables."""
        logger.info("Testing no duplicate PKs...")
        
        pk_tests = [
            ("cur_prices_ohlcv_daily", ["symbol_id", "date"]),
            ("cur_macro_observations", ["series_id", "period_end", "revision_id"]),
            ("cur_contract_prices", ["contract_id", "ts", "outcome"]),
            ("cur_contract_trades", ["contract_id", "trade_id"]),
            ("dim_contract", ["venue", "venue_market_id"]),
        ]
        
        violations = []
        for table, pk_cols in pk_tests:
            if not self.db.table_exists(table):
                continue

            table = _validate_identifier(table)
            pk_str = ", ".join(_validate_identifier(c) for c in pk_cols)
            query = f"""
                SELECT {pk_str}, COUNT(*) as cnt
                FROM {table}
                GROUP BY {pk_str}
                HAVING COUNT(*) > 1
            """
            result = self.db.run_query(query)
            
            if result:
                violations.append(f"{table}: {len(result)} duplicates")
        
        passed = len(violations) == 0
        message = "PASSED" if passed else f"Violations: {', '.join(violations)}"
        self.results["no_duplicate_pks"] = (passed, message)
        logger.info(f"No duplicate PKs: {message}")
        return passed
    
    def test_referential_integrity(self) -> bool:
        """DQ4: Referential integrity across tables."""
        logger.info("Testing referential integrity...")
        
        fk_tests = [
            ("cur_contract_prices", "contract_id", "dim_contract", "contract_id"),
            ("cur_contract_trades", "contract_id", "dim_contract", "contract_id"),
            ("cur_prices_ohlcv_daily", "symbol_id", "dim_symbol", "symbol_id"),
            ("cur_macro_observations", "series_id", "dim_macro_series", "series_id"),
        ]
        
        violations = []
        for child_table, fk_col, parent_table, pk_col in fk_tests:
            if not self.db.table_exists(child_table) or not self.db.table_exists(parent_table):
                continue

            child_table = _validate_identifier(child_table)
            fk_col = _validate_identifier(fk_col)
            parent_table = _validate_identifier(parent_table)
            pk_col = _validate_identifier(pk_col)
            query = f"""
                SELECT COUNT(DISTINCT c.{fk_col}) as orphan_count
                FROM {child_table} c
                LEFT JOIN {parent_table} p ON c.{fk_col} = p.{pk_col}
                WHERE p.{pk_col} IS NULL
            """
            result = self.db.run_query(query)
            count = result[0]["orphan_count"] if result else 0
            
            if count > 0:
                violations.append(f"{child_table}.{fk_col}: {count} orphans")
        
        passed = len(violations) == 0
        message = "PASSED" if passed else f"Violations: {', '.join(violations)}"
        self.results["referential_integrity"] = (passed, message)
        logger.info(f"Referential integrity: {message}")
        return passed
    
    def test_coverage_sanity(self) -> bool:
        """DQ5: Coverage sanity checks."""
        logger.info("Testing coverage sanity...")
        
        violations = []
        
        # Check OHLCV: no negative volume
        if self.db.table_exists("cur_prices_ohlcv_daily"):
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_prices_ohlcv_daily
                WHERE volume < 0
            """
            result = self.db.run_query(query)
            if result and result[0]["cnt"] > 0:
                violations.append(f"Negative volume: {result[0]['cnt']} rows")
        
        # Check OHLCV: non-negative prices
        if self.db.table_exists("cur_prices_ohlcv_daily"):
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_prices_ohlcv_daily
                WHERE open < 0 OR high < 0 OR low < 0 OR close < 0
            """
            result = self.db.run_query(query)
            if result and result[0]["cnt"] > 0:
                violations.append(f"Negative prices: {result[0]['cnt']} rows")
        
        # Check contract prices: within [0,1] after normalization
        if self.db.table_exists("cur_contract_prices"):
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_contract_prices
                WHERE price_normalized < 0 OR price_normalized > 1
            """
            result = self.db.run_query(query)
            if result and result[0]["cnt"] > 0:
                violations.append(f"Price out of range [0,1]: {result[0]['cnt']} rows")
        
        passed = len(violations) == 0
        message = "PASSED" if passed else f"Violations: {', '.join(violations)}"
        self.results["coverage_sanity"] = (passed, message)
        logger.info(f"Coverage sanity: {message}")
        return passed
    
    def test_snapshot_anti_lookahead(self, sample_size: int = 100) -> bool:
        """DQ3: Verify no look-ahead bias in snapshots."""
        logger.info("Testing snapshot anti-look-ahead...")
        
        if not self.db.table_exists("snap_contract_features"):
            self.results["snapshot_anti_lookahead"] = (True, "No snapshots to test")
            return True
        
        # Sample random snapshots
        query = """
            SELECT contract_id, asof_ts
            FROM snap_contract_features
            ORDER BY RANDOM()
            LIMIT :sample_size
        """
        samples = self.db.run_query(query, {"sample_size": sample_size})
        
        if not samples:
            self.results["snapshot_anti_lookahead"] = (True, "No snapshots to test")
            return True
        
        violations = []
        
        for sample in samples:
            contract_id = sample["contract_id"]
            asof_ts = sample["asof_ts"]
            
            # Check contract prices
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_contract_prices
                WHERE contract_id = :contract_id
                  AND available_time > :asof_ts
                  AND ts <= :asof_ts
            """
            result = self.db.run_query(query, {
                "contract_id": contract_id,
                "asof_ts": asof_ts
            })
            if result and result[0]["cnt"] > 0:
                violations.append(f"Price look-ahead: {contract_id} at {asof_ts}")
            
            # Check trades
            query = """
                SELECT COUNT(*) as cnt
                FROM cur_contract_trades
                WHERE contract_id = :contract_id
                  AND available_time > :asof_ts
                  AND ts <= :asof_ts
            """
            result = self.db.run_query(query, {
                "contract_id": contract_id,
                "asof_ts": asof_ts
            })
            if result and result[0]["cnt"] > 0:
                violations.append(f"Trade look-ahead: {contract_id} at {asof_ts}")
        
        passed = len(violations) == 0
        message = "PASSED" if passed else f"Violations: {len(violations)} samples"
        self.results["snapshot_anti_lookahead"] = (passed, message)
        logger.info(f"Snapshot anti-look-ahead: {message}")
        return passed
    
    def print_report(self) -> None:
        """Print formatted test results."""
        print("\n" + "=" * 60)
        print("DATA QUALITY TEST REPORT")
        print("=" * 60)
        
        for test_name, (passed, message) in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
            if not passed:
                print(f"       {message}")
        
        print("=" * 60)
        total = len(self.results)
        passed_count = sum(1 for p, _ in self.results.values() if p)
        print(f"SUMMARY: {passed_count}/{total} tests passed")
        print("=" * 60 + "\n")


def run_dq_tests() -> bool:
    """CLI-friendly wrapper for running DQ tests."""
    tester = DataQualityTests()
    tester.run_all_tests()
    tester.print_report()
    
    return all(passed for passed, _ in tester.results.values())
