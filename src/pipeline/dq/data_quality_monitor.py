"""Data quality monitoring and alerting for production pipelines."""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pipeline.db import _validate_identifier, get_db_manager

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityAlert:
    """Data quality alert."""

    alert_id: str
    table_name: str
    check_name: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None


class DataQualityMonitor:
    """Monitor data quality across all tables."""

    def __init__(self):
        self.db = get_db_manager()
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure monitoring tables exist."""
        from sqlalchemy import text

        with self.db.engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS meta_data_quality_checks (
                    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    table_name VARCHAR(100) NOT NULL,
                    check_name VARCHAR(100) NOT NULL,
                    check_type VARCHAR(50) NOT NULL,
                    threshold_value NUMERIC,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            )

            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS meta_data_quality_alerts (
                    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    check_id UUID REFERENCES meta_data_quality_checks(check_id),
                    table_name VARCHAR(100) NOT NULL,
                    check_name VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    resolved_at TIMESTAMPTZ,
                    resolved_by VARCHAR(100)
                )
            """)
            )

            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS meta_data_quality_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    table_name VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value NUMERIC,
                    sample_size INTEGER,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            )

            conn.commit()

    def check_freshness(
        self, table_name: str, timestamp_col: str = "available_time", max_age_hours: float = 24.0
    ) -> DataQualityAlert | None:
        """Check if data is fresh."""
        table_name = _validate_identifier(table_name)
        timestamp_col = _validate_identifier(timestamp_col)
        result = self.db.run_query(f"""
            SELECT
                MAX({timestamp_col}) as latest_ts,
                COUNT(*) as total_rows
            FROM {table_name}
        """)

        if not result or not result[0]["latest_ts"]:
            return DataQualityAlert(
                alert_id=f"freshness_{table_name}_{datetime.now(UTC).isoformat()}",
                table_name=table_name,
                check_name="freshness",
                severity=Severity.ERROR,
                message=f"No data found in {table_name}",
                details={"total_rows": 0},
            )

        latest_ts = result[0]["latest_ts"]
        if isinstance(latest_ts, str):
            latest_ts = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))

        age_hours = (datetime.now(UTC) - latest_ts.replace(tzinfo=None)).total_seconds() / 3600

        if age_hours > max_age_hours:
            severity = Severity.CRITICAL if age_hours > max_age_hours * 2 else Severity.WARNING
            return DataQualityAlert(
                alert_id=f"freshness_{table_name}_{datetime.now(UTC).isoformat()}",
                table_name=table_name,
                check_name="freshness",
                severity=severity,
                message=(
                    f"Data in {table_name} is {age_hours:.1f} hours stale"
                    f" (threshold: {max_age_hours}h)"
                ),
                details={
                    "latest_timestamp": latest_ts.isoformat(),
                    "age_hours": age_hours,
                    "threshold_hours": max_age_hours,
                    "total_rows": result[0]["total_rows"],
                },
            )

        return None

    def check_completeness(
        self, table_name: str, required_cols: list[str], min_completeness_pct: float = 95.0
    ) -> DataQualityAlert | None:
        """Check data completeness."""
        table_name = _validate_identifier(table_name)
        col_checks = ", ".join(
            [
                f"SUM(CASE WHEN {_validate_identifier(col)} IS NULL THEN 1 ELSE 0 END)"
                f" as {_validate_identifier(col)}_nulls"
                for col in required_cols
            ]
        )

        result = self.db.run_query(f"""
            SELECT
                COUNT(*) as total_rows,
                {col_checks}
            FROM {table_name}
        """)

        if not result:
            return None

        total = result[0]["total_rows"]
        incomplete_cols = []

        for col in required_cols:
            nulls = result[0].get(f"{col}_nulls", 0)
            completeness = (1 - nulls / total) * 100 if total > 0 else 0

            if completeness < min_completeness_pct:
                incomplete_cols.append(
                    {"column": col, "completeness_pct": completeness, "null_count": nulls}
                )

        if incomplete_cols:
            return DataQualityAlert(
                alert_id=f"completeness_{table_name}_{datetime.now(UTC).isoformat()}",
                table_name=table_name,
                check_name="completeness",
                severity=Severity.WARNING,
                message=(
                    f"Low completeness in {table_name}:"
                    f" {len(incomplete_cols)} columns below threshold"
                ),
                details={"incomplete_columns": incomplete_cols, "total_rows": total},
            )

        return None

    def check_price_anomalies(
        self, table_name: str = "cur_prices_ohlcv_daily", zscore_threshold: float = 4.0
    ) -> DataQualityAlert | None:
        """Check for price anomalies."""
        table_name = _validate_identifier(table_name)
        result = self.db.run_query(
            f"""
            WITH med AS (
                SELECT
                    symbol_id,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY close) AS median_price
                FROM {table_name}
                GROUP BY symbol_id
                HAVING COUNT(*) >= 20
            ),
            mad AS (
                SELECT
                    p.symbol_id,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(p.close - m.median_price))
                        AS mad_price
                FROM {table_name} p
                JOIN med m ON p.symbol_id = m.symbol_id
                GROUP BY p.symbol_id
            )
            SELECT
                p.symbol_id,
                s.ticker,
                p.date,
                p.close,
                m.median_price,
                d.mad_price,
                ABS(0.6745 * (p.close - m.median_price) / NULLIF(d.mad_price, 0)) as zscore
            FROM {table_name} p
            JOIN med m ON p.symbol_id = m.symbol_id
            JOIN mad d ON p.symbol_id = d.symbol_id
            JOIN dim_symbol s ON p.symbol_id = s.symbol_id
            WHERE ABS(0.6745 * (p.close - m.median_price) / NULLIF(d.mad_price, 0)) > :threshold
            ORDER BY zscore DESC
            LIMIT 10
        """,
            {"threshold": zscore_threshold},
        )

        if result:
            return DataQualityAlert(
                alert_id=f"price_anomaly_{datetime.now(UTC).isoformat()}",
                table_name=table_name,
                check_name="price_anomaly",
                severity=Severity.WARNING,
                message=f"Found {len(result)} price anomalies (Z-score > {zscore_threshold})",
                details={"anomalies": [dict(r) for r in result]},
            )

        return None

    def check_survivor_bias(self, min_delisted_pct: float = 5.0) -> DataQualityAlert | None:
        """Check for survivor bias in universe."""
        result = self.db.run_query("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_delisted THEN 1 ELSE 0 END) as delisted,
                SUM(CASE WHEN NOT is_delisted THEN 1 ELSE 0 END) as active
            FROM dim_symbol
        """)

        if not result:
            return None

        total = result[0]["total"] or 0
        delisted = result[0]["delisted"] or 0
        delisted_pct = (delisted / total * 100) if total > 0 else 0

        if delisted_pct < min_delisted_pct:
            return DataQualityAlert(
                alert_id=f"survivor_bias_{datetime.now(UTC).isoformat()}",
                table_name="dim_symbol",
                check_name="survivor_bias",
                severity=Severity.WARNING,
                message=(
                    f"Potential survivor bias: only {delisted_pct:.1f}%"
                    f" delisted (min: {min_delisted_pct}%)"
                ),
                details={
                    "total_symbols": total,
                    "delisted": delisted,
                    "active": result[0]["active"],
                    "delisted_pct": delisted_pct,
                },
            )

        return None

    def check_look_ahead_bias(
        self, table_name: str = "snap_contract_features"
    ) -> DataQualityAlert | None:
        """Check for look-ahead bias in snapshots."""
        table_name = _validate_identifier(table_name)
        result = self.db.run_query(f"""
            SELECT
                s.contract_id,
                s.asof_ts,
                p.ts as price_ts,
                p.available_time
            FROM {table_name} s
            JOIN cur_contract_prices p ON s.contract_id = p.contract_id
            WHERE p.available_time > s.asof_ts
              AND p.ts <= s.asof_ts
            LIMIT 10
        """)

        if result:
            return DataQualityAlert(
                alert_id=f"look_ahead_{datetime.now(UTC).isoformat()}",
                table_name=table_name,
                check_name="look_ahead_bias",
                severity=Severity.CRITICAL,
                message=f"CRITICAL: Found {len(result)} instances of look-ahead bias!",
                details={"examples": [dict(r) for r in result]},
            )

        return None

    def run_all_checks(self) -> list[DataQualityAlert]:
        """Run all data quality checks."""
        alerts = []

        # Freshness checks
        freshness_configs = [
            ("cur_prices_ohlcv_daily", "available_time", 48),
            ("cur_contract_prices", "available_time", 2),
            ("cur_macro_observations", "available_time", 168),  # 1 week for macro
        ]

        for table, col, threshold in freshness_configs:
            alert = self.check_freshness(table, col, threshold)
            if alert:
                alerts.append(alert)

        # Completeness checks
        completeness_configs = [
            ("cur_prices_ohlcv_daily", ["open", "high", "low", "close", "volume"]),
            ("cur_contract_prices", ["price_normalized", "ts"]),
        ]

        for table, cols in completeness_configs:
            alert = self.check_completeness(table, cols)
            if alert:
                alerts.append(alert)

        # Price anomalies
        alert = self.check_price_anomalies()
        if alert:
            alerts.append(alert)

        # Survivor bias
        alert = self.check_survivor_bias()
        if alert:
            alerts.append(alert)

        # Look-ahead bias
        alert = self.check_look_ahead_bias()
        if alert:
            alerts.append(alert)

        # Store alerts
        self._store_alerts(alerts)

        return alerts

    def _store_alerts(self, alerts: list[DataQualityAlert]):
        """Store alerts in database."""
        from sqlalchemy import text

        with self.db.engine.connect() as conn:
            for alert in alerts:
                conn.execute(
                    text("""
                    INSERT INTO meta_data_quality_alerts
                    (table_name, check_name, severity, message, details)
                    VALUES (:table_name, :check_name, :severity, :message, :details)
                """),
                    {
                        "table_name": alert.table_name,
                        "check_name": alert.check_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "details": json.dumps(alert.details),
                    },
                )
            conn.commit()

    def generate_quality_report(self, output_path: Path | None = None) -> dict:
        """Generate comprehensive quality report."""
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "checks": {},
            "alerts": [],
            "recommendations": [],
        }

        # Run checks
        alerts = self.run_all_checks()
        report["alerts"] = [
            {
                "table": a.table_name,
                "check": a.check_name,
                "severity": a.severity.value,
                "message": a.message,
            }
            for a in alerts
        ]

        # Table-level metrics
        tables = [
            "cur_prices_ohlcv_daily",
            "cur_prices_adjusted_daily",
            "cur_contract_prices",
            "cur_macro_observations",
            "snap_contract_features",
            "snap_universe_membership",
        ]

        for table in tables:
            if self.db.table_exists(table):
                count = self.db.get_table_count(table)
                report["checks"][table] = {"row_count": count}

        # Recommendations
        if any(a.severity == Severity.CRITICAL for a in alerts):
            report["recommendations"].append(
                "CRITICAL: Fix look-ahead bias immediately before using data for backtesting"
            )

        if any(a.check_name == "survivor_bias" for a in alerts):
            report["recommendations"].append(
                "Add delisted tickers to universe to reduce survivor bias"
            )

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report


def run_quality_monitor() -> dict:
    """CLI-friendly wrapper for quality monitoring."""
    monitor = DataQualityMonitor()
    report = monitor.generate_quality_report()

    # Print summary
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    for table, metrics in report["checks"].items():
        print(f"\n{table}: {metrics['row_count']:,} rows")

    if report["alerts"]:
        print(f"\n⚠️  {len(report['alerts'])} alerts found:")
        for alert in report["alerts"]:
            icon = "🔴" if alert["severity"] == "critical" else "🟡"
            print(f"  {icon} [{alert['severity'].upper()}] {alert['table']}: {alert['message']}")
    else:
        print("\n✅ No quality issues found")

    if report["recommendations"]:
        print("\n📋 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print("=" * 60 + "\n")

    return report
