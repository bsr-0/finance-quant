"""Data lineage tracking for reproducibility."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from pipeline.db import _validate_identifier, get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class DataLineage:
    """Record of data transformation lineage."""

    lineage_id: UUID = field(default_factory=uuid4)
    run_id: UUID | None = None
    source_table: str = ""
    target_table: str = ""
    transformation_name: str = ""
    source_query: str = ""
    transformation_logic: str = ""
    source_hash: str = ""
    target_hash: str = ""
    record_count_source: int = 0
    record_count_target: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lineage_id": str(self.lineage_id),
            "run_id": str(self.run_id) if self.run_id else None,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "transformation_name": self.transformation_name,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "record_count_source": self.record_count_source,
            "record_count_target": self.record_count_target,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class LineageTracker:
    """Track data lineage through the pipeline."""

    def __init__(self):
        self.db = get_db_manager()
        self._ensure_table()

    def _ensure_table(self):
        """Ensure lineage table exists."""
        with self.db.engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS meta_data_lineage (
                    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_id UUID REFERENCES meta_pipeline_runs(run_id),
                    source_table VARCHAR(100) NOT NULL,
                    target_table VARCHAR(100) NOT NULL,
                    transformation_name VARCHAR(200) NOT NULL,
                    source_query TEXT,
                    transformation_logic TEXT,
                    source_hash VARCHAR(64),
                    target_hash VARCHAR(64),
                    record_count_source INTEGER,
                    record_count_target INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB
                )
            """)
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_lineage_run_id ON meta_data_lineage(run_id)
            """)
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_lineage_source ON meta_data_lineage(source_table)
            """)
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_lineage_target ON meta_data_lineage(target_table)
            """)
            )
            conn.commit()

    def compute_table_hash(self, table_name: str, query_filter: str = "") -> str:
        """Compute hash of table contents for versioning."""
        try:
            table_name = _validate_identifier(table_name)
            query = f"""
                SELECT md5(string_agg(row_hash, ',' ORDER BY row_hash)) as table_hash
                FROM (
                    SELECT md5(row_to_json(t)::text) as row_hash
                    FROM {table_name} t
                    {query_filter}
                ) subq
            """
            result = self.db.run_query(query)
            return result[0]["table_hash"] if result else ""
        except Exception as e:
            logger.warning(f"Could not compute hash for {table_name}: {e}")
            return ""

    def get_table_count(self, table_name: str, query_filter: str = "") -> int:
        """Get row count for a table."""
        table_name = _validate_identifier(table_name)
        query = f"SELECT COUNT(*) as cnt FROM {table_name} {query_filter}"
        result = self.db.run_query(query)
        return result[0]["cnt"] if result else 0

    def record_lineage(
        self,
        source_table: str,
        target_table: str,
        transformation_name: str,
        run_id: UUID | None = None,
        source_query: str = "",
        transformation_logic: str = "",
        metadata: dict | None = None,
    ) -> UUID:
        """Record data lineage."""

        # Compute hashes and counts
        source_hash = self.compute_table_hash(source_table)
        target_hash = self.compute_table_hash(target_table)
        source_count = self.get_table_count(source_table)
        target_count = self.get_table_count(target_table)

        lineage_id = uuid4()

        with self.db.engine.connect() as conn:
            conn.execute(
                text("""
                INSERT INTO meta_data_lineage
                (lineage_id, run_id, source_table, target_table, transformation_name,
                 source_query, transformation_logic, source_hash, target_hash,
                 record_count_source, record_count_target, metadata)
                VALUES (:lineage_id, :run_id, :source_table, :target_table, :transformation_name,
                        :source_query, :transformation_logic, :source_hash, :target_hash,
                        :record_count_source, :record_count_target, :metadata)
            """),
                {
                    "lineage_id": str(lineage_id),
                    "run_id": str(run_id) if run_id else None,
                    "source_table": source_table,
                    "target_table": target_table,
                    "transformation_name": transformation_name,
                    "source_query": source_query,
                    "transformation_logic": transformation_logic,
                    "source_hash": source_hash,
                    "target_hash": target_hash,
                    "record_count_source": source_count,
                    "record_count_target": target_count,
                    "metadata": json.dumps(metadata or {}),
                },
            )
            conn.commit()

        logger.info(
            f"Recorded lineage: {source_table} -> {target_table} "
            f"({source_count} -> {target_count} rows)"
        )

        return lineage_id

    def get_lineage_for_table(
        self, table_name: str, direction: str = "target"
    ) -> list[DataLineage]:
        """Get lineage records for a table."""
        column = _validate_identifier("target_table" if direction == "target" else "source_table")

        query = f"""
            SELECT * FROM meta_data_lineage
            WHERE {column} = :table_name
            ORDER BY created_at DESC
        """

        results = self.db.run_query(query, {"table_name": table_name})

        lineage_records = []
        for row in results:
            lineage = DataLineage(
                lineage_id=UUID(row["lineage_id"]),
                run_id=UUID(row["run_id"]) if row["run_id"] else None,
                source_table=row["source_table"],
                target_table=row["target_table"],
                transformation_name=row["transformation_name"],
                source_query=row["source_query"] or "",
                transformation_logic=row["transformation_logic"] or "",
                source_hash=row["source_hash"] or "",
                target_hash=row["target_hash"] or "",
                record_count_source=row["record_count_source"] or 0,
                record_count_target=row["record_count_target"] or 0,
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            lineage_records.append(lineage)

        return lineage_records

    def get_data_dependencies(self, table_name: str) -> list[str]:
        """Get upstream dependencies for a table."""
        query = """
            SELECT DISTINCT source_table
            FROM meta_data_lineage
            WHERE target_table = :table_name
        """
        results = self.db.run_query(query, {"table_name": table_name})
        return [r["source_table"] for r in results]

    def get_data_dependents(self, table_name: str) -> list[str]:
        """Get downstream dependents of a table."""
        query = """
            SELECT DISTINCT target_table
            FROM meta_data_lineage
            WHERE source_table = :table_name
        """
        results = self.db.run_query(query, {"table_name": table_name})
        return [r["target_table"] for r in results]

    def export_lineage(self, output_path: Path) -> None:
        """Export lineage to JSON file."""
        query = "SELECT * FROM meta_data_lineage ORDER BY created_at DESC"
        results = self.db.run_query(query)

        with open(output_path, "w") as f:
            json.dump([dict(r) for r in results], f, indent=2, default=str)

        logger.info(f"Lineage exported to {output_path}")


# Global tracker instance
_lineage_tracker: LineageTracker | None = None


def get_lineage_tracker() -> LineageTracker:
    """Get or create global lineage tracker."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = LineageTracker()
    return _lineage_tracker


class LineageContext:
    """Context manager for tracking lineage of a transformation."""

    def __init__(
        self,
        source_table: str,
        target_table: str,
        transformation_name: str,
        run_id: UUID | None = None,
    ):
        self.tracker = get_lineage_tracker()
        self.source_table = source_table
        self.target_table = target_table
        self.transformation_name = transformation_name
        self.run_id = run_id
        self.lineage_id: UUID | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success - record lineage
            self.lineage_id = self.tracker.record_lineage(
                source_table=self.source_table,
                target_table=self.target_table,
                transformation_name=self.transformation_name,
                run_id=self.run_id,
            )
        return False