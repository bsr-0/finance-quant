"""Database connection and utility functions.

Supports DuckDB (default, zero-config) and PostgreSQL backends.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# Allowlist pattern for SQL identifiers (table/column names)
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Validate and return a safe SQL identifier, or raise ValueError."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


class DatabaseManager:
    """Manages database connections and schema operations."""

    def __init__(self, connection_string: str | None = None):
        settings = get_settings()
        self.connection_string = connection_string or settings.database.connection_string
        self.backend = settings.database.backend
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            if self.backend == "duckdb":
                # Ensure parent directory exists for DuckDB file
                db_path = get_settings().database.path
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self._engine = create_engine(self.connection_string, echo=False)
            else:
                self._engine = create_engine(
                    self.connection_string, pool_pre_ping=True, pool_recycle=300, echo=False
                )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_sql_file(self, file_path: Path) -> None:
        """Execute SQL file against the database."""
        with open(file_path) as f:
            sql = f.read()

        with self.engine.connect() as conn:
            # DuckDB needs statements executed individually
            if self.backend == "duckdb":
                for statement in self._split_sql(sql):
                    if statement.strip():
                        conn.execute(text(statement))
            else:
                conn.execute(text(sql))
            conn.commit()
            logger.info(f"Executed SQL file: {file_path}")

    @staticmethod
    def _split_sql(sql: str) -> list[str]:
        """Split SQL into individual statements, respecting $$ blocks."""
        statements = []
        current = []
        in_dollar_block = False

        for line in sql.split("\n"):
            stripped = line.strip()

            # Track $$ delimited blocks (PL/pgSQL etc.)
            if "$$" in stripped:
                in_dollar_block = not in_dollar_block

            current.append(line)

            if not in_dollar_block and stripped.endswith(";"):
                stmt = "\n".join(current).strip()
                if stmt and not stmt.startswith("--"):
                    statements.append(stmt)
                current = []

        # Leftover
        if current:
            stmt = "\n".join(current).strip()
            if stmt and not stmt.startswith("--"):
                statements.append(stmt)

        return statements

    def init_schema(self, ddl_dir: Path) -> None:
        """Initialize database schema from DDL files.

        Picks the right DDL directory based on backend:
        - DuckDB: uses ddl_dir/../ddl_duckdb/ if it exists, else ddl_dir
        - PostgreSQL: uses ddl_dir as-is
        """
        if self.backend == "duckdb":
            duckdb_dir = ddl_dir.parent / "ddl_duckdb"
            if duckdb_dir.exists():
                ddl_dir = duckdb_dir
        ddl_files = sorted(ddl_dir.glob("*.sql"))
        for sql_file in ddl_files:
            self.execute_sql_file(sql_file)
        logger.info(f"Initialized schema from {len(ddl_files)} DDL files")

    def run_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute query and return results as list of dicts."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        table_name = _validate_identifier(table_name)
        result = self.run_query(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return result[0]["cnt"] if result else 0

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = :table_name
            ) as exists
        """
        if self.backend == "duckdb":
            query = """
                SELECT COUNT(*) > 0 as exists
                FROM information_schema.tables
                WHERE table_name = :table_name
            """
        result = self.run_query(query, {"table_name": table_name})
        return result[0]["exists"] if result else False

    def get_min_max_dates(self, table_name: str, date_column: str) -> dict | None:
        """Get min and max dates from a table."""
        try:
            table_name = _validate_identifier(table_name)
            date_column = _validate_identifier(date_column)
            query = f"""
                SELECT
                    MIN({date_column}) as min_date,
                    MAX({date_column}) as max_date
                FROM {table_name}
            """
            result = self.run_query(query)
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Could not get date range for {table_name}: {e}")
            return None


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db_manager() -> None:
    """Reset global database manager (useful for testing)."""
    global _db_manager
    _db_manager = None
