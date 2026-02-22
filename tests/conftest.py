"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

# Use test database
os.environ["DB_NAME"] = "market_data_test"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def fresh_db():
    """Provide a fresh database for each test."""
    from pipeline.db import DatabaseManager, reset_db_manager
    
    reset_db_manager()
    db = DatabaseManager()
    
    # Drop and recreate schema
    with db.engine.connect() as conn:
        from sqlalchemy import text
        
        # Drop all tables
        conn.execute(text("""
            DO $$
            DECLARE
                r RECORD;
            BEGIN
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
                LOOP
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                END LOOP;
            END $$;
        """))
        conn.commit()
    
    # Re-initialize schema
    ddl_dir = Path("src/sql/ddl")
    if ddl_dir.exists():
        db.init_schema(ddl_dir)
    
    yield db
    
    reset_db_manager()
