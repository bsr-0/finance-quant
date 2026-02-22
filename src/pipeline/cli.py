"""Command-line interface for the data pipeline."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from pipeline.db import DatabaseManager, get_db_manager
from pipeline.dq.tests_sql import DataQualityTests, run_dq_tests
from pipeline.extract.fred import extract_fred
from pipeline.extract.gdelt import extract_gdelt
from pipeline.extract.polymarket import extract_polymarket
from pipeline.extract.prices_daily import extract_prices
from pipeline.load.raw_loader import RawLoader
from pipeline.settings import get_settings
from pipeline.snapshot.contract_snapshots import build_snapshots
from pipeline.transform.curated import CuratedTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Market Data Warehouse Pipeline CLI")
console = Console()


def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def record_pipeline_run(
    pipeline_name: str,
    params: dict,
    status: str = "running"
) -> str:
    """Record pipeline run in meta table."""
    db = get_db_manager()
    run_id = str(uuid4())
    
    with db.engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text("""
            INSERT INTO meta_pipeline_runs (run_id, pipeline_name, params, status, git_sha)
            VALUES (:run_id, :pipeline_name, :params, :status, :git_sha)
        """), {
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "params": str(params),
            "status": status,
            "git_sha": get_git_sha()
        })
        conn.commit()
    
    return run_id


def update_pipeline_run(run_id: str, status: str, row_counts: Optional[dict] = None, errors: Optional[str] = None) -> None:
    """Update pipeline run status."""
    db = get_db_manager()
    
    with db.engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text("""
            UPDATE meta_pipeline_runs
            SET status = :status,
                finished_at = NOW(),
                row_counts = :row_counts,
                errors = :errors
            WHERE run_id = :run_id
        """), {
            "run_id": run_id,
            "status": status,
            "row_counts": str(row_counts) if row_counts else None,
            "errors": errors
        })
        conn.commit()


@app.command()
def extract(
    source: str = typer.Argument(..., help="Source to extract (fred, gdelt, polymarket, prices)"),
    start: Optional[str] = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory")
):
    """Extract data from a source to raw lake."""
    settings = get_settings()
    raw_path = output_dir or settings.raw_lake_path
    
    # Record pipeline run
    run_id = record_pipeline_run(
        f"extract_{source}",
        {"source": source, "start": start, "end": end}
    )
    
    try:
        if source == "fred":
            files = extract_fred(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "gdelt":
            if not start or not end:
                raise typer.BadParameter("GDELT extraction requires --start and --end dates")
            files = extract_gdelt(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "polymarket":
            files = extract_polymarket(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "prices":
            files = extract_prices(raw_path, start_date=start, end_date=end, run_id=run_id)
        else:
            raise typer.BadParameter(f"Unknown source: {source}")
        
        file_count = len(files) if isinstance(files, list) else sum(len(v) for v in files.values())
        update_pipeline_run(run_id, "success", {"files_created": file_count})
        
        console.print(f"[green]✓ Extracted {file_count} files from {source}[/green]")
        
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Extraction failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def load_raw(
    source: str = typer.Argument(..., help="Source to load (fred, gdelt, polymarket, prices)"),
    raw_dir: Optional[Path] = typer.Option(None, "--raw-dir", "-r", help="Raw data directory")
):
    """Load raw files into database raw tables."""
    settings = get_settings()
    raw_path = raw_dir or settings.raw_lake_path
    
    run_id = record_pipeline_run(
        f"load_raw_{source}",
        {"source": source, "raw_dir": str(raw_path)}
    )
    
    try:
        loader = RawLoader()
        rows = loader.load_all_raw_files(raw_path, source, run_id=run_id)
        
        update_pipeline_run(run_id, "success", {"rows_loaded": rows})
        console.print(f"[green]✓ Loaded {rows} rows into raw_{source}* tables[/green]")
        
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Load failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def transform_curated():
    """Transform raw data into curated tables."""
    run_id = record_pipeline_run("transform_curated", {})
    
    try:
        transformer = CuratedTransformer()
        results = transformer.transform_all()
        
        update_pipeline_run(run_id, "success", results)
        
        console.print("[green]✓ Transformed data into curated tables:[/green]")
        for table, count in results.items():
            console.print(f"  - {table}: {count} rows")
        
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Transform failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def build_snapshots(
    contracts: Optional[List[str]] = typer.Option(None, "--contract", "-c", help="Contract IDs"),
    start: Optional[str] = typer.Option(None, "--start", "-s", help="Start timestamp"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End timestamp"),
    freq: str = typer.Option("1h", "--freq", "-f", help="Snapshot frequency (1h, 1d, 15min)")
):
    """Build training snapshots for contracts."""
    run_id = record_pipeline_run(
        "build_snapshots",
        {"contracts": contracts, "start": start, "end": end, "freq": freq}
    )
    
    try:
        from pipeline.snapshot.contract_snapshots import ContractSnapshotBuilder
        
        builder = ContractSnapshotBuilder()
        count = builder.build_snapshots_for_range(
            contract_ids=[c for c in contracts] if contracts else None,
            start_ts=datetime.fromisoformat(start) if start else None,
            end_ts=datetime.fromisoformat(end) if end else None,
            frequency=freq
        )
        
        update_pipeline_run(run_id, "success", {"snapshots_created": count})
        console.print(f"[green]✓ Built {count} snapshots[/green]")
        
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Snapshot build failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def dq():
    """Run data quality tests."""
    try:
        passed = run_dq_tests()
        if not passed:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ DQ tests failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def inventory():
    """Print data inventory report."""
    db = get_db_manager()
    
    # Define tables to report on
    tables = [
        ("dim_source", "date", None),
        ("dim_symbol", "date", None),
        ("dim_contract", "date", "created_time"),
        ("dim_macro_series", "date", None),
        ("cur_prices_ohlcv_daily", "date", "date"),
        ("cur_macro_observations", "date", "period_end"),
        ("cur_world_events", "date", "event_time"),
        ("cur_contract_prices", "timestamp", "ts"),
        ("cur_contract_trades", "timestamp", "ts"),
        ("snap_contract_features", "timestamp", "asof_ts"),
    ]
    
    table = Table(title="Data Inventory")
    table.add_column("Table", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Min Date", style="green")
    table.add_column("Max Date", style="green")
    table.add_column("Row Count", justify="right", style="yellow")
    table.add_column("Last Updated", style="blue")
    
    for table_name, date_type, date_col in tables:
        if not db.table_exists(table_name):
            continue
        
        row_count = db.get_table_count(table_name)
        
        date_range = None
        if date_col:
            date_range = db.get_min_max_dates(table_name, date_col)
        
        min_date = date_range["min_date"].isoformat() if date_range and date_range["min_date"] else "N/A"
        max_date = date_range["max_date"].isoformat() if date_range and date_range["max_date"] else "N/A"
        
        # Get last updated from meta_pipeline_runs
        last_updated = "N/A"
        
        table.add_row(
            table_name,
            date_type,
            min_date,
            max_date,
            f"{row_count:,}",
            last_updated
        )
    
    console.print(table)


@app.command()
def init_db(
    ddl_dir: Optional[Path] = typer.Option(None, "--ddl-dir", "-d", help="DDL directory"),
    force: bool = typer.Option(False, "--force", help="Force re-initialization")
):
    """Initialize database schema."""
    if ddl_dir is None:
        ddl_dir = Path("src/sql/ddl")
    
    db = get_db_manager()
    
    try:
        db.init_schema(ddl_dir)
        console.print(f"[green]✓ Initialized database schema from {ddl_dir}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Schema initialization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run_pipeline(
    sources: List[str] = typer.Argument(default=["fred", "gdelt", "prices"]),
    start: str = typer.Option("2024-01-01", "--start", "-s"),
    end: str = typer.Option("2024-12-31", "--end", "-e"),
    skip_extract: bool = typer.Option(False, "--skip-extract"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots")
):
    """Run full pipeline for specified sources."""
    console.print("[bold blue]Running full pipeline...[/bold blue]")
    
    if not skip_extract:
        for source in sources:
            console.print(f"\n[bold]Extracting {source}...[/bold]")
            extract(source, start=start, end=end)
            load_raw(source)
    
    console.print("\n[bold]Transforming to curated...[/bold]")
    transform_curated()
    
    if not skip_snapshots:
        console.print("\n[bold]Building snapshots...[/bold]")
        build_snapshots(start=start, end=end)
    
    console.print("\n[bold]Running data quality tests...[/bold]")
    dq()
    
    console.print("\n[bold]Data inventory:[/bold]")
    inventory()
    
    console.print("\n[green bold]✓ Pipeline completed successfully![/green bold]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
