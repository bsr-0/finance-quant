"""Command-line interface for the data pipeline."""

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pipeline.db import get_db_manager
from pipeline.dq.data_quality_monitor import DataQualityMonitor, Severity
from pipeline.dq.tests_sql import run_dq_tests
from pipeline.extract.fred import extract_fred
from pipeline.extract.factors_ff import extract_factors_ff
from pipeline.extract.gdelt import extract_gdelt
from pipeline.extract.polymarket import extract_polymarket
from pipeline.extract.prices_daily import extract_prices
from pipeline.extract.sec_fundamentals import extract_sec_fundamentals
from pipeline.extract.sec_insider import extract_sec_insider
from pipeline.extract.sec_13f import extract_sec_13f
from pipeline.extract.options_data import extract_options
from pipeline.extract.earnings import extract_earnings
from pipeline.extract.reddit_sentiment import extract_reddit_sentiment
from pipeline.extract.short_interest import extract_short_interest
from pipeline.extract.etf_flows import extract_etf_flows
from pipeline.load.raw_loader import RawLoader
from pipeline.logging_config import configure_logging
from pipeline.settings import get_settings
from pipeline.snapshot.orderbook_runner import OrderbookSnapshotRunner
from pipeline.snapshot.symbol_snapshots import SymbolSnapshotBuilder
from pipeline.transform.curated import CuratedTransformer
from pipeline.historical.latency import refresh_latency_stats

# Setup logging (respect LOG_FORMAT=json and LOG_LEVEL env vars)
configure_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    json_output=os.environ.get("LOG_FORMAT", "").lower() == "json",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Market Data Warehouse Pipeline CLI")
console = Console()


def get_git_sha() -> str | None:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def record_pipeline_run(pipeline_name: str, params: dict, status: str = "running") -> str:
    """Record pipeline run in meta table."""
    db = get_db_manager()
    run_id = str(uuid4())
    settings = get_settings()
    config = settings.model_dump() if hasattr(settings, "model_dump") else settings.dict()
    config_json = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    params = params or {}
    params["config_hash"] = config_hash
    params["config"] = json.loads(config_json)

    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("""
            INSERT INTO meta_pipeline_runs (run_id, pipeline_name, params, status, git_sha)
            VALUES (:run_id, :pipeline_name, :params, :status, :git_sha)
        """),
            {
                "run_id": run_id,
                "pipeline_name": pipeline_name,
                "params": params,
                "status": status,
                "git_sha": get_git_sha(),
            },
        )
        conn.commit()

    return run_id


def update_pipeline_run(
    run_id: str, status: str, row_counts: dict | None = None, errors: str | None = None
) -> None:
    """Update pipeline run status."""
    db = get_db_manager()

    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("""
            UPDATE meta_pipeline_runs
            SET status = :status,
                finished_at = NOW(),
                row_counts = :row_counts,
                errors = :errors
            WHERE run_id = :run_id
        """),
            {
                "run_id": run_id,
                "status": status,
                "row_counts": str(row_counts) if row_counts else None,
                "errors": errors,
            },
        )
        conn.commit()


@app.command()
def extract(
    source: str = typer.Argument(
        ...,
        help="Source to extract (fred, gdelt, polymarket, prices, factors, "
             "sec-fundamentals, sec-insider, sec-13f, options, earnings, "
             "reddit-sentiment, short-interest, etf-flows)",
    ),
    start: str | None = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),  # noqa: B008
    end: str | None = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),  # noqa: B008
    output_dir: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output directory"
    ),
):
    """Extract data from a source to raw lake."""
    settings = get_settings()
    raw_path = output_dir or settings.raw_lake_path

    # Record pipeline run
    run_id = record_pipeline_run(
        f"extract_{source}", {"source": source, "start": start, "end": end}
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
        elif source == "factors":
            files = extract_factors_ff(raw_path, run_id=run_id)
        elif source == "sec-fundamentals":
            files = extract_sec_fundamentals(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "sec-insider":
            files = extract_sec_insider(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "sec-13f":
            files = extract_sec_13f(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "options":
            files = extract_options(raw_path, run_id=run_id)
        elif source == "earnings":
            files = extract_earnings(raw_path, start_date=start, end_date=end, run_id=run_id)
        elif source == "reddit-sentiment":
            files = extract_reddit_sentiment(raw_path, run_id=run_id)
        elif source == "short-interest":
            files = extract_short_interest(raw_path, run_id=run_id)
        elif source == "etf-flows":
            files = extract_etf_flows(raw_path, run_id=run_id)
        else:
            raise typer.BadParameter(f"Unknown source: {source}")

        file_count = len(files) if isinstance(files, list) else sum(len(v) for v in files.values())
        update_pipeline_run(run_id, "success", {"files_created": file_count})

        console.print(f"[green]✓ Extracted {file_count} files from {source}[/green]")

    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Extraction failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def load_raw(
    source: str = typer.Argument(
        ...,
        help="Source to load (fred, gdelt, polymarket, prices, factors, "
             "sec_fundamentals, sec_insider, sec_13f, options, earnings, "
             "reddit_sentiment, short_interest, etf_flows)",
    ),
    raw_dir: Path | None = typer.Option(  # noqa: B008
        None, "--raw-dir", "-r", help="Raw data directory"
    ),
):
    """Load raw files into database raw tables."""
    settings = get_settings()
    raw_path = raw_dir or settings.raw_lake_path

    run_id = record_pipeline_run(f"load_raw_{source}", {"source": source, "raw_dir": str(raw_path)})

    try:
        loader = RawLoader()
        rows = loader.load_all_raw_files(raw_path, source, run_id=run_id)

        update_pipeline_run(run_id, "success", {"rows_loaded": rows})
        console.print(f"[green]✓ Loaded {rows} rows into raw_{source}* tables[/green]")

    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Load failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def transform_curated():
    """Transform raw data into curated tables."""
    run_id = record_pipeline_run("transform_curated", {})

    try:
        transformer = CuratedTransformer(run_id=run_id)
        results = transformer.transform_all()

        update_pipeline_run(run_id, "success", results)

        console.print("[green]✓ Transformed data into curated tables:[/green]")
        for table, count in results.items():
            console.print(f"  - {table}: {count} rows")

        # DQ gating: fail on any CRITICAL alert
        monitor = DataQualityMonitor()
        report = monitor.generate_quality_report()
        critical = [a for a in report.get("alerts", []) if a.get("severity") == Severity.CRITICAL.value]
        if critical:
            update_pipeline_run(run_id, "failed", errors="Critical data quality alerts")
            console.print(f"[red]✗ {len(critical)} CRITICAL data quality alerts detected[/red]")
            raise typer.Exit(1)

    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Transform failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def build_snapshots(
    contracts: list[str] | None = typer.Option(  # noqa: B008
        None, "--contract", "-c", help="Contract IDs"
    ),
    start: str | None = typer.Option(None, "--start", "-s", help="Start timestamp"),  # noqa: B008
    end: str | None = typer.Option(None, "--end", "-e", help="End timestamp"),  # noqa: B008
    freq: str = typer.Option(  # noqa: B008
        "1h", "--freq", "-f", help="Snapshot frequency (1h, 1d, 15min)"
    ),
):
    """Build training snapshots for contracts."""
    run_id = record_pipeline_run(
        "build_snapshots", {"contracts": contracts, "start": start, "end": end, "freq": freq}
    )

    try:
        from pipeline.snapshot.contract_snapshots import ContractSnapshotBuilder

        builder = ContractSnapshotBuilder()
        count = builder.build_snapshots_for_range(
            contract_ids=list(contracts) if contracts else None,
            start_ts=datetime.fromisoformat(start) if start else None,
            end_ts=datetime.fromisoformat(end) if end else None,
            frequency=freq,
        )

        update_pipeline_run(run_id, "success", {"snapshots_created": count})
        console.print(f"[green]✓ Built {count} snapshots[/green]")

    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Snapshot build failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def build_symbol_snapshots(
    symbols: list[str] | None = typer.Option(None, "--symbol", "-s", help="Symbol IDs"),  # noqa: B008
    start: str | None = typer.Option(None, "--start", help="Start timestamp"),  # noqa: B008
    end: str | None = typer.Option(None, "--end", help="End timestamp"),  # noqa: B008
    freq: str = typer.Option("1d", "--freq", help="Snapshot frequency (1h, 1d, 15min)"),  # noqa: B008
):
    """Build training snapshots for equity symbols."""
    run_id = record_pipeline_run(
        "build_symbol_snapshots", {"symbols": symbols, "start": start, "end": end, "freq": freq}
    )
    try:
        builder = SymbolSnapshotBuilder()
        count = builder.build_snapshots_for_range(
            symbol_ids=[UUID(s) for s in symbols] if symbols else None,
            start_ts=datetime.fromisoformat(start) if start else None,
            end_ts=datetime.fromisoformat(end) if end else None,
            frequency=freq,
        )
        update_pipeline_run(run_id, "success", {"snapshots_created": count})
        console.print(f"[green]✓ Built {count} symbol snapshots[/green]")
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Symbol snapshot build failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def orderbook_snapshots(
    interval: str | None = typer.Option(None, "--interval", "-i", help="Snapshot interval (e.g., 1m, 5m, 1h, off)"),  # noqa: B008
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of iterations (0 = forever)"),  # noqa: B008
    retention_days: int = typer.Option(30, "--retention-days", help="Retention window in days"),  # noqa: B008
    transform: bool = typer.Option(True, "--transform/--no-transform", help="Transform snapshots to curated"),  # noqa: B008
    max_markets: int | None = typer.Option(None, "--max-markets", help="Override market count"),  # noqa: B008
):
    """Capture Polymarket orderbook snapshots on a schedule."""
    settings = get_settings()
    interval = interval or settings.polymarket.orderbook_snapshot_freq
    if interval.lower() in {"off", "none", "0"}:
        console.print("[yellow]Orderbook snapshot interval is off; skipping.[/yellow]")
        return

    run_id = record_pipeline_run(
        "orderbook_snapshots",
        {
            "interval": interval,
            "iterations": iterations,
            "retention_days": retention_days,
            "transform": transform,
            "max_markets": max_markets,
        },
    )

    try:
        runner = OrderbookSnapshotRunner(run_id=run_id)
        count = runner.run(
            interval=interval,
            iterations=iterations,
            retention_days=retention_days,
            transform=transform,
            max_markets=max_markets,
        )
        update_pipeline_run(run_id, "success", {"snapshots_captured": count})
        console.print(f"[green]✓ Captured {count} orderbook snapshots[/green]")
    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Orderbook snapshots failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def dq():
    """Run data quality tests."""
    try:
        passed = run_dq_tests()
        if not passed:
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ DQ tests failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def latency_stats(
    window_days: int | None = typer.Option(None, "--days", "-d", help="Lookback window in days"),
):
    """Compute and store latency stats for conservative availability."""
    run_id = record_pipeline_run("latency_stats", {"window_days": window_days})
    try:
        results = refresh_latency_stats(window_days)
        update_pipeline_run(run_id, "success", {"sources": list(results.keys())})

        table = Table(title="Latency Stats (minutes)")
        table.add_column("Source", style="cyan")
        table.add_column("p50", style="green")
        table.add_column("p90", style="green")
        table.add_column("p95", style="yellow")
        table.add_column("mean", style="magenta")
        table.add_column("samples", style="blue")

        for source, metrics in results.items():
            table.add_row(
                source,
                f"{metrics.get('p50', 'n/a'):.2f}" if metrics.get("p50") is not None else "n/a",
                f"{metrics.get('p90', 'n/a'):.2f}" if metrics.get("p90") is not None else "n/a",
                f"{metrics.get('p95', 'n/a'):.2f}" if metrics.get("p95") is not None else "n/a",
                f"{metrics.get('mean', 'n/a'):.2f}" if metrics.get("mean") is not None else "n/a",
                f"{metrics.get('sample_size', 0)}",
            )

        console.print(table)

    except Exception as e:
        update_pipeline_run(run_id, "failed", errors=str(e))
        console.print(f"[red]✗ Latency stats failed: {e}[/red]")
        raise typer.Exit(1) from e


def _read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported file type: {path}")


@app.command()
def evaluate(
    scope: str = typer.Option("equity", "--scope", "-s", help="equity or prediction"),
    signals_path: Path | None = typer.Option(None, "--signals", help="Signals file path"),
    probs_path: Path | None = typer.Option(None, "--probs", help="Prediction market probs file path"),
    prices_path: Path | None = typer.Option(None, "--prices", help="Prices file path"),
    outcomes_path: Path | None = typer.Option(None, "--outcomes", help="Outcomes file path"),
    factors_from_db: bool = typer.Option(True, "--factors-from-db", help="Load factors from DB"),
    model_name: str = typer.Option("model", "--model-name"),
    dataset_id: str | None = typer.Option(None, "--dataset-id"),
):
    """Evaluate a model using institutional-grade rubric metrics."""
    settings = get_settings()
    run_id = str(uuid4())

    from pipeline.eval.evaluator import DatabaseResultStore, Evaluator
    from pipeline.eval.portfolio import ProbPortfolioConfig, SignalPortfolioConfig

    evaluator = Evaluator(cost_bps=settings.evaluation.cost_bps)
    store = DatabaseResultStore()

    factors_df = None
    if factors_from_db:
        db = get_db_manager()
        if db.table_exists("cur_factor_returns"):
            factors_df = pd.DataFrame(db.run_query("SELECT * FROM cur_factor_returns ORDER BY date"))
            if not factors_df.empty:
                factors_df["date"] = pd.to_datetime(factors_df["date"])
                factors_df = factors_df.set_index("date")[
                    ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
                ]

    if scope == "equity":
        if not signals_path or not prices_path:
            raise typer.BadParameter("Equity scope requires --signals and --prices")
        signals = _read_data(signals_path)
        prices = _read_data(prices_path)
        config = SignalPortfolioConfig()

        benchmark_prices = None
        if "symbol" in prices.columns and settings.evaluation.benchmark_symbol:
            bench = prices[prices["symbol"] == settings.evaluation.benchmark_symbol].copy()
            if not bench.empty:
                bench["date"] = pd.to_datetime(bench["date"])
                benchmark_prices = bench.set_index("date")["price"].sort_index()

        result = evaluator.evaluate_equity(
            signals=signals,
            prices=prices,
            factor_returns=factors_df,
            benchmark_prices=benchmark_prices,
            config=config,
        )
        eval_cfg = (
            settings.evaluation.model_dump()
            if hasattr(settings.evaluation, "model_dump")
            else settings.evaluation.dict()
        )
        store.write_results(run_id, model_name, scope, dataset_id, eval_cfg, result)
        console.print(f"[green]✓ Evaluation complete (run_id={run_id})[/green]")
    elif scope == "prediction":
        if not probs_path or not prices_path:
            raise typer.BadParameter("Prediction scope requires --probs and --prices")
        probs = _read_data(probs_path)
        prices = _read_data(prices_path)
        outcomes = _read_data(outcomes_path) if outcomes_path else None
        config = ProbPortfolioConfig(edge_threshold=settings.evaluation.edge_threshold)

        result = evaluator.evaluate_prediction_markets(
            probs=probs,
            prices=prices,
            outcomes=outcomes,
            factor_returns=factors_df,
            config=config,
        )
        eval_cfg = (
            settings.evaluation.model_dump()
            if hasattr(settings.evaluation, "model_dump")
            else settings.evaluation.dict()
        )
        store.write_results(run_id, model_name, scope, dataset_id, eval_cfg, result)
        console.print(f"[green]✓ Evaluation complete (run_id={run_id})[/green]")
    else:
        raise typer.BadParameter(f"Unknown scope: {scope}")


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
        ("cur_factor_returns", "date", "date"),
        ("snap_contract_features", "timestamp", "asof_ts"),
        ("snap_symbol_features", "timestamp", "asof_ts"),
        # New data sources
        ("cur_fundamentals_quarterly", "date", "fiscal_period_end"),
        ("cur_insider_trades", "date", "transaction_date"),
        ("cur_institutional_holdings", "date", "report_date"),
        ("cur_options_summary_daily", "date", "date"),
        ("cur_earnings_events", "date", "report_date"),
        ("cur_short_interest", "date", "settlement_date"),
        ("cur_etf_flows_daily", "date", "date"),
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

        min_date = (
            date_range["min_date"].isoformat() if date_range and date_range["min_date"] else "N/A"
        )
        max_date = (
            date_range["max_date"].isoformat() if date_range and date_range["max_date"] else "N/A"
        )

        # Get last updated from meta_pipeline_runs
        last_updated = "N/A"

        table.add_row(table_name, date_type, min_date, max_date, f"{row_count:,}", last_updated)

    console.print(table)


@app.command()
def init_db(
    ddl_dir: Path | None = typer.Option(None, "--ddl-dir", "-d", help="DDL directory"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Force re-initialization"),  # noqa: B008
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
        raise typer.Exit(1) from e


@app.command()
def run_pipeline(
    sources: list[str] = typer.Argument(default=["fred", "gdelt", "prices"]),  # noqa: B008
    start: str = typer.Option("2024-01-01", "--start", "-s"),
    end: str = typer.Option("2024-12-31", "--end", "-e"),
    skip_extract: bool = typer.Option(False, "--skip-extract"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
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
