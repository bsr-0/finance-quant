"""Command-line interface for the data pipeline."""

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pipeline.db import get_db_manager
from pipeline.dq.data_quality_monitor import DataQualityMonitor, Severity
from pipeline.dq.tests_sql import run_dq_tests
from pipeline.extract.earnings import extract_earnings
from pipeline.extract.etf_flows import extract_etf_flows
from pipeline.extract.factors_ff import extract_factors_ff
from pipeline.extract.fred import extract_fred
from pipeline.extract.gdelt import extract_gdelt
from pipeline.extract.options_data import extract_options
from pipeline.extract.polymarket import extract_polymarket
from pipeline.extract.prices_daily import extract_prices
from pipeline.extract.reddit_sentiment import extract_reddit_sentiment
from pipeline.extract.sec_13f import extract_sec_13f
from pipeline.extract.sec_fundamentals import extract_sec_fundamentals
from pipeline.extract.sec_insider import extract_sec_insider
from pipeline.extract.short_interest import extract_short_interest
from pipeline.historical.latency import refresh_latency_stats
from pipeline.load.raw_loader import RawLoader
from pipeline.logging_config import configure_logging
from pipeline.settings import get_settings
from pipeline.snapshot.orderbook_runner import OrderbookSnapshotRunner
from pipeline.snapshot.symbol_snapshots import SymbolSnapshotBuilder
from pipeline.transform.curated import CuratedTransformer

# Setup logging (respect LOG_FORMAT=json and LOG_LEVEL env vars)
configure_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    json_output=os.environ.get("LOG_FORMAT", "").lower() == "json",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Market Data Warehouse Pipeline CLI")
console = Console()


def _validate_range(min_val=None, max_val=None):
    """Create a Typer callback that enforces numeric bounds."""

    def _check(value):
        if value is None:
            return value
        if min_val is not None and value < min_val:
            raise typer.BadParameter(f"Must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise typer.BadParameter(f"Must be <= {max_val}, got {value}")
        return value

    return _check


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
                "params": json.dumps(params, default=str),
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
                "row_counts": json.dumps(row_counts, default=str) if row_counts else None,
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
    start: str | None = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),  # noqa: B008
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
        files: list[Path] | dict[str, list[Path]]
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
            files = extract_sec_fundamentals(
                raw_path, start_date=start, end_date=end, run_id=run_id
            )
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
        rows = loader.load_all_raw_files(raw_path, source, run_id=UUID(run_id))

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
        critical = [
            a for a in report.get("alerts", []) if a.get("severity") == Severity.CRITICAL.value
        ]
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
            contract_ids=[UUID(c) for c in contracts] if contracts else None,
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
    symbols: list[str] | None = typer.Option(  # noqa: B008
        None, "--symbol", "-s", help="Symbol IDs"
    ),
    start: str | None = typer.Option(None, "--start", help="Start timestamp"),  # noqa: B008
    end: str | None = typer.Option(None, "--end", help="End timestamp"),  # noqa: B008
    freq: str = typer.Option(  # noqa: B008
        "1d", "--freq", help="Snapshot frequency (1h, 1d, 15min)"
    ),
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
    interval: str | None = typer.Option(
        None, "--interval", "-i", help="Snapshot interval (e.g., 1m, 5m, 1h, off)"
    ),  # noqa: B008
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-n",
        help="Number of iterations (0 = forever)",
        callback=_validate_range(min_val=0),
    ),  # noqa: B008
    retention_days: int = typer.Option(
        30,
        "--retention-days",
        help="Retention window in days",
        callback=_validate_range(min_val=1),
    ),  # noqa: B008
    transform: bool = typer.Option(
        True, "--transform/--no-transform", help="Transform snapshots to curated"
    ),  # noqa: B008
    max_markets: int | None = typer.Option(
        None,
        "--max-markets",
        help="Override market count",
        callback=_validate_range(min_val=1),
    ),  # noqa: B008
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
    scope: str = typer.Option("equity", "--scope", "-s", help="equity or prediction"),  # noqa: B008
    signals_path: Path | None = typer.Option(  # noqa: B008
        None, "--signals", help="Signals file path"
    ),
    probs_path: Path | None = typer.Option(  # noqa: B008
        None, "--probs", help="Prediction market probs file path"
    ),
    prices_path: Path | None = typer.Option(  # noqa: B008
        None, "--prices", help="Prices file path"
    ),
    outcomes_path: Path | None = typer.Option(  # noqa: B008
        None, "--outcomes", help="Outcomes file path"
    ),
    factors_from_db: bool = typer.Option(
        True, "--factors-from-db", help="Load factors from DB"
    ),  # noqa: B008
    model_name: str = typer.Option("model", "--model-name"),  # noqa: B008
    dataset_id: str | None = typer.Option(None, "--dataset-id"),  # noqa: B008
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
            factors_df = pd.DataFrame(
                db.run_query("SELECT * FROM cur_factor_returns ORDER BY date")
            )
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
        prob_config = ProbPortfolioConfig(edge_threshold=settings.evaluation.edge_threshold)

        result = evaluator.evaluate_prediction_markets(
            probs=probs,
            prices=prices,
            outcomes=outcomes,
            factor_returns=factors_df,
            config=prob_config,
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
    ddl_dir: Path | None = typer.Option(  # noqa: B008
        None, "--ddl-dir", "-d", help="DDL directory"
    ),
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


@app.command()
def generate_signals(
    date: str | None = typer.Option(  # noqa: B008
        None, "--date", "-d", help="Signal date (YYYY-MM-DD). Default: latest in data."
    ),
    prices_dir: Path | None = typer.Option(  # noqa: B008
        None, "--prices-dir", help="Directory with per-ticker CSV/parquet files"
    ),
    spy_path: Path | None = typer.Option(  # noqa: B008
        None, "--spy", help="SPY prices CSV/parquet for regime classification"
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("data/signals"), "--output", "-o", help="Output directory for signal CSV"
    ),
    threshold: int = typer.Option(  # noqa: B008
        60,
        "--threshold",
        "-t",
        help="Minimum signal score",
        callback=_validate_range(min_val=0, max_val=100),
    ),
    min_volume: float = typer.Option(  # noqa: B008
        50_000,
        "--min-volume",
        help="Minimum average daily volume",
        callback=_validate_range(min_val=0),
    ),
):
    """Generate trading signals for the current universe.

    Loads price data, computes indicators and composite signal scores,
    runs pre-trade checks, and outputs a standardized signal CSV.

    Examples:
        pipeline generate-signals --prices-dir data/prices/ --output data/signals/
        pipeline generate-signals -d 2024-12-31 --prices-dir data/prices/
    """
    from pipeline.strategy.pre_trade_checks import filter_signals
    from pipeline.strategy.signal_output import format_signals, write_signal_csv
    from pipeline.strategy.signals import SignalEngine, compute_indicators

    console.print("[bold blue]Generating trading signals...[/bold blue]")

    # Load price data
    if prices_dir is None:
        console.print(
            "[red]--prices-dir is required (directory of per-ticker CSV/parquet files)[/red]"
        )
        raise typer.Exit(1)

    if not prices_dir.exists():
        console.print(f"[red]Prices directory not found: {prices_dir}[/red]")
        raise typer.Exit(1)

    price_data: dict[str, pd.DataFrame] = {}
    for f in sorted(prices_dir.iterdir()):
        if f.suffix.lower() in {".csv", ".parquet", ".pq"}:
            ticker = f.stem.upper()
            df = _read_data(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            price_data[ticker] = df

    if not price_data:
        console.print(f"[red]No price files found in {prices_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(price_data)} tickers")

    # SPY for regime classification
    spy_prices = None
    if spy_path and spy_path.exists():
        spy_df = _read_data(spy_path)
        if "date" in spy_df.columns:
            spy_df["date"] = pd.to_datetime(spy_df["date"])
            spy_df = spy_df.set_index("date").sort_index()
        spy_prices = spy_df["close"]
    elif "SPY" in price_data:
        spy_prices = price_data["SPY"]["close"]

    # Compute indicators
    indicator_data: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        if df.empty:
            continue
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            logger.warning("Skipping %s: missing required columns", ticker)
            continue
        indicator_data[ticker] = compute_indicators(df)

    # Determine signal date
    if date:
        signal_date = pd.Timestamp(date)
    else:
        all_dates = set()
        for df in indicator_data.values():
            if not df.empty:
                all_dates.add(df.index[-1])
        if not all_dates:
            console.print("[red]No data available for signal generation[/red]")
            raise typer.Exit(1)
        signal_date = max(all_dates)

    console.print(f"  Signal date: {signal_date.date()}")

    # Score universe
    engine = SignalEngine(entry_threshold=threshold)
    scores = engine.score_universe(indicator_data, spy_prices=spy_prices, date=signal_date)
    eligible = [s for s in scores if s.entry_eligible]

    console.print(
        f"  Scored {len(scores)} symbols, {len(eligible)} eligible (score >= {threshold})"
    )

    # Pre-trade checks
    passed_signals, check_results = filter_signals(
        signals=eligible,
        price_data=indicator_data,
        min_volume=min_volume,
    )

    rejected = len(eligible) - len(passed_signals)
    if rejected > 0:
        console.print(f"  Pre-trade checks rejected {rejected} signals")

    # Format and write output
    signals_df = format_signals(
        scores=passed_signals,
        price_data=indicator_data,
        date=signal_date,
    )

    if signals_df.empty:
        console.print("[yellow]No actionable signals for this date.[/yellow]")
    else:
        filepath = write_signal_csv(signals_df, output_dir, signal_date)
        console.print(f"[green]✓ Wrote {len(signals_df)} signals to {filepath}[/green]")

        # Print summary table
        sig_table = Table(title=f"Signals for {signal_date.date()}")
        sig_table.add_column("Ticker", style="cyan")
        sig_table.add_column("Score", justify="right", style="green")
        sig_table.add_column("Confidence", style="yellow")
        sig_table.add_column("Entry", justify="right")
        sig_table.add_column("Stop", justify="right", style="red")
        sig_table.add_column("Target", justify="right", style="green")
        sig_table.add_column("Regime", style="magenta")

        for _, row in signals_df.iterrows():
            sig_table.add_row(
                str(row["ticker"]),
                str(row["score"]),
                str(row["confidence"]),
                f"${row['entry_price']:.2f}",
                f"${row['stop_price']:.2f}",
                f"${row['target_1']:.2f}",
                str(row["regime"]),
            )
        console.print(sig_table)


@app.command()
def execute_signals(
    signal_csv: Path = typer.Argument(  # noqa: B008
        ..., help="Path to signal CSV file from generate-signals"
    ),
    max_capital: float = typer.Option(
        300.0,
        "--max-capital",
        help="Maximum capital to deploy ($)",
        callback=_validate_range(min_val=0),
    ),
    max_positions: int = typer.Option(
        2,
        "--max-positions",
        help="Maximum simultaneous positions",
        callback=_validate_range(min_val=1),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate everything but don't submit orders"
    ),
    paper: bool = typer.Option(True, "--paper/--live", help="Use paper trading (default) or live"),
):
    """Execute trading signals through Alpaca broker.

    Reads a signal CSV (from generate-signals) and submits orders to
    Alpaca with full QAQC capital guards.  Default mode is paper trading.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.

    Examples:
        # Paper trading (safe):
        pipeline execute-signals data/signals/signals_20250306.csv --dry-run
        pipeline execute-signals data/signals/signals_20250306.csv

        # Live trading (real money):
        pipeline execute-signals data/signals/signals_20250306.csv --live --max-capital 200
    """
    if not signal_csv.exists():
        console.print(f"[red]Signal CSV not found: {signal_csv}[/red]")
        raise typer.Exit(1)

    if not paper and not dry_run:
        console.print("[bold red]*** LIVE TRADING MODE — Real money at risk ***[/bold red]")
        console.print(f"  Max capital: ${max_capital:.2f}")
        console.print(f"  Max positions: {max_positions}")
        confirm = typer.confirm("Are you sure you want to trade with real money?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    # Set base URL based on mode
    if paper:
        os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    else:
        os.environ.setdefault("ALPACA_BASE_URL", "https://api.alpaca.markets")

    try:
        from pipeline.execution.alpaca_broker import AlpacaBroker
        from pipeline.execution.runner import RunnerConfig, TradingRunner

        broker = AlpacaBroker.from_env()
        config = RunnerConfig(
            max_capital=max_capital,
            max_positions=max_positions,
            paper_mode=paper,
            dry_run=dry_run,
        )
        runner = TradingRunner(broker=broker, config=config)

        mode_label = "DRY RUN" if dry_run else ("PAPER" if paper else "LIVE")
        console.print(f"[bold blue]Executing signals in {mode_label} mode...[/bold blue]")

        # Show account status
        status = runner.status()
        status_table = Table(title="Account Status")
        status_table.add_column("Field", style="cyan")
        status_table.add_column("Value", justify="right")
        status_table.add_row("Mode", mode_label)
        status_table.add_row("Equity", f"${status.get('account_equity', 0):.2f}")
        status_table.add_row("Cash", f"${status.get('account_cash', 0):.2f}")
        status_table.add_row("Buying Power", f"${status.get('buying_power', 0):.2f}")
        status_table.add_row("Open Positions", str(status.get("positions_count", 0)))
        status_table.add_row("Max Capital", f"${max_capital:.2f}")
        status_table.add_row("Margin Account", "YES ⚠️" if status.get("is_margin") else "NO ✓")
        console.print(status_table)

        # Execute
        result = runner.run_daily(signal_csv)

        # Show results
        result_table = Table(title="Execution Results")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", justify="right")
        result_table.add_row("Signals Parsed", str(result.signals_parsed))
        result_table.add_row("Signals Eligible", str(result.signals_eligible))
        result_table.add_row("Orders Submitted", str(result.orders_submitted))
        result_table.add_row("Orders Filled", str(result.orders_filled))
        result_table.add_row("Orders Rejected", str(result.orders_rejected))
        result_table.add_row("Guard Rejections", str(result.guard_rejections))
        console.print(result_table)

        if result.details:
            detail_table = Table(title="Order Details")
            detail_table.add_column("Ticker", style="cyan")
            detail_table.add_column("Action", style="yellow")
            detail_table.add_column("Details")
            for d in result.details:
                ticker = d.get("ticker", "—")
                action = d.get("action", "—")
                extra = ""
                if "shares" in d:
                    extra = (
                        f"{d['shares']:.4f} shares @ ${d.get('limit_price', d.get('price', 0)):.2f}"
                    )
                elif "reason" in d:
                    extra = d["reason"]
                elif "summary" in d:
                    extra = d["summary"]
                detail_table.add_row(ticker, action, extra)
            console.print(detail_table)

        console.print(f"\n[green bold]✓ {result.summary()}[/green bold]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install alpaca-py[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Execution failed: {e}[/red]")
        logger.exception("Signal execution failed")
        raise typer.Exit(1) from None


@app.command()
def trading_status(
    paper: bool = typer.Option(True, "--paper/--live", help="Check paper or live account"),
):
    """Show current Alpaca account and position status.

    Examples:
        pipeline trading-status
        pipeline trading-status --live
    """
    if paper:
        os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    else:
        os.environ.setdefault("ALPACA_BASE_URL", "https://api.alpaca.markets")

    try:
        from pipeline.execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker.from_env()
        account = broker.get_account_snapshot()
        positions = broker.get_positions()

        mode = "PAPER" if paper else "LIVE"
        acct_table = Table(title=f"Alpaca Account ({mode})")
        acct_table.add_column("Field", style="cyan")
        acct_table.add_column("Value", justify="right")
        acct_table.add_row("Equity", f"${account.equity:.2f}")
        acct_table.add_row("Cash", f"${account.cash:.2f}")
        acct_table.add_row("Buying Power", f"${account.buying_power:.2f}")
        acct_table.add_row("Positions Value", f"${account.positions_market_value:.2f}")
        acct_table.add_row("Position Count", str(account.position_count))
        acct_table.add_row("Margin Account", "YES ⚠️" if account.is_margin_account else "NO ✓")
        console.print(acct_table)

        if positions:
            pos_table = Table(title="Open Positions")
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Avg Entry", justify="right")
            pos_table.add_column("Current", justify="right")
            pos_table.add_column("P&L", justify="right")
            pos_table.add_column("Side")

            for p in positions:
                pnl_style = "green" if p.unrealised_pnl >= 0 else "red"
                pos_table.add_row(
                    p.symbol,
                    f"{p.qty:.4f}",
                    f"${p.avg_entry_price:.2f}",
                    f"${p.current_price:.2f}",
                    f"[{pnl_style}]${p.unrealised_pnl:.2f}[/{pnl_style}]",
                    p.side,
                )
            console.print(pos_table)
        else:
            console.print("[dim]No open positions[/dim]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install alpaca-py[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Failed to fetch status: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def monitor_prices(
    symbols: list[str] | None = typer.Option(  # noqa: B008
        None, "--symbol", "-s", help="Symbols to monitor"
    ),
    mode: str = typer.Option("websocket", "--mode", "-m", help="Feed mode: websocket or polling"),
    interval: int = typer.Option(
        5,
        "--interval",
        "-i",
        help="Display refresh interval (seconds)",
        callback=_validate_range(min_val=1),
    ),
    duration: int = typer.Option(
        0,
        "--duration",
        "-d",
        help="Run for N seconds (0 = until Ctrl-C)",
        callback=_validate_range(min_val=0),
    ),
    paper: bool = typer.Option(True, "--paper/--live", help="Use paper or live API keys"),
):
    """Monitor real-time prices via Alpaca WebSocket or polling.

    Streams live prices for the configured universe (or specified symbols)
    and displays them in a refreshing table.  Useful for verifying feed
    connectivity and observing intraday stop levels.

    Examples:
        pipeline monitor-prices
        pipeline monitor-prices -s AAPL -s MSFT --mode polling
        pipeline monitor-prices --duration 60
    """
    import time as _time

    if paper:
        os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    try:
        from pipeline.execution.realtime_feed import RealtimePriceFeed

        feed = RealtimePriceFeed.from_env(symbols=symbols)

        # Override mode if specified
        if mode != feed._mode:
            feed._mode = mode

        console.print(
            f"[bold blue]Starting real-time price monitor "
            f"(mode={feed._mode}, symbols={len(feed.symbols)})[/bold blue]"
        )
        console.print(
            f"  Symbols: {', '.join(feed.symbols[:10])}"
            + (f" ... +{len(feed.symbols) - 10} more" if len(feed.symbols) > 10 else "")
        )
        console.print("  Press Ctrl-C to stop\n")

        feed.start()

        # Wait briefly for initial data
        _time.sleep(min(interval, 3))

        start_time = _time.time()

        try:
            while True:
                quotes = feed.get_all_latest()

                table = Table(title=f"Live Prices ({datetime.now().strftime('%H:%M:%S')})")
                table.add_column("Symbol", style="cyan")
                table.add_column("Price", justify="right")
                table.add_column("Bid", justify="right", style="dim")
                table.add_column("Ask", justify="right", style="dim")
                table.add_column("High", justify="right", style="green")
                table.add_column("Low", justify="right", style="red")
                table.add_column("Age", justify="right")
                table.add_column("Source", style="dim")

                for sym in sorted(feed.symbols):
                    q = quotes.get(sym)
                    if q:
                        age = f"{q.age_seconds:.0f}s"
                        age_style = (
                            "green"
                            if q.age_seconds < 60
                            else ("yellow" if q.age_seconds < 120 else "red")
                        )
                        table.add_row(
                            sym,
                            f"${q.price:.2f}",
                            f"${q.bid:.2f}" if q.bid > 0 else "—",
                            f"${q.ask:.2f}" if q.ask > 0 else "—",
                            f"${q.high:.2f}" if q.high > 0 else "—",
                            f"${q.low:.2f}" if q.low > 0 else "—",
                            f"[{age_style}]{age}[/{age_style}]",
                            q.source.replace("alpaca_", ""),
                        )
                    else:
                        table.add_row(sym, "—", "—", "—", "—", "—", "[red]no data[/red]", "")

                console.print(table)

                if duration > 0 and (_time.time() - start_time) >= duration:
                    break

                _time.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")

        feed.stop()
        console.print("[green]Feed stopped.[/green]")

    except Exception as e:
        console.print(f"[red]Monitor failed: {e}[/red]")
        logger.exception("Price monitor failed")
        raise typer.Exit(1) from None


@app.command()
def monitor_positions(
    poll_seconds: int = typer.Option(
        60,
        "--poll",
        "-p",
        help="Check interval in seconds",
        callback=_validate_range(min_val=1),
    ),
    duration: int = typer.Option(
        0,
        "--duration",
        "-d",
        help="Run for N seconds (0 = until Ctrl-C)",
        callback=_validate_range(min_val=0),
    ),
    paper: bool = typer.Option(True, "--paper/--live", help="Use paper or live keys"),
    realtime: bool = typer.Option(
        True, "--realtime/--no-realtime", help="Use real-time prices for stop checks"
    ),
):
    """Continuously monitor open positions with real-time stop enforcement.

    Combines the existing PositionMonitor with the real-time price feed.
    Checks exit conditions (stops, trailing stops, profit targets, time exits)
    using live intraday prices instead of stale closing prices.

    Examples:
        pipeline monitor-positions --poll 30
        pipeline monitor-positions --no-realtime  # use broker prices only
    """
    import time as _time

    if paper:
        os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    try:
        from pipeline.execution.alpaca_broker import AlpacaBroker
        from pipeline.execution.capital_guard import CapitalGuardConfig
        from pipeline.execution.position_monitor import PositionMonitor
        from pipeline.execution.realtime_feed import RealtimePriceFeed

        broker = AlpacaBroker.from_env()

        # Build feed if requested
        rt_feed = None
        if realtime:
            positions = broker.get_positions()
            if positions:
                syms = [p.symbol for p in positions]
                rt_feed = RealtimePriceFeed.create_for_positions(syms)
                rt_feed.start()
                console.print(
                    f"[bold blue]Real-time feed started for {len(syms)} positions[/bold blue]"
                )
            else:
                console.print("[yellow]No open positions — realtime feed not started[/yellow]")

        settings = get_settings()
        exec_cfg: Any = settings.execution if hasattr(settings, "execution") else {}
        max_cap = exec_cfg.get("max_capital", 300.0) if isinstance(exec_cfg, dict) else 300.0
        guard_config = CapitalGuardConfig(max_capital=max_cap)
        monitor = PositionMonitor(broker=broker, guard_config=guard_config, realtime_feed=rt_feed)
        monitor.initialize()

        mode_label = "PAPER" if paper else "LIVE"
        console.print(
            f"[bold blue]Position monitor started ({mode_label}, "
            f"interval={poll_seconds}s, realtime={'ON' if rt_feed else 'OFF'})[/bold blue]"
        )
        console.print("  Press Ctrl-C to stop\n")

        start_time = _time.time()

        try:
            while True:
                result = monitor.check_and_exit()
                console.print(f"  {result.summary()}")

                for action in result.actions:
                    style = "green" if action.success else "red"
                    console.print(
                        f"  [{style}]EXIT {action.symbol}: {action.reason.value} "
                        f"→ P&L=${action.pnl_estimate:.2f} "
                        f"({'OK' if action.success else action.error})[/{style}]"
                    )

                if duration > 0 and (_time.time() - start_time) >= duration:
                    break

                _time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")

        if rt_feed:
            rt_feed.stop()
        console.print("[green]Position monitor stopped.[/green]")

    except Exception as e:
        console.print(f"[red]Position monitor failed: {e}[/red]")
        logger.exception("Position monitor failed")
        raise typer.Exit(1) from None


@app.command()
def model_search(
    problem_id: str = typer.Option("equity_direction", help="Problem identifier"),
    target_col: str = typer.Option("fwd_return_1d", help="Target column name"),
    data_path: str = typer.Option("data/features.parquet", help="Path to feature data (Parquet)"),
    output_dir: str = typer.Option("data/model_search", help="Output directory for results"),
    primary_metric: str = typer.Option("sharpe", help="Primary metric for model selection"),
    task_type: str = typer.Option("regression", help="Task type: regression or classification"),
    max_per_family: int = typer.Option(20, help="Max candidates per model family"),
    train_size: int = typer.Option(252, help="Walk-forward training window size"),
    test_size: int = typer.Option(63, help="Walk-forward test window size"),
):
    """Run model search across diverse model families (Section 7)."""
    configure_logging()

    from pipeline.experiment_registry import ExperimentRegistry, KnowledgeStore
    from pipeline.model_search import ModelSearcher, default_equity_search_spaces

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(storage_path=out / "experiment_registry.json")
    knowledge_store = KnowledgeStore(storage_path=out / "knowledge_store.json")

    searcher = ModelSearcher(
        registry=registry,
        knowledge_store=knowledge_store,
        problem_id=problem_id,
        primary_metric=primary_metric,
    )

    data_file = Path(data_path)
    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_path}[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(data_file)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)

    spaces = default_equity_search_spaces(task_type=task_type)

    def eval_fn(y_true, y_pred):
        from pipeline.eval.metrics import hit_rate, sharpe_sortino

        aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
        rmse = float(((aligned_true - aligned_pred) ** 2).mean() ** 0.5)
        hr = hit_rate(aligned_true, aligned_pred)
        sharpe, sortino = sharpe_sortino(aligned_true - aligned_pred)
        return {"rmse": rmse, "hit_rate": hr, "sharpe": sharpe, "sortino": sortino}

    results = searcher.run_search(
        df=df,
        target_col=target_col,
        spaces=spaces,
        eval_fn=eval_fn,
        train_size=train_size,
        test_size=test_size,
        max_per_family=max_per_family,
    )

    searcher.update_meta_knowledge(results)

    from pipeline.report_generators import generate_model_search_report

    report = generate_model_search_report(
        results=[
            {
                "model_family": r.model_spec.model_family,
                "hyperparameters": r.model_spec.hyperparameters,
                "primary_metric_value": r.primary_metric_value,
                "secondary_metrics": r.secondary_metrics,
                "compute_seconds": r.compute_seconds,
            }
            for r in results
        ],
        primary_metric=primary_metric,
        meta_learning_insights=knowledge_store.generate_meta_learning_insights(registry),
    )

    report_path = out / "model_search_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    table = Table(title="Model Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Family", style="green")
    table.add_column(primary_metric.upper(), style="yellow")
    table.add_column("Time (s)", style="blue")

    for i, r in enumerate(results[:10]):
        table.add_row(
            str(i + 1),
            r.model_spec.model_family,
            f"{r.primary_metric_value:.4f}",
            f"{r.compute_seconds:.1f}",
        )

    console.print(table)
    console.print(f"[green]Report saved to {report_path}[/green]")


@app.command()
def ensemble_build(
    problem_id: str = typer.Option("ensemble_search", help="Problem identifier"),
    target_col: str = typer.Option("fwd_return_1d", help="Target column name"),
    data_path: str = typer.Option("data/features.parquet", help="Path to feature data"),
    search_dir: str = typer.Option("data/model_search", help="Model search output directory"),
    output_dir: str = typer.Option("data/ensemble", help="Output directory"),
    primary_metric: str = typer.Option("sharpe", help="Primary metric"),
    train_size: int = typer.Option(252, help="Walk-forward training window"),
    test_size: int = typer.Option(63, help="Walk-forward test window"),
):
    """Build ensemble from model search results (Section 8)."""
    configure_logging()

    from pipeline.ensemble import EnsembleBuilder, EnsembleComponent
    from pipeline.experiment_registry import ExperimentRegistry
    from pipeline.model_search import ModelSearcher, ModelSpec

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(storage_path=out / "experiment_registry.json")

    data_file = Path(data_path)
    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_path}[/red]")
        raise typer.Exit(1)

    df = pd.read_parquet(data_file)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)

    search_registry_path = Path(search_dir) / "experiment_registry.json"
    if not search_registry_path.exists():
        console.print(f"[red]Search registry not found: {search_registry_path}[/red]")
        console.print("Run 'mdw model-search' first.")
        raise typer.Exit(1)

    search_registry = ExperimentRegistry(storage_path=search_registry_path)
    from pipeline.experiment_registry import ExperimentStatus

    completed = search_registry.list_experiments(status=ExperimentStatus.COMPLETED)

    if not completed:
        console.print("[red]No completed experiments found in search registry.[/red]")
        raise typer.Exit(1)

    sorted_exps = sorted(completed, key=lambda e: e.primary_metric_value or 0.0, reverse=True)[:5]

    searcher = ModelSearcher(registry=search_registry, problem_id=problem_id)
    components = []
    for exp in sorted_exps:
        spec = ModelSpec(
            model_family=exp.model_family,
            hyperparameters=exp.hyperparameters,
        )
        train_fn, predict_fn = searcher.build_model(spec, target_col=target_col)
        components.append(
            EnsembleComponent(
                component_id=exp.experiment_id[:8],
                model_spec=spec,
                train_fn=train_fn,
                predict_fn=predict_fn,
            )
        )

    builder = EnsembleBuilder(
        registry=registry, primary_metric=primary_metric, problem_id=problem_id
    )

    def eval_fn(y_true, y_pred):
        from pipeline.eval.metrics import hit_rate, sharpe_sortino

        aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
        rmse = float(((aligned_true - aligned_pred) ** 2).mean() ** 0.5)
        hr = hit_rate(aligned_true, aligned_pred)
        sharpe, sortino = sharpe_sortino(aligned_true - aligned_pred)
        return {"rmse": rmse, "hit_rate": hr, "sharpe": sharpe, "sortino": sortino}

    result = builder.run_ensemble_search(
        df,
        target_col,
        components,
        eval_fn,
        train_size=train_size,
        test_size=test_size,
    )

    from pipeline.report_generators import generate_ensemble_report

    report = generate_ensemble_report(
        ensemble_method=result.method,
        component_weights=[
            {
                "component_id": c.component_id,
                "model_family": c.model_spec.model_family if c.model_spec else "unknown",
                "weight": c.weight,
            }
            for c in result.components
        ],
        primary_metric=primary_metric,
        primary_metric_value=result.primary_metric_value,
    )

    report_path = out / "ensemble_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"[green]Best ensemble method: {result.method}[/green]")
    console.print(f"[green]{primary_metric}: {result.primary_metric_value:.4f}[/green]")
    console.print(f"[green]Report saved to {report_path}[/green]")


@app.command()
def daily_predictions(
    date: str | None = typer.Option(
        None, "--date", "-d", help="Signal date (YYYY-MM-DD). Default: today."
    ),  # noqa: B008
    threshold: int = typer.Option(
        60,
        "--threshold",
        "-t",
        help="Minimum signal score",
        callback=_validate_range(min_val=0, max_val=100),
    ),  # noqa: B008
):
    """Run the full daily predictions pipeline: generate signals, track outcomes, build static site.

    This is the main command used by the daily GitHub Actions workflow. It:
    1. Loads price data from the database for the ETF universe
    2. Generates trading signals via the strategy engine
    3. Updates the prediction history with new signals and resolves past outcomes
    4. Builds a static HTML site for GitHub Pages deployment
    """
    import yaml as _yaml

    from pipeline.strategy.signals import SignalEngine, compute_indicators
    from pipeline.web.performance_tracker import PerformanceTracker
    from pipeline.web.static_builder import build_static_site

    console.print("[bold blue]Running daily predictions pipeline...[/bold blue]")

    # Load config
    config_path = Path("config.yaml")
    config = (_yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
    dp_config = config.get("daily_predictions", {})
    universe = dp_config.get("universe", ["SPY", "QQQ", "IWM"])
    signals_dir = Path(dp_config.get("signals_dir", "data/signals"))
    history_path = Path(dp_config.get("history_file", "data/prediction_history.json"))
    output_dir = Path(dp_config.get("output_dir", "site"))
    lookback = dp_config.get("lookback_days", 252)

    signals_dir.mkdir(parents=True, exist_ok=True)

    # Signal date is resolved after loading price data so we can fall back
    # to the latest available trading day when no explicit date is given.
    explicit_date = pd.Timestamp(date) if date else None

    console.print(f"  Universe: {len(universe)} ETFs")

    # Load price data from database
    db = get_db_manager()
    price_data: dict[str, pd.DataFrame] = {}

    for ticker in universe:
        try:
            rows = db.run_query(
                "SELECT p.date, p.open, p.high, p.low, p.close, p.volume "
                "FROM cur_prices_ohlcv_daily p "
                "JOIN dim_symbol s ON p.symbol_id = s.symbol_id "
                "WHERE s.ticker = :ticker "
                "ORDER BY p.date DESC LIMIT :lookback",
                {"ticker": ticker, "lookback": lookback},
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for col in ("open", "high", "low", "close", "volume"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if len(df) >= 50:  # Need enough data for indicators
                    price_data[ticker] = df
                else:
                    logger.warning("Skipping %s: only %d bars (need 50+)", ticker, len(df))
            else:
                logger.warning("No price data for %s in database", ticker)
        except Exception as exc:
            logger.warning("Could not load %s: %s", ticker, exc)

    if not price_data:
        # Fallback: try loading from raw files in data/raw/prices/
        raw_prices_dir = Path("data/raw/prices")
        if raw_prices_dir.exists():
            for f in raw_prices_dir.iterdir():
                if f.suffix.lower() in {".csv", ".parquet", ".pq"}:
                    ticker = f.stem.upper()
                    if ticker in universe:
                        df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_parquet(f)
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date").sort_index()
                        if len(df) >= 50:
                            price_data[ticker] = df

    if not price_data:
        console.print("[red]No price data available. Run extract prices first.[/red]")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(price_data)} tickers with price data")

    # Resolve signal date: use explicit date if given, otherwise the latest
    # trading day present in the price data (today's bar may not exist yet).
    if explicit_date is not None:
        signal_date = explicit_date
    else:
        latest_dates = [df.index.max() for df in price_data.values() if not df.empty]
        signal_date = max(latest_dates) if latest_dates else pd.Timestamp.now().normalize()

    console.print(f"  Signal date: {signal_date.date()}")

    # Compute indicators and generate signals
    indicator_data: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        required = {"open", "high", "low", "close", "volume"}
        if required.issubset(set(df.columns)):
            indicator_data[ticker] = compute_indicators(df)

    # Get SPY for regime classification
    spy_prices = None
    if "SPY" in indicator_data:
        spy_prices = indicator_data["SPY"]["close"]

    engine = SignalEngine(entry_threshold=threshold)
    scores = engine.score_universe(indicator_data, spy_prices=spy_prices, date=signal_date)
    eligible = [s for s in scores if s.entry_eligible]

    console.print(
        f"  Scored {len(scores)} symbols, {len(eligible)} eligible (score >= {threshold})"
    )

    # Format signals
    signals_df = pd.DataFrame()
    if eligible:
        from pipeline.strategy.signal_output import format_signals, write_signal_csv

        signals_df = format_signals(scores=eligible, price_data=indicator_data, date=signal_date)
        if not signals_df.empty:
            filepath = write_signal_csv(signals_df, signals_dir, signal_date)
            console.print(f"  Wrote {len(signals_df)} signals to {filepath}")

    # Update prediction history
    tracker = PerformanceTracker(history_path)

    if not signals_df.empty:
        added = tracker.add_signals(signals_df, str(signal_date.date()))
        console.print(f"  Added {added} new predictions to history")

    # Resolve past outcomes
    summary = tracker.resolve_outcomes(price_data, str(signal_date.date()))
    console.print(
        f"  Outcomes: {summary.get('hit_target', 0)} wins, "
        f"{summary.get('stopped_out', 0)} stopped, "
        f"{summary.get('expired', 0)} expired, "
        f"{summary.get('still_active', 0)} active"
    )
    tracker.save()

    # Build static site
    site_path = build_static_site(
        output_dir=output_dir,
        signals_dir=signals_dir,
        history_path=history_path,
    )
    console.print(f"[green]  Static site built at {site_path}/[/green]")

    # Print summary
    stats = tracker.get_stats()
    console.print("\n[bold green]Daily predictions complete![/bold green]")
    console.print(f"  Win rate: {stats['win_rate']}% ({stats['resolved']} resolved)")
    console.print(f"  View at: {output_dir}/index.html")


@app.command()
def backfill_predictions(
    start: str = typer.Option(
        ..., "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end: str | None = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD). Default: today."
    ),
    threshold: int = typer.Option(60, "--threshold", "-t", help="Minimum signal score"),
):
    """Backfill prediction history by running signals over a date range.

    Extracts prices once, then iterates over each trading day in the range,
    generating signals and resolving outcomes against future price data.
    This seeds the prediction history and site for initial deployment.
    """
    import yaml as _yaml

    from pipeline.strategy.signals import SignalEngine, compute_indicators
    from pipeline.web.performance_tracker import PerformanceTracker
    from pipeline.web.static_builder import build_static_site

    console.print("[bold blue]Backfilling predictions...[/bold blue]")

    config_path = Path("config.yaml")
    config = (_yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
    dp_config = config.get("daily_predictions", {})
    universe = dp_config.get("universe", ["SPY", "QQQ", "IWM"])
    signals_dir = Path(dp_config.get("signals_dir", "data/signals"))
    history_path = Path(dp_config.get("history_file", "data/prediction_history.json"))
    output_dir = Path(dp_config.get("output_dir", "site"))
    lookback = dp_config.get("lookback_days", 252)

    signals_dir.mkdir(parents=True, exist_ok=True)

    end_date = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    start_date = pd.Timestamp(start)

    # Load price data from database (full range needed for backfill)
    db = get_db_manager()
    price_data: dict[str, pd.DataFrame] = {}

    for ticker in universe:
        try:
            rows = db.run_query(
                "SELECT p.date, p.open, p.high, p.low, p.close, p.volume "
                "FROM cur_prices_ohlcv_daily p "
                "JOIN dim_symbol s ON p.symbol_id = s.symbol_id "
                "WHERE s.ticker = :ticker "
                "ORDER BY p.date",
                {"ticker": ticker},
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for col in ("open", "high", "low", "close", "volume"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if len(df) >= 50:
                    price_data[ticker] = df
        except Exception as exc:
            logger.warning("Could not load %s: %s", ticker, exc)

    if not price_data:
        console.print("[red]No price data available. Run extract + load first.[/red]")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(price_data)} tickers")

    # Build trading day schedule within the date range
    trading_days = pd.bdate_range(start_date, end_date)
    # Filter to dates where we have enough price history for indicators
    earliest_data = min(df.index.min() for df in price_data.values())
    min_signal_date = earliest_data + pd.Timedelta(days=70)  # need ~50 bars
    trading_days = [d for d in trading_days if d >= min_signal_date and d <= end_date]

    console.print(f"  Backfilling {len(trading_days)} trading days: {trading_days[0].date()} → {trading_days[-1].date()}")

    tracker = PerformanceTracker(history_path)
    engine = SignalEngine(entry_threshold=threshold)
    total_signals = 0

    for i, signal_date in enumerate(trading_days):
        # Build indicator data using only data up to signal_date (no look-ahead)
        indicator_data: dict[str, pd.DataFrame] = {}
        for ticker, df in price_data.items():
            hist = df[df.index <= signal_date].tail(lookback)
            required = {"open", "high", "low", "close", "volume"}
            if len(hist) >= 50 and required.issubset(set(hist.columns)):
                indicator_data[ticker] = compute_indicators(hist)

        if not indicator_data:
            continue

        spy_prices = indicator_data["SPY"]["close"] if "SPY" in indicator_data else None
        scores = engine.score_universe(indicator_data, spy_prices=spy_prices, date=signal_date)
        eligible = [s for s in scores if s.entry_eligible]

        if eligible:
            from pipeline.strategy.signal_output import format_signals, write_signal_csv

            signals_df = format_signals(scores=eligible, price_data=indicator_data, date=signal_date)
            if not signals_df.empty:
                write_signal_csv(signals_df, signals_dir, signal_date)
                added = tracker.add_signals(signals_df, str(signal_date.date()))
                total_signals += added

        # Resolve past outcomes using full price data (we have future data in backfill)
        tracker.resolve_outcomes(price_data, str(signal_date.date()))

        if (i + 1) % 20 == 0:
            console.print(f"    ... {i + 1}/{len(trading_days)} days processed, {total_signals} signals so far")

    tracker.save()

    # Build site with backfilled data
    build_static_site(output_dir=output_dir, signals_dir=signals_dir, history_path=history_path)

    stats = tracker.get_stats()
    console.print(f"\n[bold green]Backfill complete![/bold green]")
    console.print(f"  Total predictions: {stats['total']}")
    console.print(f"  Resolved: {stats['resolved']} ({stats['hit_target']} wins, {stats['stopped_out']} stopped, {stats['expired']} expired)")
    console.print(f"  Active: {stats['active']}")
    console.print(f"  Win rate: {stats['win_rate']}%")
    console.print(f"  Avg P&L: {stats['avg_pnl_pct']}%")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
