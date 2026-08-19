"""Command-line interface for the data pipeline."""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import date, datetime
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
from pipeline.extract.cftc_cot import extract_cftc_cot
from pipeline.extract.checkpoint_helpers import get_checkpoint_manager, make_operation_id
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
from pipeline.infrastructure.checkpoint import CheckpointContext
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
        "reddit-sentiment, short-interest, etf-flows, cftc-cot)",
    ),
    start: str | None = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),  # noqa: B008
    end: str | None = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),  # noqa: B008
    output_dir: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output directory"
    ),
    force: bool = typer.Option(  # noqa: B008
        False, "--force", "-f", help="Re-fetch even if output already exists"
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
            files = extract_earnings(
                raw_path, start_date=start, end_date=end, run_id=run_id, force=force
            )
        elif source == "reddit-sentiment":
            files = extract_reddit_sentiment(raw_path, run_id=run_id)
        elif source == "short-interest":
            files = extract_short_interest(raw_path, run_id=run_id)
        elif source == "etf-flows":
            files = extract_etf_flows(raw_path, run_id=run_id)
        elif source == "cftc-cot":
            s = date.fromisoformat(start) if start else None
            e = date.fromisoformat(end) if end else None
            files = extract_cftc_cot(raw_path, start_date=s, end_date=e, run_id=run_id)
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
        "reddit_sentiment, short_interest, etf_flows, cftc_cot)",
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
def transform_curated(
    include_gdelt: bool = typer.Option(
        False, "--include-gdelt", help="Also run GDELT transform (slow: 198M+ rows)"
    ),
):
    """Transform raw data into curated tables."""
    run_id = record_pipeline_run("transform_curated", {"include_gdelt": include_gdelt})

    try:
        transformer = CuratedTransformer(run_id=run_id)
        results = transformer.transform_all(include_gdelt=include_gdelt)

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


# Sources ordered by tier (see HISTORICAL_BACKFILL.md):
#   Tier 1 — no API keys: prices, factors
#   Tier 2 — free API key: fred
#   Tier 3 — no keys, slow: sec-fundamentals, sec-insider, sec-13f, cftc-cot
#   Tier 4 — shallow history: gdelt, polymarket, options, earnings,
#            reddit-sentiment, short-interest, etf-flows
_BACKFILL_SOURCES: list[tuple[str, str]] = [
    ("prices", "prices"),
    ("factors", "factors"),
    ("fred", "fred"),
    ("sec-fundamentals", "sec_fundamentals"),
    ("sec-insider", "sec_insider"),
    ("sec-13f", "sec_13f"),
    ("cftc-cot", "cftc_cot"),
    ("gdelt", "gdelt"),
    ("polymarket", "polymarket"),
    ("options", "options"),
    ("earnings", "earnings"),
    ("reddit-sentiment", "reddit_sentiment"),
    ("short-interest", "short_interest"),
    ("etf-flows", "etf_flows"),
]


@app.command()
def historical_backfill(
    start: str = typer.Option("2010-01-01", "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD, default: today)"
    ),
    sources: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--source",
        help="Restrict to specific sources (repeatable). Default: all.",
    ),
    skip_transform: bool = typer.Option(
        False, "--skip-transform", help="Skip transform / snapshot / dq steps"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-fetch even if output already exists"
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from last checkpoint (default). --no-resume forces fresh start.",
    ),
):
    """Run a full historical backfill: extract, load, transform, snapshot, dq.

    Orchestrates extraction for every source from --start to --end, loads into
    the raw warehouse, then runs curated transforms, latency stats, snapshots,
    data-quality checks, and inventory.

    Supports checkpointed resume: if a backfill is interrupted, re-running the
    same command will skip already-completed sources and items within each
    source.  Use --no-resume to force a completely fresh backfill.

    Examples:
        # Full 2010-present backfill (all sources)
        mdw historical-backfill --start 2010-01-01

        # Only prices and factors
        mdw historical-backfill --source prices --source factors

        # Extract only, skip downstream transforms
        mdw historical-backfill --skip-transform

        # Force a fresh start, ignoring any saved checkpoints
        mdw historical-backfill --no-resume
    """
    end_date = end or date.today().isoformat()
    settings = get_settings()
    raw_path = settings.raw_lake_path

    run_id = record_pipeline_run(
        "historical_backfill",
        {"start": start, "end": end_date, "sources": sources or "all"},
    )

    # Determine which sources to run
    if sources:
        source_list = [
            (extract_name, load_name)
            for extract_name, load_name in _BACKFILL_SOURCES
            if extract_name in sources
        ]
        unknown = set(sources) - {name for name, _ in _BACKFILL_SOURCES}
        if unknown:
            console.print(f"[red]Unknown sources: {unknown}[/red]")
            raise typer.Exit(1)
    else:
        source_list = list(_BACKFILL_SOURCES)

    # Clear all checkpoints when --no-resume is set
    ckpt_mgr = get_checkpoint_manager()
    source_op_id = "backfill_orchestrator"
    if not resume:
        ckpt_mgr.delete_checkpoint(source_op_id)
        for extract_name, load_name in source_list:
            ckpt_mgr.delete_checkpoint(make_operation_id(extract_name))
            ckpt_mgr.delete_checkpoint(f"load_{load_name}")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    # --- Phase 1: Extract + Load each source sequentially ---
    console.print(
        f"[bold]Historical backfill: {start} to {end_date} " f"({len(source_list)} sources)[/bold]"
    )

    with CheckpointContext(ckpt_mgr, source_op_id, resume=resume) as src_ctx:
        completed_sources: set[str] = set(src_ctx.state.get("completed_sources", []))

        for extract_name, load_name in source_list:
            if extract_name in completed_sources:
                console.print(f"\n[dim]--- {extract_name} (already done, skipping) ---[/dim]")
                succeeded.append(extract_name)
                continue

            console.print(f"\n[cyan]--- {extract_name} ---[/cyan]")

            # Extract
            try:
                files: list[Path] | dict[str, list[Path]]
                if extract_name == "factors":
                    files = extract_factors_ff(raw_path, run_id=run_id)
                elif extract_name == "options":
                    files = extract_options(raw_path, run_id=run_id, resume=resume)
                elif extract_name == "reddit-sentiment":
                    files = extract_reddit_sentiment(raw_path, run_id=run_id)
                elif extract_name == "short-interest":
                    files = extract_short_interest(raw_path, run_id=run_id)
                elif extract_name == "etf-flows":
                    files = extract_etf_flows(raw_path, run_id=run_id)
                elif extract_name == "cftc-cot":
                    s = date.fromisoformat(start)
                    e = date.fromisoformat(end_date)
                    files = extract_cftc_cot(
                        raw_path,
                        start_date=s,
                        end_date=e,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "fred":
                    files = extract_fred(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "gdelt":
                    files = extract_gdelt(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "polymarket":
                    files = extract_polymarket(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                    )
                elif extract_name == "prices":
                    files = extract_prices(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "sec-fundamentals":
                    files = extract_sec_fundamentals(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "sec-insider":
                    files = extract_sec_insider(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "sec-13f":
                    files = extract_sec_13f(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        resume=resume,
                    )
                elif extract_name == "earnings":
                    files = extract_earnings(
                        raw_path,
                        start_date=start,
                        end_date=end_date,
                        run_id=run_id,
                        force=force,
                        resume=resume,
                    )
                else:
                    console.print(f"  [yellow]⚠ No extract handler for {extract_name}[/yellow]")
                    continue

                file_count = (
                    len(files) if isinstance(files, list) else sum(len(v) for v in files.values())
                )
                console.print(f"  [green]✓ Extracted {file_count} files[/green]")
            except Exception as exc:
                console.print(f"  [red]✗ Extract failed: {exc}[/red]")
                failed.append((extract_name, str(exc)))
                continue

            # Load
            try:
                loader = RawLoader()
                rows = loader.load_all_raw_files(
                    raw_path, load_name, run_id=UUID(run_id), resume=resume
                )
                console.print(f"  [green]✓ Loaded {rows} rows[/green]")
                succeeded.append(extract_name)
            except Exception as exc:
                console.print(f"  [red]✗ Load failed: {exc}[/red]")
                failed.append((extract_name, str(exc)))
                continue

            # Record source completion in checkpoint
            done_list = src_ctx.state.get("completed_sources", [])
            done_list.append(extract_name)
            src_ctx.update(
                completed_sources=done_list,
                completed_items=len(done_list),
                total_items=len(source_list),
                last_processed=extract_name,
            )
            src_ctx.save()

    # --- Phase 2: Downstream pipeline ---
    if not skip_transform and succeeded:
        console.print("\n[bold]Running downstream pipeline...[/bold]")

        # Transform
        try:
            console.print("[cyan]--- transform-curated ---[/cyan]")
            transformer = CuratedTransformer(run_id=run_id)
            results = transformer.transform_all()
            for tbl, cnt in results.items():
                console.print(f"  {tbl}: {cnt} rows")
            console.print("[green]✓ Transform complete[/green]")
        except Exception as exc:
            console.print(f"[red]✗ Transform failed: {exc}[/red]")

        # Latency stats
        try:
            console.print("[cyan]--- latency-stats ---[/cyan]")
            lat_results = refresh_latency_stats()
            for src, metrics in lat_results.items():
                n = metrics.get("sample_size", 0)
                console.print(f"  {src}: {n} samples")
            console.print("[green]✓ Latency stats complete[/green]")
        except Exception as exc:
            console.print(f"[red]✗ Latency stats failed: {exc}[/red]")

        # Snapshots
        try:
            console.print("[cyan]--- build-snapshots ---[/cyan]")
            builder = SymbolSnapshotBuilder()
            snap_results = builder.build_snapshots_for_range(
                start_ts=datetime.fromisoformat(f"{start}T00:00:00"),
                end_ts=datetime.fromisoformat(f"{end_date}T00:00:00"),
                frequency="1d",
            )
            console.print(f"[green]✓ Snapshots built: {snap_results}[/green]")
        except Exception as exc:
            console.print(f"[red]✗ Snapshots failed: {exc}[/red]")

        # DQ
        try:
            console.print("[cyan]--- dq ---[/cyan]")
            passed = run_dq_tests()
            if passed:
                console.print("[green]✓ DQ tests passed[/green]")
            else:
                console.print("[yellow]⚠ Some DQ tests failed[/yellow]")
        except Exception as exc:
            console.print(f"[red]✗ DQ failed: {exc}[/red]")

    # --- Summary ---
    console.print("\n[bold]Backfill Summary[/bold]")
    console.print(f"  Succeeded: {len(succeeded)} source(s)")
    if failed:
        console.print(f"  Failed: {len(failed)} source(s)")
        for name, err in failed:
            console.print(f"    [red]✗ {name}: {err}[/red]")

    update_pipeline_run(
        run_id,
        "success" if not failed else "partial",
        {"succeeded": succeeded, "failed": [f[0] for f in failed]},
    )

    if failed:
        raise typer.Exit(1)


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
def test_signal_alpha(
    signals_path: Path = typer.Option(  # noqa: B008
        ..., "--signals", help="Signals file (wide format: dates x symbols, CSV/parquet)"
    ),
    prices_path: Path = typer.Option(  # noqa: B008
        ..., "--prices", help="Prices file (wide format: dates x symbols, CSV/parquet)"
    ),
    signal_name: str = typer.Option(
        "signal", "--signal-name", help="Signal identifier"
    ),  # noqa: B008
    horizons: str = typer.Option(  # noqa: B008
        "1,5,10,21,63", "--horizons", help="Comma-separated forward-return horizons (days)"
    ),
    train_size: int = typer.Option(
        252, "--train-size", help="Walk-forward training window"
    ),  # noqa: B008
    test_size: int = typer.Option(63, "--test-size", help="Walk-forward test window"),  # noqa: B008
    embargo: int = typer.Option(
        5, "--embargo", help="Embargo days between train/test"
    ),  # noqa: B008
):
    """Test whether a signal has statistically significant predictive power (IC analysis)."""
    from pipeline.eval.signal_alpha import (
        compute_forward_returns,
        ic_decay_analysis,
        walk_forward_ic,
    )

    signals = _read_data(signals_path)
    prices = _read_data(prices_path)

    # Ensure DatetimeIndex
    if not isinstance(signals.index, pd.DatetimeIndex):
        if "date" in signals.columns:
            signals = signals.set_index("date")
        signals.index = pd.to_datetime(signals.index)
    if not isinstance(prices.index, pd.DatetimeIndex):
        if "date" in prices.columns:
            prices = prices.set_index("date")
        prices.index = pd.to_datetime(prices.index)

    horizon_list = [int(h.strip()) for h in horizons.split(",")]

    # --- IC Decay Analysis ---
    console.print(f"\n[bold]IC Decay Analysis for '{signal_name}'[/bold]\n")
    decay = ic_decay_analysis(signals, prices, signal_name=signal_name, horizons=horizon_list)

    decay_table = Table(title="IC by Forward-Return Horizon")
    decay_table.add_column("Horizon (days)", justify="right")
    decay_table.add_column("Mean IC", justify="right")
    decay_table.add_column("IC Std", justify="right")
    decay_table.add_column("IC IR", justify="right")
    for h in decay.horizons:
        ic = decay.ic_by_horizon.get(h, float("nan"))
        std = decay.ic_std_by_horizon.get(h, float("nan"))
        ir = decay.ic_ir_by_horizon.get(h, float("nan"))
        marker = " *" if h == decay.best_horizon else ""
        decay_table.add_row(
            f"{h}{marker}",
            f"{ic:.4f}" if pd.notna(ic) else "N/A",
            f"{std:.4f}" if pd.notna(std) else "N/A",
            f"{ir:.4f}" if pd.notna(ir) else "N/A",
        )
    console.print(decay_table)
    console.print(f"  Best horizon: [bold]{decay.best_horizon}d[/bold] (highest IC IR)\n")

    # --- Walk-Forward IC Test at best horizon ---
    console.print("[bold]Walk-Forward IC Significance Test[/bold]\n")
    fwd = compute_forward_returns(prices, horizon=decay.best_horizon)
    common_syms = signals.columns.intersection(fwd.columns)
    result = walk_forward_ic(
        signals[common_syms],
        fwd[common_syms],
        signal_name=signal_name,
        train_size=train_size,
        test_size=test_size,
        embargo_size=embargo,
    )

    result_table = Table(title="Walk-Forward IC Results")
    result_table.add_column("Metric", justify="left")
    result_table.add_column("Value", justify="right")
    result_table.add_row("Folds", str(result.n_folds))
    result_table.add_row("IC Mean", f"{result.ic_mean:.4f}")
    result_table.add_row("IC Std", f"{result.ic_std:.4f}")
    result_table.add_row("IC t-stat", f"{result.ic_t_stat:.2f}")
    result_table.add_row("IC p-value", f"{result.ic_p_value:.4f}")
    result_table.add_row("Deflated Sharpe Prob", f"{result.deflated_sharpe_prob:.4f}")
    console.print(result_table)

    if result.passed:
        console.print(
            f"\n[green bold]PASS[/green bold] — Signal '{signal_name}' has statistically "
            f"significant predictive power (DSR prob={result.deflated_sharpe_prob:.3f} > 0.95)"
        )
    else:
        console.print(
            f"\n[red bold]FAIL[/red bold] — Signal '{signal_name}' does NOT pass the deflated "
            f"Sharpe gate (DSR prob={result.deflated_sharpe_prob:.3f} <= 0.95). "
            "Cannot distinguish from noise."
        )


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
        ("cur_cftc_cot", "date", "report_date"),
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
def backtest_earnings_drift(
    prices_dir: Path = typer.Option(  # noqa: B008
        Path("data/prices"),
        "--prices-dir",
        "-p",
        help="Directory of per-ticker OHLCV CSV/Parquet files (ignored when --from-db).",
    ),
    earnings_dir: Path = typer.Option(  # noqa: B008
        Path("data/raw/earnings"),
        "--earnings-dir",
        "-e",
        help="Directory of per-ticker earnings parquets (from EarningsExtractor).",
    ),
    start_date: str = typer.Option("2020-01-01", "--start", help="Backtest start (YYYY-MM-DD)"),
    end_date: str = typer.Option("2025-12-31", "--end", help="Backtest end (YYYY-MM-DD)"),
    lookback_days: int = typer.Option(
        30, "--lookback", help="Days to carry PEAD signal after earnings report"
    ),
    min_surprise_pct: float = typer.Option(
        5.0, "--min-surprise", help="Minimum |EPS surprise %| to generate a signal"
    ),
    capital: float = typer.Option(1e7, "--capital", help="Initial backtest capital ($)"),
    from_db: bool = typer.Option(
        False, "--from-db", help="Load price data from the warehouse DB instead of files."
    ),
):
    """Backtest the QSG-EARNINGS-DRIFT-001 (PEAD) strategy.

    Loads OHLCV price data from --prices-dir (or the warehouse DB with --from-db)
    and earnings surprise data from --earnings-dir (written by 'make extract-all'),
    then runs the post-earnings announcement drift backtest.

    Examples:
        mdw backtest-earnings-drift --from-db
        mdw backtest-earnings-drift --from-db --min-surprise 3.0 --lookback 20
        mdw backtest-earnings-drift --prices-dir data/prices --start 2022-01-01
    """
    configure_logging()

    from pipeline.strategy.earnings_drift import run_earnings_drift_backtest

    # ------------------------------------------------------------------
    # Load price data
    # ------------------------------------------------------------------
    price_data: dict[str, pd.DataFrame] = {}

    if from_db:
        console.print("  Loading price data from warehouse DB...")
        db = get_db_manager()
        rows = db.run_query(f"""
            SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
            FROM cur_prices_ohlcv_daily p
            JOIN dim_symbol s ON s.symbol_id = p.symbol_id
            WHERE p.date >= '{start_date}' AND p.date <= '{end_date}'
            ORDER BY s.ticker, p.date
        """)
        if not rows:
            console.print("[red]No price data in DB for the requested date range.[/red]")
            raise typer.Exit(1)
        import collections

        by_ticker: dict[str, list] = collections.defaultdict(list)
        for r in rows:
            by_ticker[r["ticker"]].append(r)
        for ticker, ticker_rows in by_ticker.items():
            df = pd.DataFrame(ticker_rows).drop(columns=["ticker"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            price_data[ticker] = df
        console.print(f"  Loaded {len(price_data)} tickers from DB")
    else:
        if not prices_dir.exists():
            console.print(
                f"[red]Prices directory not found: {prices_dir}\n"
                "  Use --from-db to load from the warehouse instead.[/red]"
            )
            raise typer.Exit(1)

        for f in sorted(prices_dir.iterdir()):
            if f.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                continue
            ticker = f.stem.upper()
            df = _read_data(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                logger.warning("Skipping %s: missing OHLCV columns", ticker)
                continue
            price_data[ticker] = df

        if not price_data:
            console.print(f"[red]No OHLCV files found in {prices_dir}[/red]")
            raise typer.Exit(1)
        console.print(f"  Loaded {len(price_data)} tickers from {prices_dir}")

    # ------------------------------------------------------------------
    # Earnings data check
    # ------------------------------------------------------------------
    if not earnings_dir.exists():
        console.print(
            f"[yellow]Earnings directory not found: {earnings_dir}[/yellow]\n"
            "  Run [bold]make extract-all[/bold] or [bold]mdw extract-earnings[/bold] first.\n"
            "  Continuing — tickers without earnings data will score 0."
        )
    else:
        n_files = len(list(earnings_dir.glob("*.parquet")))
        console.print(f"  Found {n_files} earnings parquet(s) in {earnings_dir}")

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold blue]Running QSG-EARNINGS-DRIFT-001 backtest[/bold blue]\n"
        f"  Period: {start_date} → {end_date}\n"
        f"  Min surprise: {min_surprise_pct:.1f}%  |  Lookback: {lookback_days}d\n"
        f"  Capital: ${capital:,.0f}"
    )

    result = run_earnings_drift_backtest(
        price_data=price_data,
        earnings_dir=earnings_dir,
        lookback_days=lookback_days,
        min_surprise_pct=min_surprise_pct,
        initial_capital=capital,
        start_date=start_date,
        end_date=end_date,
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    console.print("\n" + result.summary_table())

    if not result.trade_log.empty:
        console.print(f"\n[dim]Total trades: {len(result.trade_log)}[/dim]")
        by_exit = result.trade_log.groupby("exit_reason").size()
        for reason, count in by_exit.items():
            console.print(f"  {reason}: {count}")

    if result.metrics.total_trades == 0:
        console.print(
            "\n[yellow]No trades generated. Check that earnings data exists in "
            f"{earnings_dir} and that the surprise threshold ({min_surprise_pct}%) "
            "is not too high.[/yellow]"
        )


@app.command()
def validate_earnings_drift(
    earnings_dir: Path = typer.Option(  # noqa: B008
        Path("data/raw/earnings"),
        "--earnings-dir",
        "-e",
        help="Directory of per-ticker earnings parquets.",
    ),
    start_date: str = typer.Option("2020-01-01", "--start", help="Validation start (YYYY-MM-DD)"),
    end_date: str = typer.Option("2025-12-31", "--end", help="Validation end (YYYY-MM-DD)"),
    window_months: int = typer.Option(
        6, "--window", "-w", help="Length of each OOS window in months"
    ),
    lookback_days: int = typer.Option(30, "--lookback", help="PEAD signal lookback days"),
    min_surprise_pct: float = typer.Option(5.0, "--min-surprise", help="Min |EPS surprise %|"),
    capital: float = typer.Option(1e6, "--capital", help="Capital per window ($)"),
    rate_hike_threshold: float = typer.Option(
        75.0,
        "--rate-hike-threshold",
        help="Block entries when FEDFUNDS 3mo change exceeds this (bps). 0 = disabled.",
    ),
):
    """Walk-forward validation for QSG-EARNINGS-DRIFT-001 (PEAD).

    Splits the date range into independent out-of-sample windows and runs
    the backtest on each. Reports per-window Sharpe, return, hit rate, and
    drawdown to show whether the edge is consistent across time.

    Loads price data from the warehouse DB. Uses a FEDFUNDS rate-hike regime
    filter to sit flat during aggressive Fed tightening cycles.

    Examples:
        mdw validate-earnings-drift
        mdw validate-earnings-drift --window 3 --min-surprise 3.0
        mdw validate-earnings-drift --rate-hike-threshold 0   # disable filter
    """
    configure_logging()

    from pipeline.strategy.earnings_drift import walk_forward_earnings_drift

    # Load all price data from DB for the full period
    console.print("  Loading price data from warehouse DB...")
    db = get_db_manager()
    rows = db.run_query(f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM cur_prices_ohlcv_daily p
        JOIN dim_symbol s ON s.symbol_id = p.symbol_id
        WHERE p.date >= '{start_date}' AND p.date <= '{end_date}'
        ORDER BY s.ticker, p.date
    """)
    if not rows:
        console.print("[red]No price data in DB for the requested date range.[/red]")
        raise typer.Exit(1)

    import collections

    by_ticker: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)

    price_data: dict[str, pd.DataFrame] = {}
    for ticker, ticker_rows in by_ticker.items():
        df = pd.DataFrame(ticker_rows).drop(columns=["ticker"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        price_data[ticker] = df

    console.print(f"  Loaded {len(price_data)} tickers from DB")

    n_files = len(list(earnings_dir.glob("*.parquet"))) if earnings_dir.exists() else 0
    console.print(f"  Found {n_files} earnings parquet(s) in {earnings_dir}")

    console.print(
        f"\n[bold blue]Walk-forward validation — QSG-EARNINGS-DRIFT-001[/bold blue]\n"
        f"  Period: {start_date} → {end_date}  |  Window: {window_months}mo\n"
        f"  Min surprise: {min_surprise_pct:.1f}%  |  Lookback: {lookback_days}d\n"
        f"  Capital per window: ${capital:,.0f}"
    )

    wf_results = walk_forward_earnings_drift(
        price_data=price_data,
        earnings_dir=earnings_dir,
        lookback_days=lookback_days,
        min_surprise_pct=min_surprise_pct,
        initial_capital=capital,
        start_date=start_date,
        end_date=end_date,
        window_months=window_months,
        embargo_days=15,
        rate_hike_threshold_bps=rate_hike_threshold if rate_hike_threshold > 0 else None,
    )

    if not wf_results:
        console.print("[red]No windows produced results.[/red]")
        raise typer.Exit(1)

    # Per-window table
    table = Table(title="Walk-Forward Results — Per Window")
    table.add_column("Win", justify="right", style="dim")
    table.add_column("Period")
    table.add_column("Trades", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Hit %", justify="right")
    table.add_column("MaxDD %", justify="right")
    table.add_column("PF", justify="right")

    n_profitable = 0
    n_positive_sharpe = 0
    sharpes = []
    returns = []

    for r in wf_results:
        ret = r["total_return_pct"]
        sharpe = r["sharpe"]
        hit = r["hit_rate_pct"]
        dd = r["max_drawdown_pct"]
        pf = r["profit_factor"]

        if ret is not None and ret > 0:
            n_profitable += 1
        if sharpe is not None and sharpe > 0:
            n_positive_sharpe += 1
        if sharpe is not None:
            sharpes.append(sharpe)
        if ret is not None:
            returns.append(ret)

        ret_style = "green" if (ret or 0) > 0 else "red"
        sharpe_style = (
            "green" if (sharpe or 0) > 0.5 else ("yellow" if (sharpe or 0) > 0 else "red")
        )

        table.add_row(
            str(r["window"]),
            f"{r['start']} → {r['end']}",
            str(r["trades"]),
            f"[{ret_style}]{ret:.1f}%[/{ret_style}]" if ret is not None else "N/A",
            f"[{sharpe_style}]{sharpe:.3f}[/{sharpe_style}]" if sharpe is not None else "N/A",
            f"{hit:.1f}%" if hit is not None else "N/A",
            f"{dd:.1f}%" if dd is not None else "N/A",
            f"{pf:.2f}" if pf is not None else "N/A",
        )

    console.print(table)

    # Summary
    n = len(wf_results)
    import statistics

    mean_sharpe = statistics.mean(sharpes) if sharpes else float("nan")
    std_sharpe = statistics.stdev(sharpes) if len(sharpes) > 1 else float("nan")
    mean_return = statistics.mean(returns) if returns else float("nan")

    summary = Table(title="Walk-Forward Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Windows evaluated", str(n))
    summary.add_row("Profitable windows", f"{n_profitable}/{n} ({n_profitable/n:.0%})")
    pos_sharpe_pct = n_positive_sharpe / n
    summary.add_row("Positive Sharpe windows", f"{n_positive_sharpe}/{n} ({pos_sharpe_pct:.0%})")
    summary.add_row("Mean Sharpe (across windows)", f"{mean_sharpe:.3f}")
    summary.add_row("Std Sharpe (across windows)", f"{std_sharpe:.3f}")
    summary.add_row("Mean window return", f"{mean_return:.1f}%")
    console.print(summary)

    # Verdict
    if mean_sharpe > 0.5 and n_profitable / n >= 0.6:
        console.print("\n[bold green]VERDICT: Edge appears consistent.[/bold green]")
        console.print(
            "  Mean Sharpe > 0.5 and >60% of windows profitable. "
            "Proceed to paper trading validation."
        )
    elif mean_sharpe > 0 and n_profitable / n >= 0.5:
        console.print("\n[bold yellow]VERDICT: Edge is marginal — more data needed.[/bold yellow]")
        console.print("  Positive but weak. Watch for regime-dependence in the window breakdown.")
    else:
        console.print("\n[bold red]VERDICT: Edge not confirmed out-of-sample.[/bold red]")
        console.print("  Do not deploy. Review parameter choices and universe selection.")


@app.command()
def test_notifications(
    severity: str = typer.Option(
        "INFO",
        "--severity",
        help="Alert severity to send: INFO | WARNING | CRITICAL",
    ),
    title: str = typer.Option("Test alert", "--title", help="Alert title"),
    message: str = typer.Option(
        "This is a test notification from the pipeline CLI.",
        "--message",
        help="Alert message body",
    ),
) -> None:
    """Send a test notification to all configured channels.

    Useful for verifying Slack webhook and SMTP settings without
    triggering a real event.

    Examples::

        mdw test-notifications
        mdw test-notifications --severity WARNING --title "DQ check" --message "Prices stale"
        mdw test-notifications --severity CRITICAL
    """
    from pipeline.infrastructure.notifier import AlertSeverity, get_notifier

    severity_upper = severity.strip().upper()
    try:
        sev = AlertSeverity[severity_upper]
    except KeyError:
        console.print(
            f"[red]Unknown severity '{severity}'. Must be INFO, WARNING, or CRITICAL.[/red]"
        )
        raise typer.Exit(1)

    notifier = get_notifier()
    console.print(f"Sending [{severity_upper}] test notification …")
    console.print(f"  Slack configured: {'yes' if notifier.slack else 'no'}")
    console.print(f"  Email configured: {'yes' if notifier.email else 'no'}")
    console.print("  Console: always on")

    notifier.send(
        severity=sev,
        title=title,
        message=message,
        context={"source": "cli:test-notifications", "requested_severity": severity_upper},
    )
    console.print("[green]Done. Check configured channels for the test alert.[/green]")


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
    signals_dir = Path(dp_config.get("signals_dir", "data/signals"))
    history_path = Path(dp_config.get("history_file", "data/prediction_history.json"))
    output_dir = Path(dp_config.get("output_dir", "site"))
    lookback = dp_config.get("lookback_days", 252)

    signals_dir.mkdir(parents=True, exist_ok=True)

    # Resolve universe: dynamic fetch or fallback to config list
    universe_source = dp_config.get("universe_source")
    universe_extra = dp_config.get("universe_extra", [])
    universe_fallback = dp_config.get("universe", ["SPY", "QQQ", "IWM"])
    if universe_source:
        from pipeline.extract.universe import get_universe

        try:
            universe = get_universe(
                source=universe_source,
                extra=universe_extra,
                fallback=universe_fallback,
            )
            console.print(
                f"  Universe: {len(universe)} symbols "
                f"({universe_source} + {len(universe_extra)} extras)"
            )
        except Exception as exc:
            console.print(f"  [yellow]Universe fetch failed ({exc}), using fallback[/yellow]")
            universe = universe_fallback
    else:
        universe = universe_fallback
        console.print(f"  Universe: {len(universe)} symbols (config list)")

    # Signal date is resolved after loading price data so we can fall back
    # to the latest available trading day when no explicit date is given.
    explicit_date = pd.Timestamp(date) if date else None

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
        scores=scores,
    )
    console.print(f"[green]  Static site built at {site_path}/[/green]")

    # Print summary
    stats = tracker.get_stats()
    console.print("\n[bold green]Daily predictions complete![/bold green]")
    console.print(f"  Win rate: {stats['win_rate']}% ({stats['resolved']} resolved)")
    console.print(f"  View at: {output_dir}/index.html")


@app.command()
def backfill_predictions(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
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

    console.print(
        f"  Backfilling {len(trading_days)} trading days:"
        f" {trading_days[0].date()} → {trading_days[-1].date()}"
    )

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

            signals_df = format_signals(
                scores=eligible,
                price_data=indicator_data,
                date=signal_date,
            )
            if not signals_df.empty:
                write_signal_csv(signals_df, signals_dir, signal_date)
                added = tracker.add_signals(signals_df, str(signal_date.date()))
                total_signals += added

        # Resolve past outcomes using full price data (we have future data in backfill)
        tracker.resolve_outcomes(price_data, str(signal_date.date()))

        if (i + 1) % 20 == 0:
            console.print(
                f"    ... {i + 1}/{len(trading_days)} days processed,"
                f" {total_signals} signals so far"
            )

    tracker.save()

    # Build site with backfilled data
    build_static_site(output_dir=output_dir, signals_dir=signals_dir, history_path=history_path)

    stats = tracker.get_stats()
    console.print("\n[bold green]Backfill complete![/bold green]")
    console.print(f"  Total predictions: {stats['total']}")
    resolved = stats["resolved"]
    wins = stats["hit_target"]
    stopped = stats["stopped_out"]
    expired = stats["expired"]
    console.print(f"  Resolved: {resolved} ({wins} wins, {stopped} stopped, {expired} expired)")
    console.print(f"  Active: {stats['active']}")
    console.print(f"  Win rate: {stats['win_rate']}%")
    console.print(f"  Avg P&L: {stats['avg_pnl_pct']}%")


@app.command()
def reresolve_history(
    as_of: str = typer.Option(..., "--as-of", help="Evaluation date (YYYY-MM-DD)"),
    from_scratch: bool = typer.Option(
        False, "--from-scratch", help="Reset every outcome to active and re-resolve"
    ),
    history_file: str = typer.Option(
        "data/prediction_history.json", "--history", help="Source history JSON"
    ),
    out_file: str | None = typer.Option(
        None, "--out", help="Destination JSON. Default: <history>.v2.json"
    ),
    max_holding_bars: int = typer.Option(15, "--max-holding-bars", help="Holding limit in bars"),
    same_bar_policy: str = typer.Option(
        "stop_first", "--same-bar-policy", help="stop_first | target_first"
    ),
    cost_bps: float = typer.Option(3.0, "--cost-bps", help="Round-trip cost in basis points"),
):
    """Re-resolve prediction outcomes under the current resolution policy.

    Writes to a new file, leaving the source history untouched. Use this after
    changing the resolution rules so historical outcomes are measured the same
    way as new ones.
    """
    from pipeline.strategy.price_panel import load_ticker_frames
    from pipeline.web.outcome_resolution import ResolutionPolicy
    from pipeline.web.performance_tracker import PerformanceTracker

    console.print("[bold blue]Re-resolving prediction history...[/bold blue]")

    history_path = Path(history_file)
    if not history_path.exists():
        console.print(f"[red]No history file at {history_path}[/red]")
        raise typer.Exit(1)

    destination = Path(out_file) if out_file else history_path.with_suffix(".v2.json")
    shutil.copyfile(history_path, destination)

    policy = ResolutionPolicy(
        max_holding_bars=max_holding_bars,
        same_bar_policy=same_bar_policy,
        cost_bps=cost_bps,
    )
    tracker = PerformanceTracker(destination, policy=policy)

    if from_scratch:
        for pred in tracker.history.predictions:
            pred.update(
                outcome="active",
                resolved_date=None,
                resolved_price=None,
                pnl_pct=None,
                days_held=None,
            )
            for key in ("bars_held", "same_bar_ambiguous", "gapped", "unresolvable_reason"):
                pred.pop(key, None)
        console.print(f"  Reset {len(tracker.history.predictions)} predictions to active")

    tickers = {p["ticker"] for p in tracker.history.predictions}
    price_data, breaks = load_ticker_frames(tickers, Path("data/raw/prices"))
    console.print(f"  Loaded prices for {len(price_data)}/{len(tickers)} tickers")
    if breaks:
        console.print(
            f"  [yellow]Repaired {len(breaks)} spurious price discontinuit"
            f"{'y' if len(breaks) == 1 else 'ies'} across "
            f"{len({b.ticker for b in breaks})} ticker(s)[/yellow]"
        )

    summary = tracker.resolve_outcomes(price_data, as_of=as_of)
    tracker.save()

    stats = tracker.get_stats()
    console.print(f"\n[bold]Wrote {destination}[/bold]")
    console.print(f"  Resolved: {stats['resolved']}  Active: {stats['active']}")
    console.print(
        f"  hit_target {summary['hit_target']}  stopped_out {summary['stopped_out']}  "
        f"expired {summary['expired']}  unresolvable {summary['unresolvable']}"
    )
    console.print(
        f"  Target-hit rate: {stats['target_hit_rate']}%   "
        f"Profitable rate: {stats['profitable_rate']}%"
    )
    console.print(f"  Avg P&L: {stats['avg_pnl_pct']}%   Mean bars held: {stats['mean_bars_held']}")
    console.print(f"  Same-bar ambiguous: {stats['n_ambiguous']}   Gap fills: {stats['n_gapped']}")


@app.command()
def repair_price_adjustments(
    apply: bool = typer.Option(
        False, "--apply", help="Write the repair. Without this the command only reports."
    ),
    min_log_ratio: float = typer.Option(
        0.1, "--min-log-ratio", help="Ignore splits closer to 1.0 than this in log space"
    ),
    fit_margin: float = typer.Option(
        0.5, "--fit-margin", help="How much better the split-ratio fit must be than no-step"
    ),
    show: int = typer.Option(15, "--show", help="How many affected splits to list"),
):
    """Undo split adjustments that were applied to already-adjusted vendor prices.

    Repairs both raw_prices_ohlcv and cur_prices_ohlcv_daily. Re-run
    `transform` afterwards to rebuild the derived adjusted table.
    """
    from pipeline.transform.price_repair import repair_double_adjusted_splits

    console.print("[bold blue]Scanning for double-adjusted splits...[/bold blue]")

    db = get_db_manager()
    summary = repair_double_adjusted_splits(
        db, min_log_ratio=min_log_ratio, fit_margin=fit_margin, dry_run=not apply
    )

    if not summary["splits"]:
        console.print("[green]No double-adjusted splits found.[/green]")
        return

    console.print(
        f"  {summary['splits']} double-adjusted split(s) across "
        f"{summary['tickers']} ticker(s); {summary['bars']} bars need rescaling"
    )
    for brk in summary["breaks"][:show]:
        console.print(f"    {brk}")
    if len(summary["breaks"]) > show:
        console.print(f"    ... and {len(summary['breaks']) - show} more")

    if summary["dry_run"]:
        console.print("\n[yellow]Dry run — nothing written. Re-run with --apply.[/yellow]")
    else:
        console.print(
            f"\n[green]Repaired.[/green] curated rows updated: {summary['curated_rows']}, "
            f"raw rows updated: {summary['raw_rows']}"
        )
        console.print("  Run `make transform` to rebuild cur_prices_adjusted_daily.")


_PANEL_A_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLC",
    "XLY", "XLP", "XLB", "XLU", "XLRE", "TLT", "GLD",
]  # fmt: skip


@app.command()
def build_score_panel(
    start: str = typer.Option("2010-01-01", "--start", help="Panel start date"),
    end: str | None = typer.Option(None, "--end", help="Panel end date. Default: today"),
    universe: str = typer.Option(
        "etf",
        "--universe",
        help="'etf' (17-ETF Panel A, no survivorship bias) or 'all' (every raw-lake ticker, "
        "survivorship-contaminated -- use only to bound the bias, never to gate on)",
    ),
    out_dir: str = typer.Option("data/cache", "--out", help="Output directory"),
):
    """Build a historical score panel for signal validation.

    Writes wide DatetimeIndex x ticker parquet files (score, component points,
    entry_eligible, close) consumable by `test-signal-alpha` and the eval/
    IC/robustness tooling, none of which have ever been run against this
    strategy's actual score.
    """
    from pipeline.strategy.price_panel import load_ticker_frames
    from pipeline.strategy.signal_panel import build_score_panel as _build_panel
    from pipeline.strategy.signals import SignalEngine

    console.print(f"[bold blue]Building '{universe}' score panel...[/bold blue]")

    if universe == "etf":
        tickers = set(_PANEL_A_ETFS)
    elif universe == "all":
        raw_dir = Path("data/raw/prices")
        tickers = {f.stem.split("_")[0].upper() for f in raw_dir.glob("*.parquet")}
        console.print(
            "[yellow]Survivorship-contaminated: today's raw-lake tickers backfilled "
            "to the panel start. Use to bound bias, do not gate on this alone.[/yellow]"
        )
    else:
        console.print(f"[red]Unknown --universe {universe!r}; use 'etf' or 'all'[/red]")
        raise typer.Exit(1)

    tickers.add("SPY")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()

    frames, breaks = load_ticker_frames(tickers, start=start, end=end_ts)
    console.print(f"  Loaded {len(frames)}/{len(tickers)} tickers")
    if breaks:
        console.print(f"  Repaired {len(breaks)} price discontinuities")
    if "SPY" not in frames:
        console.print("[red]SPY failed to load; cannot classify regimes[/red]")
        raise typer.Exit(1)

    panel = _build_panel(frames, SignalEngine(), spy_prices=frames["SPY"]["close"])
    if panel.score.empty:
        console.print("[red]No scores produced[/red]")
        raise typer.Exit(1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{universe}_{start}_{end_ts.date()}"
    for field, df in panel.to_parquet_dict().items():
        path = out / f"score_panel_{tag}_{field}.parquet"
        df.to_parquet(path)
        console.print(f"  Wrote {path}  ({df.shape[0]} dates x {df.shape[1]} tickers)")


@app.command()
def validate_signal_score(
    start: str = typer.Option("2010-01-01", "--start", help="Validation window start"),
    end: str | None = typer.Option(None, "--end", help="Validation window end. Default: today"),
    universe: str = typer.Option(
        "etf", "--universe", help="'etf' (Panel A, no survivorship) or 'all'"
    ),
    train_size: int = typer.Option(504, "--train-size", help="Walk-forward training window"),
    test_size: int = typer.Option(63, "--test-size", help="Walk-forward test window"),
    embargo_size: int = typer.Option(15, "--embargo", help="Embargo days between train/test"),
    alpha: float = typer.Option(0.05, "--alpha", help="FDR target for the trial registry"),
    report_path: str | None = typer.Option(
        None, "--report", help="Write the full report as JSON to this path"
    ),
):
    """Run the full signal-validation harness and render the G3 verdict.

    Tests whether the QSG-MICRO-SWING-001 score has any monotone relationship
    with forward returns on real 2010-present history -- something that has
    never been checked. Every variant (the score, its 4 component buckets, and
    its 10 underlying boolean conditions) is registered as a trial and
    screened together via Benjamini-Hochberg, so nothing here is cherry-picked
    after the fact.
    """
    from pipeline.eval.signal_diagnostics import run_validation
    from pipeline.strategy.price_panel import load_ticker_frames
    from pipeline.strategy.signal_panel import build_score_panel
    from pipeline.strategy.signals import SignalEngine

    console.print(f"[bold blue]Validating signal score on '{universe}' universe...[/bold blue]")

    if universe == "etf":
        tickers = set(_PANEL_A_ETFS)
    elif universe == "all":
        raw_dir = Path("data/raw/prices")
        tickers = {f.stem.split("_")[0].upper() for f in raw_dir.glob("*.parquet")}
        console.print(
            "[yellow]Survivorship-contaminated universe; results are a bias bound.[/yellow]"
        )
    else:
        console.print(f"[red]Unknown --universe {universe!r}; use 'etf' or 'all'[/red]")
        raise typer.Exit(1)
    tickers.add("SPY")

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    frames, breaks = load_ticker_frames(tickers, start=start, end=end_ts)
    console.print(f"  Loaded {len(frames)}/{len(tickers)} tickers, {len(breaks)} price repairs")
    if "SPY" not in frames:
        console.print("[red]SPY failed to load; cannot classify regimes[/red]")
        raise typer.Exit(1)

    engine = SignalEngine()
    panel = build_score_panel(frames, engine, spy_prices=frames["SPY"]["close"])
    if panel.score.empty:
        console.print("[red]No scores produced[/red]")
        raise typer.Exit(1)

    report = run_validation(
        panel,
        frames,
        engine,
        train_size=train_size,
        test_size=test_size,
        embargo_size=embargo_size,
        alpha=alpha,
    )

    decay_table = Table(title="D1: IC Decay by Horizon")
    decay_table.add_column("Horizon (d)", justify="right")
    decay_table.add_column("Mean IC", justify="right")
    decay_table.add_column("IC IR", justify="right")
    for h in report.monotonicity.decay.horizons:
        marker = " *" if h == report.monotonicity.decay.best_horizon else ""
        decay_table.add_row(
            f"{h}{marker}",
            f"{report.monotonicity.decay.ic_by_horizon.get(h, float('nan')):.4f}",
            f"{report.monotonicity.decay.ic_ir_by_horizon.get(h, float('nan')):.4f}",
        )
    console.print(decay_table)
    lo, hi = report.monotonicity.daily_ic_ci
    console.print(f"  Daily-IC block-bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]\n")

    trial_table = Table(title="D2: Component Trial Registry (BH-screened)")
    trial_table.add_column("Trial", justify="left")
    trial_table.add_column("IC Mean", justify="right")
    trial_table.add_column("DSR Prob", justify="right")
    trial_table.add_column("Significant", justify="center")
    for result, significant in report.decomposition.screened:
        trial_table.add_row(
            result.signal_name,
            f"{result.ic_mean:.4f}" if pd.notna(result.ic_mean) else "N/A",
            (
                f"{result.deflated_sharpe_prob:.3f}"
                if pd.notna(result.deflated_sharpe_prob)
                else "N/A"
            ),
            "[green]yes[/green]" if significant else "no",
        )
    console.print(trial_table)
    if pd.notna(report.decomposition.first_pc_variance_ratio):
        console.print(
            "  First PC of the 4 oversold conditions explains "
            f"{report.decomposition.first_pc_variance_ratio:.1%} of their variance"
        )
    console.print(f"  Probability of backtest overfitting (PBO): {report.pbo:.3f}\n")

    verdict_color = {"PASS": "green", "INVERTED": "yellow", "INCONCLUSIVE": "red"}[report.verdict]
    console.print(f"[bold {verdict_color}]G3 VERDICT: {report.verdict}[/bold {verdict_color}]")
    for reason in report.reasoning:
        console.print(f"  - {reason}")

    if report_path:
        import json

        def _default(o):
            if isinstance(o, pd.DataFrame):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.__dict__, default=_default, indent=2))
        console.print(f"\nWrote full report to {report_path}")


@app.command()
def train_signal_model(
    start: str = typer.Option("2010-01-01", "--start", help="Development window start"),
    holdout_start: str = typer.Option(
        "2024-01-01", "--holdout-start", help="Untouched final block start"
    ),
    end: str | None = typer.Option(None, "--end", help="Panel end date. Default: today"),
    universe: str = typer.Option("etf", "--universe", help="'etf' (Panel A) or 'all'"),
    train_size: int = typer.Option(504, "--train-size", help="Walk-forward training window"),
    test_size: int = typer.Option(63, "--test-size", help="Walk-forward test window"),
    embargo_size: int = typer.Option(15, "--embargo", help="Embargo days"),
    max_trials: int = typer.Option(12, "--max-trials", help="Trial budget for the ladder"),
    alpha: float = typer.Option(0.05, "--alpha", help="FDR target"),
    phase3_report: str | None = typer.Option(
        None,
        "--phase3-report",
        help=(
            "Path to a validate-signal-score JSON report, whose trials are "
            "combined with Phase 5's for the full-registry BH screen (G5 criterion C3)"
        ),
    ),
    explore_lightgbm: bool = typer.Option(
        False,
        "--explore-lightgbm",
        help=(
            "Also run LightGBM directly for every feature-set/horizon combo, "
            "bypassing the complexity ladder's rule that blocks it when no "
            "logistic config beat baseline. Explicitly exploratory: these "
            "trials are registered and BH-screened alongside everything else "
            "(raising, not lowering, the bar to survive correction), but do "
            "not retroactively unlock Gate G5 if the disciplined ladder "
            "already said no."
        ),
    ),
    report_path: str | None = typer.Option(
        None, "--report", help="Write the full report as JSON to this path"
    ),
):
    """Phase 5: fit a model on continuous features and evaluate against Gate G5.

    Entered only because validate-signal-score returned INCONCLUSIVE. Runs the
    complexity ladder (baseline -> logistic -> LightGBM, budget-capped and
    BH-screened) on a DEV set, then evaluates the winning config exactly once
    on an untouched holdout block. Missing any one of the eight G5 criteria
    means shipping nothing -- the site stays UNRATED regardless.
    """
    from pipeline.eval.signal_alpha import SignalAlphaResult
    from pipeline.strategy.price_panel import load_ticker_frames
    from pipeline.strategy.signal_model import run_phase5

    console.print(
        f"[bold blue]Phase 5: training signal model on '{universe}' universe...[/bold blue]"
    )

    if universe == "etf":
        tickers = set(_PANEL_A_ETFS)
    elif universe == "all":
        raw_dir = Path("data/raw/prices")
        tickers = {f.stem.split("_")[0].upper() for f in raw_dir.glob("*.parquet")}
        console.print("[yellow]Survivorship-contaminated universe.[/yellow]")
    else:
        console.print(f"[red]Unknown --universe {universe!r}; use 'etf' or 'all'[/red]")
        raise typer.Exit(1)
    tickers.add("SPY")

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    frames, breaks = load_ticker_frames(tickers, start=start, end=end_ts)
    console.print(f"  Loaded {len(frames)}/{len(tickers)} tickers, {len(breaks)} price repairs")

    price_panel = pd.DataFrame({t: df["close"] for t, df in frames.items()})

    phase3_trials: list[SignalAlphaResult] = []
    if phase3_report:
        import json as _json

        raw = _json.loads(Path(phase3_report).read_text())
        for r in raw.get("decomposition", {}).get("trials", []):
            phase3_trials.append(
                SignalAlphaResult(
                    **{k: r[k] for k in SignalAlphaResult.__dataclass_fields__ if k in r}
                )
            )
        score_r = raw.get("score_result")
        if score_r:
            phase3_trials.append(
                SignalAlphaResult(
                    **{
                        k: score_r[k]
                        for k in SignalAlphaResult.__dataclass_fields__
                        if k in score_r
                    }
                )
            )
        console.print(f"  Loaded {len(phase3_trials)} Phase 3 trials for the combined BH screen")
    else:
        console.print(
            "[yellow]No --phase3-report given; BH screen covers only Phase 5's trials, "
            "understating the true trial count.[/yellow]"
        )

    report = run_phase5(
        frames, price_panel, phase3_trials=phase3_trials, holdout_start=holdout_start,
        train_size=train_size, test_size=test_size, embargo_size=embargo_size,
        max_trials=max_trials, alpha=alpha, explore_lightgbm=explore_lightgbm,
    )  # fmt: skip

    ladder_table = Table(title="Complexity Ladder (DEV only)")
    ladder_table.add_column("Trial", justify="left")
    ladder_table.add_column("OOS Log-Loss", justify="right")
    ladder_table.add_column("Baseline LL", justify="right")
    ladder_table.add_column("Beats Baseline", justify="center")
    for t in report.ladder.dev_trials:
        ladder_table.add_row(
            t.trial_name,
            f"{t.oos_log_loss:.4f}" if pd.notna(t.oos_log_loss) else "N/A",
            f"{t.baseline_log_loss:.4f}" if pd.notna(t.baseline_log_loss) else "N/A",
            "[green]yes[/green]" if t.beats_baseline else "no",
        )
    console.print(ladder_table)
    console.print(f"  PBO: {report.ladder.pbo:.3f}")

    if report.ladder.best_trial:
        console.print(f"  Best DEV candidate: [bold]{report.ladder.best_trial.trial_name}[/bold]")
        console.print(f"  Same-bar policy stable: {report.same_bar_stable}")
        if report.holdout_result:
            hr = report.holdout_result
            console.print(
                f"  Holdout (2024+): log-loss={hr['holdout_log_loss']:.4f} "
                f"vs baseline={hr['baseline_log_loss']:.4f}, beats={hr['beats_baseline']}"
            )
    else:
        console.print("  [yellow]No candidate beat the baseline on DEV.[/yellow]")

    if report.lightgbm_exploration:
        explore_table = Table(
            title="LightGBM Exploration (bypasses the ladder gate -- see help text)"
        )
        explore_table.add_column("Trial", justify="left")
        explore_table.add_column("OOS Log-Loss", justify="right")
        explore_table.add_column("Baseline LL", justify="right")
        explore_table.add_column("Beats Baseline", justify="center")
        for t in report.lightgbm_exploration:
            explore_table.add_row(
                t.trial_name,
                f"{t.oos_log_loss:.4f}" if pd.notna(t.oos_log_loss) else "N/A",
                f"{t.baseline_log_loss:.4f}" if pd.notna(t.baseline_log_loss) else "N/A",
                "[green]yes[/green]" if t.beats_baseline else "no",
            )
        console.print(explore_table)
        console.print(
            "  [dim]These trials are included in the BH screen below but do not "
            "unlock Gate G5 on their own -- an unblinded look after the disciplined "
            "ladder already said no.[/dim]"
        )

    g5_table = Table(title="Gate G5 Criteria")
    g5_table.add_column("Criterion")
    g5_table.add_column("Met", justify="center")
    for name, met in report.g5.criteria.items():
        g5_table.add_row(name, "[green]yes[/green]" if met else "[red]no[/red]")
    console.print(g5_table)

    verdict_color = "green" if report.g5.passed else "red"
    verdict = "SHIP" if report.g5.passed else "SHIP NOTHING"
    console.print(f"\n[bold {verdict_color}]G5 VERDICT: {verdict}[/bold {verdict_color}]")
    for reason in report.g5.reasoning:
        console.print(f"  - {reason}")

    if report.lightgbm_exploration:
        from pipeline.eval.signal_alpha import signal_fdr_screen

        combined = list(phase3_trials) + report.ladder.registry.trials
        screened = signal_fdr_screen(combined, alpha=alpha)
        n_survived = sum(1 for _, sig in screened if sig)
        console.print(
            f"\n  Combined BH screen across all {len(combined)} trials "
            f"(Phase 3 + ladder + exploration): {n_survived} survive at alpha={alpha}"
        )

    if report_path:
        import json

        def _default(o):
            if isinstance(o, pd.DataFrame | pd.Series):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return {
                    k: v
                    for k, v in o.__dict__.items()
                    if k not in ("oos_predictions", "oos_labels")
                }
            return str(o)

        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.__dict__, default=_default, indent=2))
        console.print(f"\nWrote full report to {report_path}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
