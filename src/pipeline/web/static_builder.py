"""Static site builder: generates HTML pages from daily signal predictions.

Reads signal CSVs and prediction history, then renders Jinja2 templates into
a static site directory suitable for GitHub Pages deployment.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()).get("daily_predictions", {})
    return {}


def _find_latest_signal_csv(signals_dir: Path) -> Path | None:
    """Find the most recent signal CSV file."""
    csvs = sorted(signals_dir.glob("signals_*.csv"), reverse=True)
    return csvs[0] if csvs else None


def _outcome_class(outcome: str) -> str:
    """Map outcome to CSS class name."""
    return {
        "hit_target": "outcome-win",
        "stopped_out": "outcome-loss",
        "expired": "outcome-expired",
        "active": "outcome-active",
    }.get(outcome, "")


def _score_class(score: int) -> str:
    """Map signal score to CSS class for color coding."""
    if score >= 80:
        return "score-high"
    if score >= 60:
        return "score-medium"
    return "score-low"


def _pnl_class(pnl: float | None) -> str:
    if pnl is None:
        return ""
    return "pnl-positive" if pnl >= 0 else "pnl-negative"


def build_static_site(
    output_dir: str | Path = "site",
    signals_dir: str | Path = "data/signals",
    history_path: str | Path = "data/prediction_history.json",
) -> Path:
    """Build the complete static site.

    Args:
        output_dir: Where to write the static HTML files.
        signals_dir: Directory containing signal CSV files.
        history_path: Path to prediction_history.json.

    Returns:
        Path to the output directory.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for the web module. "
            "Install with: pip install market-data-warehouse[web]"
        ) from e

    output_dir = Path(output_dir)
    signals_dir = Path(signals_dir)
    history_path = Path(history_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load templates
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,
    )
    env.filters["outcome_class"] = _outcome_class
    env.filters["score_class"] = _score_class
    env.filters["pnl_class"] = _pnl_class
    env.globals["now"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Load latest signals
    latest_csv = _find_latest_signal_csv(signals_dir)
    signals_df = pd.DataFrame()
    signal_date = "N/A"
    if latest_csv:
        signals_df = pd.read_csv(latest_csv)
        # Extract date from filename: signals_YYYY-MM-DD.csv
        try:
            signal_date = latest_csv.stem.replace("signals_", "")
        except Exception:
            signal_date = "unknown"
        logger.info("Loaded %d signals from %s", len(signals_df), latest_csv.name)
    else:
        logger.warning("No signal CSVs found in %s", signals_dir)

    # Load prediction history
    history_data: dict = {"predictions": [], "last_updated": ""}
    stats: dict = {
        "total": 0, "active": 0, "resolved": 0, "hit_target": 0,
        "stopped_out": 0, "expired": 0, "win_rate": 0.0,
        "avg_pnl_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
    }
    if history_path.exists():
        try:
            history_data = json.loads(history_path.read_text())
            from pipeline.web.performance_tracker import PerformanceTracker

            tracker = PerformanceTracker(history_path)
            stats = tracker.get_stats()
        except Exception as exc:
            logger.warning("Could not load history: %s", exc)

    predictions = history_data.get("predictions", [])

    # Get unique tickers from history
    tickers = sorted({p["ticker"] for p in predictions})

    # Build context shared across all pages
    config = _load_config()
    universe = config.get("universe", [])
    base_ctx = {
        "signal_date": signal_date,
        "universe": universe,
        "tickers": tickers,
        "stats": stats,
    }

    # --- Render index.html (dashboard) ---
    signals_list = signals_df.to_dict("records") if not signals_df.empty else []
    index_tmpl = env.get_template("index.html")
    index_html = index_tmpl.render(signals=signals_list, **base_ctx)
    (output_dir / "index.html").write_text(index_html)
    logger.info("Wrote index.html (%d signals)", len(signals_list))

    # --- Render history.html ---
    recent = sorted(predictions, key=lambda p: p["signal_date"], reverse=True)[:100]
    history_tmpl = env.get_template("history.html")
    history_html = history_tmpl.render(predictions=recent, **base_ctx)
    (output_dir / "history.html").write_text(history_html)
    logger.info("Wrote history.html (%d predictions)", len(recent))

    # --- Render performance.html ---
    perf_tmpl = env.get_template("performance.html")
    # Group by month for monthly breakdown
    monthly: dict[str, dict] = {}
    for p in predictions:
        if p.get("outcome") == "active":
            continue
        month = p["signal_date"][:7]  # YYYY-MM
        if month not in monthly:
            monthly[month] = {"total": 0, "wins": 0, "pnl_sum": 0.0}
        monthly[month]["total"] += 1
        if p.get("outcome") == "hit_target":
            monthly[month]["wins"] += 1
        if p.get("pnl_pct") is not None:
            monthly[month]["pnl_sum"] += p["pnl_pct"]

    monthly_stats = []
    for month in sorted(monthly.keys(), reverse=True):
        m = monthly[month]
        monthly_stats.append({
            "month": month,
            "total": m["total"],
            "wins": m["wins"],
            "win_rate": round(m["wins"] / m["total"] * 100, 1) if m["total"] else 0,
            "total_pnl": round(m["pnl_sum"], 2),
        })

    perf_html = perf_tmpl.render(monthly_stats=monthly_stats, **base_ctx)
    (output_dir / "performance.html").write_text(perf_html)
    logger.info("Wrote performance.html")

    # --- Render per-ticker pages ---
    ticker_dir = output_dir / "ticker"
    ticker_dir.mkdir(exist_ok=True)
    ticker_tmpl = env.get_template("ticker.html")
    for ticker in tickers:
        ticker_preds = [p for p in predictions if p["ticker"] == ticker]
        ticker_preds.sort(key=lambda p: p["signal_date"], reverse=True)
        ticker_stats = _compute_ticker_stats(ticker_preds)
        ticker_html = ticker_tmpl.render(
            ticker=ticker,
            predictions=ticker_preds,
            ticker_stats=ticker_stats,
            **base_ctx,
        )
        (ticker_dir / f"{ticker}.html").write_text(ticker_html)

    logger.info("Wrote %d ticker pages", len(tickers))

    # Copy static assets if any
    assets_src = template_dir / "assets"
    if assets_src.exists():
        assets_dst = output_dir / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)

    logger.info("Static site built in %s", output_dir)
    return output_dir


def _compute_ticker_stats(predictions: list[dict]) -> dict:
    """Compute stats for a single ticker's predictions."""
    total = len(predictions)
    if total == 0:
        return {"total": 0, "win_rate": 0.0, "avg_pnl": 0.0}
    hit = sum(1 for p in predictions if p.get("outcome") == "hit_target")
    resolved = sum(1 for p in predictions if p.get("outcome") != "active")
    pnls = [p["pnl_pct"] for p in predictions if p.get("pnl_pct") is not None]
    return {
        "total": total,
        "resolved": resolved,
        "hit_target": hit,
        "win_rate": round(hit / resolved * 100, 1) if resolved else 0.0,
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "avg_score": round(
            sum(p["score"] for p in predictions) / total, 1
        ),
    }
