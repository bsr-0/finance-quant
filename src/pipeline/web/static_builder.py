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
        return (yaml.safe_load(config_path.read_text()) or {}).get("daily_predictions", {})
    return {}


def _load_validation_summary(reports_dir: str | Path = "reports") -> dict | None:
    """Load the score (G3) and model (G5) validation reports, if present.

    Both are produced by ``validate-signal-score`` and ``train-signal-model``
    and are not checked in (``reports/`` is gitignored, since they're
    regenerable and tied to whatever data the pipeline held at run time).
    Returns ``None`` if neither report exists, so the site can say plainly
    that validation hasn't been run rather than fabricate a status.
    """
    reports_dir = Path(reports_dir)
    g3_path = reports_dir / "signal_validation.json"
    g5_candidates = sorted(reports_dir.glob("phase5_validation*.json"), reverse=True)

    g3 = None
    if g3_path.exists():
        try:
            raw = json.loads(g3_path.read_text())
            decomp = raw.get("decomposition", {})
            g3 = {
                "verdict": raw.get("verdict"),
                "reasoning": raw.get("reasoning", []),
                "ic_mean": raw.get("score_result", {}).get("ic_mean"),
                "dsr_prob": raw.get("score_result", {}).get("deflated_sharpe_prob"),
                "n_folds": raw.get("score_result", {}).get("n_folds"),
                "daily_ic_ci": raw.get("monotonicity", {}).get("daily_ic_ci"),
                "n_trials": len(decomp.get("trials", [])),
                "n_bh_survivors": sum(1 for _, sig in decomp.get("screened", []) if sig),
                "first_pc_variance_ratio": decomp.get("first_pc_variance_ratio"),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Could not parse %s", g3_path)

    g5 = None
    if g5_candidates:
        try:
            raw = json.loads(g5_candidates[0].read_text())
            g5_block = raw.get("g5", {})
            n_lgbm = len(raw.get("lightgbm_exploration", []))
            g5 = {
                "passed": g5_block.get("passed"),
                "criteria": g5_block.get("criteria", {}),
                "reasoning": g5_block.get("reasoning", []),
                "n_dev_trials": len(raw.get("ladder", {}).get("dev_trials", [])),
                "n_lightgbm_explored": n_lgbm,
                "pbo": raw.get("ladder", {}).get("pbo"),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Could not parse %s", g5_candidates[0])

    if g3 is None and g5 is None:
        return None
    return {"g3": g3, "g5": g5}


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


def _score_class(score: int, mode: str = "unrated") -> str:
    """Map signal score to a CSS class for colour coding.

    Colour-coding by score makes the same unvalidated quality claim as the
    confidence label, just in CSS.  Under ``unrated`` every score renders
    neutral; ``legacy`` restores the original banding.
    """
    if mode != "legacy":
        return "score-neutral"
    if score >= 80:
        return "score-high"
    if score >= 60:
        return "score-medium"
    return "score-low"


def _pnl_class(pnl: float | None) -> str:
    if pnl is None:
        return ""
    return "pnl-positive" if pnl >= 0 else "pnl-negative"


# ---------------------------------------------------------------------------
# Dashboard metadata (static descriptions of the pipeline architecture)
# ---------------------------------------------------------------------------

_DATA_SOURCES = [
    {
        "name": "Yahoo Finance",
        "category": "Prices",
        "description": "Daily OHLCV for ETFs and equities with corporate actions adjustment.",
        "dot": "dot-green",
        "badge_class": "badge-green",
        "latency": "~15 min",
    },
    {
        "name": "FRED",
        "category": "Macro",
        "description": "28 economic series: GDP, CPI, unemployment, yields, VIX, and more.",
        "dot": "dot-green",
        "badge_class": "badge-blue",
        "latency": "~1 day",
    },
    {
        "name": "GDELT",
        "category": "Events",
        "description": "Global geopolitical events with actor, location, and tone scoring.",
        "dot": "dot-green",
        "badge_class": "badge-yellow",
        "latency": "~3 hours",
    },
    {
        "name": "Polymarket",
        "category": "Prediction",
        "description": "Prediction market prices, trades, and orderbook snapshots.",
        "dot": "dot-green",
        "badge_class": "badge-yellow",
        "latency": "~2 min",
    },
    {
        "name": "SEC EDGAR",
        "category": "Fundamentals",
        "description": "Company fundamentals, insider trades, and 13F institutional holdings.",
        "dot": "dot-green",
        "badge_class": "badge-blue",
        "latency": "~1 day",
    },
    {
        "name": "Options Chains",
        "category": "Derivatives",
        "description": "Strike-level options data with implied volatility and open interest.",
        "dot": "dot-green",
        "badge_class": "badge-muted",
        "latency": "~15 min",
    },
    {
        "name": "Reddit Sentiment",
        "category": "Sentiment",
        "description": "Posts from r/wallstreetbets, r/stocks, r/investing with ticker extraction.",
        "dot": "dot-yellow",
        "badge_class": "badge-yellow",
        "latency": "~5 min",
    },
    {
        "name": "Fama-French",
        "category": "Factors",
        "description": "6-factor model returns: MKT, SMB, HML, RMW, CMA, MOM.",
        "dot": "dot-green",
        "badge_class": "badge-muted",
        "latency": "~1 day",
    },
]

_STRATEGY_COMPONENTS = [
    {
        "name": "Trend Alignment",
        "weight": 40,
        "description": "Close > SMA50, SMA50 > SMA200, positive slope confirmation.",
    },
    {
        "name": "Pullback Depth",
        "weight": 30,
        "description": "RSI < 35, price at Bollinger lower band, Stochastic K < 20.",
    },
    {
        "name": "Volume Confirmation",
        "weight": 15,
        "description": "Volume below 20-day MA (dry-up), positive OBV slope.",
    },
    {
        "name": "Volatility & Momentum",
        "weight": 15,
        "description": "ATR within range, MACD histogram rising, Williams %R < -80.",
    },
]

_DQ_GATES = [
    {
        "name": "Freshness",
        "description": "Prices < 48h old, contracts < 2h, macro < 168h.",
        "dot": "dot-green",
    },
    {
        "name": "Completeness",
        "description": "Min 95% non-null in required columns.",
        "dot": "dot-green",
    },
    {
        "name": "Time Monotonicity",
        "description": "available_time >= event_time for all rows.",
        "dot": "dot-green",
    },
    {
        "name": "PK Uniqueness",
        "description": "No duplicate primary keys across curated tables.",
        "dot": "dot-green",
    },
    {
        "name": "Referential Integrity",
        "description": "All foreign keys resolve; no orphan records.",
        "dot": "dot-green",
    },
]

_RAW_TABLES = [
    {
        "name": "raw_fred_observations",
        "source": "FRED",
        "pk": "series_code, date, realtime_start",
    },
    {"name": "raw_gdelt_events", "source": "GDELT", "pk": "gdelt_event_id"},
    {"name": "raw_polymarket_markets", "source": "Polymarket", "pk": "venue_market_id"},
    {
        "name": "raw_polymarket_prices",
        "source": "Polymarket",
        "pk": "venue_market_id, ts, outcome",
    },
    {"name": "raw_polymarket_trades", "source": "Polymarket", "pk": "trade_id"},
    {"name": "raw_prices_ohlcv", "source": "Yahoo Finance", "pk": "ticker, date"},
    {"name": "raw_factor_returns", "source": "Fama-French", "pk": "date"},
    {
        "name": "raw_sec_fundamentals",
        "source": "SEC EDGAR",
        "pk": "ticker, metric, period, form, accession",
    },
    {
        "name": "raw_sec_insider_trades",
        "source": "SEC EDGAR",
        "pk": "accession, insider_cik, date, type",
    },
    {
        "name": "raw_options_chain",
        "source": "Yahoo Finance",
        "pk": "ticker, date, expiry, strike, type",
    },
    {"name": "raw_earnings_calendar", "source": "Yahoo Finance", "pk": "ticker, report_date"},
    {"name": "raw_reddit_posts", "source": "Reddit", "pk": "post_id"},
    {"name": "raw_short_interest", "source": "FINRA", "pk": "ticker, settlement_date"},
    {"name": "raw_etf_flows", "source": "Yahoo Finance", "pk": "ticker, date"},
]

_CURATED_TABLES = [
    {
        "name": "cur_prices_ohlcv_daily",
        "description": "Adjusted daily prices with corporate actions",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_macro_observations",
        "description": "FRED macro data with revision tracking",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_world_events",
        "description": "GDELT events with latency-aware timestamps",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_contract_prices",
        "description": "Polymarket prices normalized 0-1",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_contract_trades",
        "description": "Polymarket trade executions",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_contract_state_daily",
        "description": "Daily contract status snapshots",
        "quality": "inferred",
        "badge": "badge-yellow",
    },
    {
        "name": "cur_factor_returns_daily",
        "description": "Fama-French 6-factor daily returns",
        "quality": "confirmed",
        "badge": "badge-green",
    },
    {
        "name": "cur_corporate_actions",
        "description": "Stock splits and dividend adjustments",
        "quality": "confirmed",
        "badge": "badge-green",
    },
]

_FEATURE_FAMILIES = [
    {
        "name": "Trend",
        "indicators": ["SMA(20)", "SMA(50)", "SMA(200)", "EMA(12)", "EMA(26)"],
    },
    {
        "name": "Momentum",
        "indicators": ["RSI(14)", "MACD", "Stochastic(14,3)", "Williams %R", "ROC"],
    },
    {
        "name": "Volatility",
        "indicators": ["Bollinger Bands", "ATR(14)", "Realized Vol", "Parkinson Vol"],
    },
    {
        "name": "Volume",
        "indicators": ["OBV", "Volume SMA(20)", "Volume ratio"],
    },
    {
        "name": "Risk",
        "indicators": ["VaR(95%)", "Max Drawdown", "Yang-Zhang Vol", "Garman-Klass Vol"],
    },
    {
        "name": "Seasonal",
        "indicators": ["Day-of-week", "Month", "Quarter-end", "Week-of-year"],
    },
]

_DQ_CHECKS = [
    {
        "name": "Freshness",
        "severity": "CRITICAL",
        "description": "Data within staleness threshold per source.",
        "badge": "badge-green",
    },
    {
        "name": "Completeness",
        "severity": "ERROR",
        "description": "Min 95% non-null in required columns.",
        "badge": "badge-yellow",
    },
    {
        "name": "Time Monotonicity",
        "severity": "CRITICAL",
        "description": "available_time >= event_time in all tables.",
        "badge": "badge-green",
    },
    {
        "name": "PK Uniqueness",
        "severity": "CRITICAL",
        "description": "No duplicate composite primary keys.",
        "badge": "badge-green",
    },
    {
        "name": "Referential Integrity",
        "severity": "ERROR",
        "description": "Foreign keys resolve to valid parent rows.",
        "badge": "badge-yellow",
    },
    {
        "name": "Coverage Sanity",
        "severity": "WARNING",
        "description": "No negative prices/volume, prices in [0,1].",
        "badge": "badge-muted",
    },
    {
        "name": "OHLC Logic",
        "severity": "ERROR",
        "description": "low <= open,close <= high for all bars.",
        "badge": "badge-yellow",
    },
]


def _radar_rows(scores: list) -> list[dict]:
    """Convert SignalScore objects to dicts with a human-readable blocker field."""
    rows = []
    for s in scores:
        blockers: list[str] = []
        if not s.entry_eligible:
            if s.regime == "BEAR":
                blockers.append("Bear regime")
            if s.trend_pts < 25:
                blockers.append("Weak trend")
            if s.pullback_pts == 0:
                blockers.append("No pullback")
            if not blockers:
                blockers.append("Score too low")
        rows.append(
            {
                "symbol": s.symbol,
                "score": s.score,
                "trend_pts": s.trend_pts,
                "pullback_pts": s.pullback_pts,
                "volume_pts": s.volume_pts,
                "volatility_pts": s.volatility_pts,
                "regime": s.regime,
                "entry_eligible": s.entry_eligible,
                "blocker": ", ".join(blockers),
            }
        )
    return rows


def build_static_site(
    output_dir: str | Path = "site",
    signals_dir: str | Path = "data/signals",
    history_path: str | Path = "data/prediction_history.json",
    scores: list | None = None,
) -> Path:
    """Build the complete static site.

    Args:
        output_dir: Where to write the static HTML files.
        signals_dir: Directory containing signal CSV files.
        history_path: Path to prediction_history.json.
        scores: Optional list of SignalScore objects from today's run, used
            to render the Signal Radar section on the dashboard.

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

    # Load latest signals — scan CSVs newest-first and keep only signals that
    # are still active (not yet resolved in prediction_history.json).
    resolved_keys: set[tuple[str, str]] = set()
    if history_path.exists():
        try:
            _hist = json.loads(history_path.read_text())
            for p in _hist.get("predictions", []):
                if p.get("outcome") and p["outcome"] != "active":
                    resolved_keys.add((p["signal_date"], p["ticker"]))
        except Exception:
            pass

    signals_df = pd.DataFrame()
    signal_date = "N/A"
    for csv_path in sorted(signals_dir.glob("signals_*.csv"), reverse=True):
        df = pd.read_csv(csv_path)
        date_str = csv_path.stem.replace("signals_", "")
        # Normalise to YYYY-MM-DD for comparison with history keys
        try:
            date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            date_fmt = date_str
        # Drop rows already resolved
        if not df.empty and "ticker" in df.columns:
            df = df[~df["ticker"].apply(lambda t, k=date_fmt: (k, t) in resolved_keys)]
        if not df.empty:
            signals_df = df
            signal_date = date_fmt  # YYYY-MM-DD
            logger.info("Loaded %d signals from %s", len(df), csv_path.name)
            break
    else:
        logger.warning("No active signals found in %s", signals_dir)

    # Load prediction history
    history_data: dict = {"predictions": [], "last_updated": ""}
    stats: dict = {
        "total": 0,
        "active": 0,
        "resolved": 0,
        "hit_target": 0,
        "stopped_out": 0,
        "expired": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
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
    tickers = sorted({p["ticker"] for p in predictions})

    config = _load_config()
    universe = config.get("universe", [])
    signals_list = signals_df.to_dict("records") if not signals_df.empty else []
    indicator_count = sum(len(f["indicators"]) for f in _FEATURE_FAMILIES)

    validation = _load_validation_summary()

    base_ctx = {
        "signal_date": signal_date,
        "universe": universe,
        "tickers": tickers,
        "stats": stats,
        "signals": signals_list,
        "radar": _radar_rows(scores) if scores else [],
        "radar_date": (str(scores[0].date)[:10] if scores else signal_date),
        "data_sources": _DATA_SOURCES,
        "strategy_components": _STRATEGY_COMPONENTS,
        "dq_gates": _DQ_GATES,
        "indicator_count": indicator_count,
        "raw_tables": _RAW_TABLES,
        "curated_tables": _CURATED_TABLES,
        "feature_families": _FEATURE_FAMILIES,
        "dq_checks": _DQ_CHECKS,
        "validation": validation,
    }

    # --- Render index.html (dashboard) ---
    index_tmpl = env.get_template("index.html")
    index_html = index_tmpl.render(**base_ctx)
    (output_dir / "index.html").write_text(index_html)
    logger.info("Wrote index.html (%d signals)", len(signals_list))

    # --- Render pipeline.html ---
    pipeline_tmpl = env.get_template("pipeline.html")
    pipeline_html = pipeline_tmpl.render(**base_ctx)
    (output_dir / "pipeline.html").write_text(pipeline_html)
    logger.info("Wrote pipeline.html")

    # --- Render history.html ---
    recent = sorted(predictions, key=lambda p: p["signal_date"], reverse=True)[:100]
    history_tmpl = env.get_template("history.html")
    history_html = history_tmpl.render(predictions=recent, **base_ctx)
    (output_dir / "history.html").write_text(history_html)
    logger.info("Wrote history.html (%d predictions)", len(recent))

    # --- Render performance.html ---
    perf_tmpl = env.get_template("performance.html")
    monthly: dict[str, dict] = {}
    for p in predictions:
        if p.get("outcome") == "active":
            continue
        month = p["signal_date"][:7]
        if month not in monthly:
            monthly[month] = {"total": 0, "wins": 0, "pnl_sum": 0.0}
        monthly[month]["total"] += 1
        # "Win" means profitable, matching PerformanceTracker.get_stats --
        # a profitable expiry counts even though it never touched the target.
        if p.get("pnl_pct") is not None and p["pnl_pct"] > 0:
            monthly[month]["wins"] += 1
        if p.get("pnl_pct") is not None:
            monthly[month]["pnl_sum"] += p["pnl_pct"]

    monthly_stats = []
    for month in sorted(monthly.keys(), reverse=True):
        m = monthly[month]
        monthly_stats.append(
            {
                "month": month,
                "total": m["total"],
                "wins": m["wins"],
                "win_rate": round(m["wins"] / m["total"] * 100, 1) if m["total"] else 0,
                "total_pnl": round(m["pnl_sum"], 2),
            }
        )

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
    profitable = sum(1 for p in pnls if p > 0)
    # Matches PerformanceTracker.get_stats: "win" means profitable, which
    # includes profitable expiries. Previously this used hit/resolved while the
    # site-wide stat used a different rule, so the two disagreed.
    return {
        "total": total,
        "resolved": resolved,
        "hit_target": hit,
        "win_rate": round(profitable / len(pnls) * 100, 1) if pnls else 0.0,
        "target_hit_rate": round(hit / resolved * 100, 1) if resolved else 0.0,
        "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "avg_score": round(sum(p["score"] for p in predictions) / total, 1),
    }
