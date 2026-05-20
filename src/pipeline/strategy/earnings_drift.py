"""QSG-EARNINGS-DRIFT-001: Post-Earnings Announcement Drift (PEAD).

Exploits the empirically well-documented tendency for stocks to continue
trending in the direction of an earnings surprise for 5-30 days after the
announcement date.

Key references:
  Ball & Brown (1968) — "An Empirical Evaluation of Accounting Income Numbers"
  Foster, Olsen & Shevlin (1984) — "Earnings Releases, Anomalies, and the
    Behavior of Security Returns"
  Chan, Jegadeesh & Lakonishok (1996) — "Momentum Strategies"

Signal construction
-------------------
  score(i, t) = eps_surprise_pct(i, q) × recency_decay(t)
  recency_decay = max(0, 1 − days_since_report / lookback_days)

Only tickers with |eps_surprise_pct| ≥ min_surprise_pct produce a non-zero
score. The strategy is long-only; negative surprise scores are flat.

Entry: score > 0 (positive surprise within lookback window), BULL/NEUTRAL regime
Exit: 15-day max hold, 2× ATR stop, 3.5× ATR target

Data source
-----------
Reads parquet files written by ``EarningsExtractor`` (extract/earnings.py):
  data/raw/earnings/{TICKER}_{start}_{end}.parquet

Run ``make extract-all`` or call the extractor directly before backtesting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.strategy.backtest_harness import BacktestConfig
from pipeline.strategy.benchmark import US_EQUITY_BENCHMARKS
from pipeline.strategy.entry_rules import institutional_entry_rules
from pipeline.strategy.exits import ExitEngine
from pipeline.strategy.position_sizing import (
    InstitutionalSizingConfig,
    SizingMethod,
    create_sizer,
)
from pipeline.strategy.risk_constraints import institutional_constraints
from pipeline.strategy.signal_library import (
    NormalizationMethod,
    RawIndicator,
    SignalConfig,
    SignalDefinition,
    SignalFamily,
)
from pipeline.strategy.strategy_definition import StrategyDefinition, StrategyThesis
from pipeline.strategy.universe import US_LARGE_CAP_EQUITY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Macro regime filter
# ---------------------------------------------------------------------------


def load_rate_hike_regime(
    rate_hike_threshold_bps: float = 75.0,
    lookback_months: int = 3,
) -> pd.Series:
    """Load a boolean daily Series: True = Fed is actively hiking (block PEAD entries).

    Uses FEDFUNDS from the warehouse DB, forward-filled to daily frequency.
    Falls back to yfinance's ^IRX (13-week T-bill) if DB has no data.

    Parameters
    ----------
    rate_hike_threshold_bps:
        If the Fed Funds Rate has risen by more than this many basis points
        over the past ``lookback_months`` months, mark the day as a rate-hike
        regime.  Default 150 bps (1.50 percentage points).
    lookback_months:
        Rolling window (in months) for measuring the rate change (default 12).

    Returns
    -------
    pd.Series[bool]
        Daily index (all trading days), True where rate-hike regime is active.
    """
    df: pd.DataFrame | None = None

    try:
        from pipeline.db import get_db_manager

        db = get_db_manager()
        rows = db.run_query("""
            SELECT mo.event_time, mo.value
            FROM cur_macro_observations mo
            JOIN dim_macro_series ms ON ms.series_id = mo.series_id
            WHERE ms.provider_series_code = 'FEDFUNDS'
            ORDER BY mo.event_time
        """)
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["event_time"]).dt.tz_localize(None).dt.normalize()
            df["rate"] = pd.to_numeric(df["value"], errors="coerce")
            df = (
                df[["date", "rate"]]
                .dropna()
                .drop_duplicates("date")
                .sort_values("date")
                .set_index("date")
            )
    except Exception as exc:
        logger.warning("Could not load FEDFUNDS from DB: %s — falling back to yfinance", exc)

    if df is None or df.empty:
        try:
            import yfinance as yf

            irx = yf.download("^IRX", start="2000-01-01", progress=False)["Close"]
            df = pd.DataFrame({"rate": irx / 100}).rename_axis("date")
        except Exception as exc2:
            logger.warning("yfinance fallback also failed: %s — no rate filter applied", exc2)
            return pd.Series(dtype=bool)

    # Monthly FEDFUNDS → resample to keep last observation per month, then
    # forward-fill to daily (rate is known at month-end, apply going forward)
    monthly = df["rate"].resample("MS").last().ffill()
    change_bps = (monthly - monthly.shift(lookback_months)) * 100
    hike_months = change_bps > rate_hike_threshold_bps

    # Forward-fill to daily: rate-hike flag applies from the first day of
    # each month until the next data point
    daily_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = hike_months.reindex(daily_idx).ffill().fillna(False)

    n_hike = int(daily.sum())
    logger.info(
        "Rate-hike regime filter: %d days blocked (threshold=%.0f bps, lookback=%d mo)",
        n_hike,
        rate_hike_threshold_bps,
        lookback_months,
    )
    return daily


# ---------------------------------------------------------------------------
# Signal pipeline
# ---------------------------------------------------------------------------


class EarningsDriftPipeline:
    """Compute PEAD signals from earnings surprise data.

    Duck-type compatible with ``SignalPipeline`` — implements
    ``.run(price_data)`` returning a (dates × tickers) DataFrame.

    Parameters
    ----------
    earnings_dir:
        Directory containing per-ticker earnings parquet files written by
        ``EarningsExtractor``. Files are matched by glob ``{TICKER}_*.parquet``.
    lookback_days:
        Maximum days after a report date to carry the signal. Default 30.
    min_surprise_pct:
        Minimum |eps_surprise_pct| to generate a signal. Default 2.0 (2%).
    rate_hike_regime:
        Optional boolean daily Series from ``load_rate_hike_regime()``.
        When True for a date, all signals on that date are zeroed out so
        the strategy sits flat during aggressive Fed tightening cycles.
    """

    def __init__(
        self,
        earnings_dir: Path | str = "data/raw/earnings",
        lookback_days: int = 30,
        min_surprise_pct: float = 5.0,
        rate_hike_regime: pd.Series | None = None,
        spy_drawdown_threshold: float | None = None,
    ) -> None:
        self.earnings_dir = Path(earnings_dir)
        self.lookback_days = lookback_days
        self.min_surprise_pct = min_surprise_pct
        self.rate_hike_regime = rate_hike_regime
        self.spy_drawdown_threshold = spy_drawdown_threshold  # e.g. -0.05 = -5%
        self._cache: dict[str, pd.DataFrame] = {}

    def _load_earnings(self, ticker: str) -> pd.DataFrame:
        """Load and cache earnings parquet(s) for one ticker."""
        if ticker in self._cache:
            return self._cache[ticker]

        files = list(self.earnings_dir.glob(f"{ticker}_*.parquet"))
        if not files:
            logger.debug("No earnings files for %s in %s", ticker, self.earnings_dir)
            self._cache[ticker] = pd.DataFrame()
            return self._cache[ticker]

        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:
                logger.warning("Failed to read earnings parquet %s: %s", f, exc)

        if not frames:
            self._cache[ticker] = pd.DataFrame()
            return self._cache[ticker]

        df = pd.concat(frames, ignore_index=True)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = (
            df.dropna(subset=["report_date", "eps_surprise_pct"])
            .sort_values("report_date")
            .drop_duplicates(subset=["ticker", "report_date"])
            .reset_index(drop=True)
        )
        self._cache[ticker] = df
        return df

    def _ticker_signal(self, ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Compute recency-decayed surprise score for one ticker across dates."""
        earnings = self._load_earnings(ticker)
        result = pd.Series(0.0, index=dates, dtype=float)

        if earnings.empty:
            return result

        td_lookback = pd.Timedelta(days=self.lookback_days)

        for i, date in enumerate(dates):
            ts = pd.Timestamp(date)
            mask = (earnings["report_date"] <= ts) & (
                earnings["report_date"] >= ts - td_lookback
            )
            recent = earnings[mask]
            if recent.empty:
                continue

            latest = recent.iloc[-1]
            surprise_pct = float(latest["eps_surprise_pct"])

            if abs(surprise_pct) < self.min_surprise_pct:
                continue

            days_since = int((ts - latest["report_date"]).days)
            decay = max(0.0, 1.0 - days_since / self.lookback_days)
            result.iloc[i] = surprise_pct * decay

        return result

    def run(self, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute PEAD composite signal for all tickers.

        Parameters
        ----------
        price_data:
            ``{ticker: OHLCV DataFrame}`` — the full price history per ticker.

        Returns
        -------
        pd.DataFrame
            Shape (dates × tickers). Positive values → long signal.
            Zero → no recent reportor below min_surprise_pct threshold.
        """
        columns: dict[str, pd.Series] = {}
        tickers_with_signal = 0

        for ticker, df in price_data.items():
            if df.empty:
                continue
            ticker_dates = pd.DatetimeIndex(sorted(df.index))
            sig = self._ticker_signal(ticker, ticker_dates)
            columns[ticker] = sig
            if (sig != 0).any():
                tickers_with_signal += 1

        if not columns:
            return pd.DataFrame()

        logger.info(
            "EarningsDriftPipeline: %d tickers with at least one non-zero signal",
            tickers_with_signal,
        )
        result = pd.DataFrame(columns)

        # --- SPY trend filter -------------------------------------------
        # Zero out signals on days when SPY's 20-day return is below
        # spy_drawdown_threshold (default -5%).  Captures bear-market
        # periods where positive earnings surprises are overwhelmed by
        # macro selling — without needing any external macro data source.
        if "SPY" in price_data and self.spy_drawdown_threshold is not None:
            spy_close = price_data["SPY"]["close"].astype(float)
            spy_20d_ret = spy_close.pct_change(20)
            bad_spy_dates = set(
                spy_20d_ret[spy_20d_ret < self.spy_drawdown_threshold].index
            )
            spy_blocked = 0
            for date in result.index:
                if date in bad_spy_dates:
                    result.loc[date, :] = 0.0
                    spy_blocked += 1
            if spy_blocked:
                logger.info(
                    "SPY trend filter zeroed signals on %d of %d dates (threshold=%.0f%%)",
                    spy_blocked,
                    len(result),
                    self.spy_drawdown_threshold * 100,
                )

        # --- Rate-hike macro filter -------------------------------------
        # Additional filter using FEDFUNDS 3-month change.
        if self.rate_hike_regime is not None and not self.rate_hike_regime.empty:
            rate_blocked = 0
            for date in result.index:
                ts = pd.Timestamp(date).normalize()
                if self.rate_hike_regime.get(ts, False):
                    result.loc[date, :] = 0.0
                    rate_blocked += 1
            if rate_blocked:
                logger.info(
                    "Rate-hike filter zeroed signals on %d of %d dates (%.1f%%)",
                    rate_blocked,
                    len(result),
                    100 * rate_blocked / len(result),
                )

        return result


# ---------------------------------------------------------------------------
# Metadata signal definition (for strategy memo / StrategyDefinition)
# ---------------------------------------------------------------------------


class _EarningsSurprisePlaceholder(RawIndicator):
    """Placeholder indicator — actual computation lives in EarningsDriftPipeline."""

    @property
    def name(self) -> str:
        return "eps_surprise_decay"

    @property
    def formula(self) -> str:
        return (
            r"\text{PEAD}_{i,t} = \text{eps\_surprise\_pct}_{i,q}"
            r"\times \max\!\left(0,1-\frac{t-t_q}{30}\right)"
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover
        return pd.Series(np.nan, index=df.index)


def _earnings_drift_signal_def() -> SignalDefinition:
    """Return a ``SignalDefinition`` for memo/metadata purposes."""
    sig = SignalDefinition(
        name="pead_surprise_decay",
        family=SignalFamily.MOMENTUM,
        description=(
            "Recency-decayed EPS surprise percentage. "
            "Captures post-earnings announcement drift (PEAD) premium."
        ),
    )
    sig.add_indicator(
        _EarningsSurprisePlaceholder(),
        SignalConfig(
            name="eps_surprise_decay",
            description="EPS surprise % × linear recency decay over 30 days",
            formula=r"\text{PEAD}_{i,t} = \text{surprise\%} \times (1 - t/30)",
            normalization=NormalizationMethod.RAW,
            weight=1.0,
            higher_is_better=True,
        ),
    )
    return sig


# ---------------------------------------------------------------------------
# Strategy definition
# ---------------------------------------------------------------------------


def earnings_drift_strategy(
    earnings_dir: Path | str = "data/raw/earnings",
    lookback_days: int = 30,
    min_surprise_pct: float = 5.0,
    initial_capital: float = 1e7,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> StrategyDefinition:
    """Build the QSG-EARNINGS-DRIFT-001 strategy definition.

    Parameters
    ----------
    earnings_dir:
        Directory of earnings parquets from ``EarningsExtractor``.
    lookback_days:
        Days after a report date to carry the PEAD signal (default 30).
    min_surprise_pct:
        Minimum |EPS surprise %| to trigger a signal (default 2.0).
    initial_capital:
        Starting capital for the backtest.
    start_date / end_date:
        Backtest date range (ISO-8601 strings).
    """
    thesis = StrategyThesis(
        name="QSG-EARNINGS-DRIFT-001",
        classification="Systematic Equity | Long-Only | Event-Driven | Post-Earnings Drift",
        inefficiency=(
            "Markets systematically underreact to earnings surprises. Prices continue "
            "drifting in the direction of the surprise for 5-30 days post-announcement "
            "(Post-Earnings Announcement Drift; Ball & Brown 1968, Chan et al. 1996)."
        ),
        drivers=[
            "Behavioral: Investor anchoring causes slow price adjustment to new earnings",
            "Behavioral: Gradual information diffusion across heterogeneous investor base",
            "Structural: Sell-side analysts revise estimates with a lag after the report",
            "Structural: Institutional constraints prevent immediate full position rebalancing",
        ],
        holding_period="5-15 days (event-driven, short-term)",
        turnover_regime="High — enter T+1 after report, exit by day 15",
        alpha_source="Post-earnings announcement drift (PEAD) premium",
        risk_premium_exposure="Idiosyncratic earnings surprise, limited market beta",
        when_edge_disappears=[
            "HFT fully arbitrages earnings drift within hours of announcement",
            "Crowding — too many funds trading PEAD simultaneously",
            "Persistent BEAR regime suppresses all long-side signals",
            "Earnings dispersion collapses (homogeneous guidance reduces surprise magnitude)",
        ],
        target_aum="$10M–$1B (capacity-constrained by event frequency)",
        status="RESEARCH",
    )

    # Block entries in bear regimes; min threshold is the surprise pct
    entry_rules = institutional_entry_rules(
        signal_threshold=min_surprise_pct,
        blocked_regimes=["BEAR"],
        max_sector_exposure=0.40,
    )

    # PEAD exhausts in 20-30 days — keep hold tight and stop generous
    exit_engine = ExitEngine(
        max_holding_days=15,
        stop_atr_multiple=2.0,
        trailing_atr_multiple=2.5,
        trailing_activation_atr=1.5,
        target_atr_multiple=3.5,
    )

    sizing = InstitutionalSizingConfig(
        method=SizingMethod.VOLATILITY_SCALED,
        total_capital=initial_capital,
        target_position_risk=0.005,  # 50 bps per position
        vol_lookback_days=30,
        max_position_weight=0.05,
        min_position_weight=0.001,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        min_trade_notional=100.0,  # Low minimum for micro-capital accounts
    )

    constraints = institutional_constraints(
        max_position_weight=0.05,
        max_sector_exposure=0.40,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_drawdown=0.20,
    )

    backtest_cfg = BacktestConfig(
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        spread_bps=5.0,         # Wider spread — earnings events have worse execution
        commission_per_share=0.005,
        slippage_bps=5.0,       # Higher slippage around announcements
        rebalance_frequency="daily",
        signal_lag_days=1,      # T+1 execution (next-day open after signal)
    )

    return StrategyDefinition(
        thesis=thesis,
        universe_filter=US_LARGE_CAP_EQUITY,
        signal_definitions=[_earnings_drift_signal_def()],
        entry_rules=entry_rules,
        exit_engine=exit_engine,
        sizing_config=sizing,
        risk_constraints=constraints,
        benchmark_suite=US_EQUITY_BENCHMARKS,
        backtest_config=backtest_cfg,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_earnings_drift_backtest(
    price_data: dict[str, pd.DataFrame],
    earnings_dir: Path | str = "data/raw/earnings",
    lookback_days: int = 30,
    min_surprise_pct: float = 5.0,
    initial_capital: float = 1e7,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    rate_hike_regime: pd.Series | None = None,
):
    """Run the QSG-EARNINGS-DRIFT-001 backtest.

    Parameters
    ----------
    price_data:
        ``{ticker: OHLCV DataFrame}`` indexed by date.
    earnings_dir:
        Directory of earnings parquets (see ``EarningsExtractor``).
    lookback_days:
        Days to carry a PEAD signal after the report date.
    min_surprise_pct:
        Minimum |EPS surprise %| threshold (default 2.0).
    initial_capital:
        Starting portfolio capital.
    start_date / end_date:
        Backtest window.
    rate_hike_regime:
        Optional boolean daily Series from ``load_rate_hike_regime()``.
        Zeroes PEAD signals during Fed tightening cycles.

    Returns
    -------
    HarnessBacktestResult
    """
    from pipeline.strategy.backtest_harness import BacktestHarness

    strategy = earnings_drift_strategy(
        earnings_dir=earnings_dir,
        lookback_days=lookback_days,
        min_surprise_pct=min_surprise_pct,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
    )

    pipeline = EarningsDriftPipeline(
        earnings_dir=earnings_dir,
        lookback_days=lookback_days,
        min_surprise_pct=min_surprise_pct,
        rate_hike_regime=rate_hike_regime,
        spy_drawdown_threshold=None,
    )

    sizer = create_sizer(strategy.sizing_config)

    harness = BacktestHarness(
        signal_pipeline=pipeline,  # type: ignore[arg-type]  # duck-typed
        entry_rules=strategy.entry_rules,
        exit_engine=strategy.exit_engine,
        sizing_model=sizer,
        risk_constraints=strategy.risk_constraints,
        config=strategy.backtest_config,
        benchmark_suite=strategy.benchmark_suite,
    )

    return harness.run(price_data)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def walk_forward_earnings_drift(
    price_data: dict[str, pd.DataFrame],
    earnings_dir: Path | str = "data/raw/earnings",
    lookback_days: int = 30,
    min_surprise_pct: float = 5.0,
    initial_capital: float = 1e6,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    window_months: int = 6,
    embargo_days: int = 15,
    rate_hike_threshold_bps: float | None = None,
) -> list[dict]:
    """Walk-forward validation for QSG-EARNINGS-DRIFT-001.

    Since PEAD uses fixed, academically-motivated parameters (no fitting),
    each window is fully out-of-sample. We simply ask: is the edge
    consistent across independent sub-periods?

    Parameters
    ----------
    price_data:
        Full price history ``{ticker: OHLCV DataFrame}``.
    window_months:
        Length of each independent test window in months (default 6).
    embargo_days:
        Gap between consecutive windows to avoid holding open positions
        that span window boundaries (default 15 = one max hold period).

    Returns
    -------
    list[dict]
        Per-window results with keys: window, start, end, trades,
        sharpe, total_return, hit_rate, max_drawdown.
    """
    from dateutil.relativedelta import relativedelta

    # Load rate-hike regime once for the full period (shared across windows)
    rate_hike_regime: pd.Series | None = None
    if rate_hike_threshold_bps is not None:
        rate_hike_regime = load_rate_hike_regime(
            rate_hike_threshold_bps=rate_hike_threshold_bps
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        w_end = min(cursor + relativedelta(months=window_months), end)
        if w_end > cursor + pd.Timedelta(days=30):  # skip tiny tail windows
            windows.append((cursor, w_end))
        cursor = w_end + pd.Timedelta(days=embargo_days)

    logger.info("Walk-forward: %d windows of %d months", len(windows), window_months)

    results = []
    for i, (w_start, w_end) in enumerate(windows):
        logger.info("Window %d/%d: %s → %s", i + 1, len(windows), w_start.date(), w_end.date())

        # Slice price data to this window
        window_prices: dict[str, pd.DataFrame] = {}
        for ticker, df in price_data.items():
            sliced = df.loc[w_start:w_end]
            if not sliced.empty:
                window_prices[ticker] = sliced

        if not window_prices:
            continue

        result = run_earnings_drift_backtest(
            price_data=window_prices,
            earnings_dir=earnings_dir,
            lookback_days=lookback_days,
            min_surprise_pct=min_surprise_pct,
            initial_capital=initial_capital,
            start_date=str(w_start.date()),
            end_date=str(w_end.date()),
            rate_hike_regime=rate_hike_regime,
        )

        m = result.metrics
        results.append(
            {
                "window": i + 1,
                "start": str(w_start.date()),
                "end": str(w_end.date()),
                "trades": m.total_trades,
                "sharpe": round(m.sharpe_ratio, 3) if not np.isnan(m.sharpe_ratio) else None,
                "total_return_pct": (
                    round(m.total_return * 100, 2) if not np.isnan(m.total_return) else None
                ),
                "hit_rate_pct": (
                    round(m.hit_rate * 100, 1) if not np.isnan(m.hit_rate) else None
                ),
                "max_drawdown_pct": (
                    round(m.max_drawdown * 100, 2) if not np.isnan(m.max_drawdown) else None
                ),
                "profit_factor": (
                    round(m.profit_factor, 3) if not np.isnan(m.profit_factor) else None
                ),
            }
        )

    return results
