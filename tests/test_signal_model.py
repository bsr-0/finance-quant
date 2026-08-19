"""Tests for pipeline.strategy.signal_model (Phase 5: conditional model
replacement).

Entered only because Phase 3's harness returned INCONCLUSIVE. The critical
property to test here is the same one that mattered for the Phase 3 harness:
the complexity ladder and G5 gate must actually discriminate a real
relationship from noise, not just always agree to move forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.strategy.signal_model import (
    FULL_FEATURE_SET,
    ModelTrialResult,
    build_dataset,
    build_features,
    evaluate_g5,
    evaluate_on_holdout,
    label_ticker,
    run_model_ladder,
    run_model_trial,
    split_dev_holdout,
)
from pipeline.strategy.signals import compute_indicators
from pipeline.web.outcome_resolution import ResolutionPolicy


def make_bars(
    rows: list[tuple[float, float, float, float]], start: str = "2020-01-06"
) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(
        [{"open": o, "high": h, "low": lo, "close": c, "volume": 500_000} for o, h, lo, c in rows],
        index=idx,
    )
    return df


def flat_bar(price: float = 100.0) -> tuple[float, float, float, float]:
    return (price, price + 0.3, price - 0.3, price)


# --- label_ticker -----------------------------------------------------------


def test_label_ticker_marks_target_hit_as_one():
    # 60 warmup bars so ATR is defined, then a bar that hits the target hard
    # without touching the stop (low held above entry), to avoid the
    # same-bar stop_first tiebreak resolving it as a loss instead.
    bars = [flat_bar(100 + 0.01 * i) for i in range(60)] + [(100.0, 110.0, 100.0, 109.0)]
    df = make_bars(bars)
    indicators = compute_indicators(df)

    labels = label_ticker(indicators, ResolutionPolicy(cost_bps=0.0))
    # The warmup date just before the big bar should be labeled a win.
    entry_date = indicators.index[-2]
    assert labels.loc[entry_date, "label"] == 1.0
    assert labels.loc[entry_date, "pnl_pct"] > 0


def test_label_ticker_marks_stop_hit_as_zero():
    bars = [flat_bar(100 + 0.01 * i) for i in range(60)] + [(100.0, 100.5, 90.0, 91.0)]
    df = make_bars(bars)
    indicators = compute_indicators(df)

    labels = label_ticker(indicators, ResolutionPolicy(cost_bps=0.0))
    entry_date = indicators.index[-2]
    assert labels.loc[entry_date, "label"] == 0.0
    assert labels.loc[entry_date, "pnl_pct"] < 0


def test_label_ticker_leaves_unresolvable_trailing_dates_as_nan():
    bars = [flat_bar(100 + 0.01 * i) for i in range(60)]
    df = make_bars(bars)
    indicators = compute_indicators(df)

    labels = label_ticker(indicators, ResolutionPolicy(cost_bps=0.0, max_holding_bars=15))
    # The last ~15 bars can't have a full holding window ahead of them.
    assert labels["label"].iloc[-1:].isna().all()


def test_label_ticker_skips_rows_with_zero_or_missing_atr():
    bars = [flat_bar() for _ in range(5)]  # too few bars for ATR to be defined
    df = make_bars(bars)
    indicators = compute_indicators(df)

    labels = label_ticker(indicators, ResolutionPolicy(cost_bps=0.0))
    assert labels["label"].isna().all()


# --- build_features -----------------------------------------------------------


def test_build_features_produces_all_expected_columns():
    bars = [flat_bar(100 + 0.02 * i) for i in range(120)]
    df = make_bars(bars)
    indicators = compute_indicators(df)

    feats = build_features(indicators)
    assert list(feats.columns) == FULL_FEATURE_SET
    assert len(feats) == len(indicators)


def test_bb_position_is_zero_at_lower_band_and_one_at_upper_band():
    idx = pd.bdate_range("2020-01-01", periods=3)
    indicators = pd.DataFrame(
        {
            "close": [100.0, 100.0, 90.0],
            "bb_lower": [90.0, 90.0, 90.0],
            "bb_upper": [110.0, 100.0, 110.0],
        },
        index=idx,
    )
    feats = build_features(indicators)
    assert feats["bb_position"].iloc[0] == pytest.approx(0.5)  # midway
    assert feats["bb_position"].iloc[1] == pytest.approx(1.0)  # at upper band
    assert feats["bb_position"].iloc[2] == pytest.approx(0.0)  # at lower band


# --- build_dataset -----------------------------------------------------------


def _noisy_ohlcv(n: int, seed: int, start_price: float = 100.0) -> pd.DataFrame:
    """A price series with genuine up/down variation, unlike ``flat_bar``'s
    deterministic ramp -- RSI and OBV-slope-based features are undefined
    (NaN) on a series with no down days or zero-variance slope, which a
    monotonic synthetic ramp produces by construction."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-06", periods=n)
    close = pd.Series(start_price * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n))), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(300_000, 700_000, n).astype(float),
        }
    )


def test_build_dataset_pools_multiple_tickers():
    # 250+ bars so sma_200 (a required feature) is defined for some rows.
    frames = {"A": _noisy_ohlcv(260, seed=1), "B": _noisy_ohlcv(260, seed=2, start_price=50.0)}
    dataset = build_dataset(frames, ResolutionPolicy(cost_bps=0.0))

    assert set(dataset["ticker"].unique()) == {"A", "B"}
    assert dataset["label"].isin([0.0, 1.0]).all()
    assert "date" in dataset.columns
    assert dataset[FULL_FEATURE_SET].notna().all().all()


def test_build_dataset_drops_unresolvable_rows():
    frames = {"A": _noisy_ohlcv(260, seed=3)}
    dataset = build_dataset(frames, ResolutionPolicy(cost_bps=0.0))
    assert dataset["label"].notna().all()


def test_build_dataset_drops_rows_with_missing_features():
    """Bars before sma_200 warms up (< 200 bars) must not reach the model."""
    frames = {"A": _noisy_ohlcv(260, seed=4)}
    dataset = build_dataset(frames, ResolutionPolicy(cost_bps=0.0))
    assert len(dataset) <= 260 - 200


def test_build_dataset_survives_a_bad_ticker():
    frames = {"GOOD": _noisy_ohlcv(260, seed=5), "BAD": pd.DataFrame({"close": [1.0]})}
    dataset = build_dataset(frames, ResolutionPolicy(cost_bps=0.0))
    assert "GOOD" in dataset["ticker"].unique()
    assert "BAD" not in dataset["ticker"].unique()


# --- synthetic pooled datasets for model-ladder tests ------------------------


def _planted_dataset(n_dates: int, n_tickers: int, seed: int) -> pd.DataFrame:
    """A pooled dataset where label truly depends on rsi_14 -- a real,
    learnable relationship for the model ladder's positive control."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for t in range(n_tickers):
            rsi = rng.uniform(0, 100)
            prob = 0.2 + 0.5 * (rsi < 35)  # oversold genuinely predicts a win
            label = float(rng.random() < prob)
            row = {c: rng.normal(0, 1) for c in FULL_FEATURE_SET}
            row["rsi_14"] = rsi
            row["label"] = label
            row["pnl_pct"] = 3.0 if label else -3.0
            row["date"] = d
            row["ticker"] = f"T{t}"
            rows.append(row)
    return pd.DataFrame(rows)


def _null_dataset(n_dates: int, n_tickers: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for t in range(n_tickers):
            label = float(rng.random() < 0.4)
            row = {c: rng.normal(0, 1) for c in FULL_FEATURE_SET}
            row["label"] = label
            row["pnl_pct"] = 3.0 if label else -3.0
            row["date"] = d
            row["ticker"] = f"T{t}"
            rows.append(row)
    return pd.DataFrame(rows)


# --- run_model_trial ----------------------------------------------------------


def test_baseline_trial_predicts_the_training_base_rate():
    dataset = _null_dataset(n_dates=400, n_tickers=5, seed=0)
    result = run_model_trial(
        dataset, "baseline", FULL_FEATURE_SET, "baseline_test",
        train_size=150, test_size=40, embargo_size=5, label_horizon=5,
    )  # fmt: skip
    assert result.n_folds > 0
    assert not result.beats_baseline  # baseline can't beat itself


def test_logistic_beats_baseline_on_a_planted_relationship():
    dataset = _planted_dataset(n_dates=400, n_tickers=6, seed=1)
    result = run_model_trial(
        dataset, "logistic", FULL_FEATURE_SET, "logistic_test",
        train_size=150, test_size=40, embargo_size=5, label_horizon=5,
    )  # fmt: skip
    assert result.n_folds > 0
    assert result.beats_baseline


def test_logistic_does_not_reliably_beat_baseline_on_pure_noise():
    beats = 0
    n_seeds = 5
    for seed in range(n_seeds):
        dataset = _null_dataset(n_dates=300, n_tickers=5, seed=seed)
        result = run_model_trial(
            dataset, "logistic", FULL_FEATURE_SET, "logistic_null",
            train_size=120, test_size=30, embargo_size=5, label_horizon=5,
        )  # fmt: skip
        beats += result.beats_baseline
    assert beats <= 2, f"logistic beat baseline on {beats}/{n_seeds} pure-noise datasets"


# --- run_model_ladder: complexity ladder gating ------------------------------


def test_ladder_skips_lightgbm_when_logistic_does_not_beat_baseline():
    dataset = _null_dataset(n_dates=300, n_tickers=5, seed=2)
    price_panel = _price_panel_for(dataset)

    report = run_model_ladder(
        dataset, price_panel, horizons=[5], feature_sets={"full": FULL_FEATURE_SET},
        train_size=120, test_size=30, embargo_size=5, max_trials=12,
    )  # fmt: skip
    model_classes = {t.model_class for t in report.dev_trials}
    logistic = next(t for t in report.dev_trials if t.model_class == "logistic")
    if not logistic.beats_baseline:
        assert "lightgbm" not in model_classes


def test_ladder_respects_trial_budget():
    dataset = _planted_dataset(n_dates=300, n_tickers=5, seed=3)
    price_panel = _price_panel_for(dataset)

    feature_sets = {"full": FULL_FEATURE_SET, "trend_only": ["rsi_14"]}
    report = run_model_ladder(
        dataset,
        price_panel,
        horizons=[5, 10],
        feature_sets=feature_sets,
        train_size=100,
        test_size=25,
        embargo_size=5,
        max_trials=4,
    )
    assert len(report.dev_trials) <= 4


def _price_panel_for(dataset: pd.DataFrame) -> pd.DataFrame:
    """A synthetic close-price panel consistent with the dataset's dates and
    tickers, for the IC-registration step inside run_model_ladder."""
    rng = np.random.default_rng(7)
    dates = pd.DatetimeIndex(sorted(dataset["date"].unique()))
    tickers = sorted(dataset["ticker"].unique())
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(tickers))), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# --- split_dev_holdout / evaluate_on_holdout ---------------------------------


def test_split_dev_holdout_partitions_by_date():
    dataset = _null_dataset(n_dates=100, n_tickers=2, seed=4)
    mid_date = sorted(dataset["date"].unique())[50]

    dev, holdout = split_dev_holdout(dataset, holdout_start=str(mid_date.date()))
    assert (dev["date"] < mid_date).all()
    assert (holdout["date"] >= mid_date).all()
    assert len(dev) + len(holdout) == len(dataset)


def test_evaluate_on_holdout_reports_beats_baseline():
    dataset = _planted_dataset(n_dates=300, n_tickers=5, seed=5)
    dates = sorted(dataset["date"].unique())
    cutoff = dates[200]
    dev, holdout = split_dev_holdout(dataset, holdout_start=str(cutoff.date()))

    result = evaluate_on_holdout(dev, holdout, "logistic", FULL_FEATURE_SET)
    assert "beats_baseline" in result
    assert result["n_holdout_rows"] == len(holdout)


# --- evaluate_g5 --------------------------------------------------------------


def _fake_trial(name: str, beats: bool, ece: float, index: pd.Index) -> ModelTrialResult:
    preds = pd.Series(np.full(len(index), 0.5), index=index)
    labels = pd.Series(np.zeros(len(index)), index=index)
    t = ModelTrialResult(
        trial_name=name, model_class="logistic", feature_set="full",
        oos_log_loss=0.5 if beats else 0.9, oos_brier=0.2, oos_ece=ece,
        baseline_log_loss=0.7, n_folds=3, n_oos_rows=len(index),
        oos_predictions=preds, oos_labels=labels,
    )  # fmt: skip
    return t


def test_evaluate_g5_fails_when_no_trial_beats_baseline():
    result = evaluate_g5(
        best_trial=None, dataset=pd.DataFrame(), combined_registry_trials=[], pbo=0.3,
        holdout_result=None,
    )  # fmt: skip
    assert result.passed is False
    assert not any(result.criteria.values())


def test_evaluate_g5_requires_all_criteria_to_pass():
    """Even if some criteria pass, ANY single failure must fail the gate --
    this is the property that keeps G5 from shipping on a partial case."""
    idx = pd.RangeIndex(20)
    dataset = pd.DataFrame(
        {
            "date": [f"2020-01-{(i % 20) + 1:02d}" for i in range(20)],
            "ticker": [f"T{i % 4}" for i in range(20)],
            "pnl_pct": np.random.default_rng(0).normal(0, 1, 20),
        },
        index=idx,
    )
    trial = _fake_trial("logistic_full_5d", beats=True, ece=0.05, index=idx)

    # No IC trial registered matching the name, no holdout result -> several
    # criteria should fail even though beats_baseline (C1) is True.
    result = evaluate_g5(
        best_trial=trial, dataset=dataset, combined_registry_trials=[], pbo=0.3,
        holdout_result=None,
    )  # fmt: skip
    assert result.passed is False
    assert result.criteria["oos_beats_baseline"] is True
    assert result.criteria["ic_passes"] is False
    assert result.criteria["holdout_beats_baseline"] is False


def test_evaluate_g5_high_pbo_fails_the_gate():
    idx = pd.RangeIndex(20)
    dataset = pd.DataFrame(
        {
            "date": [f"2020-01-{(i % 20) + 1:02d}" for i in range(20)],
            "ticker": [f"T{i % 4}" for i in range(20)],
            "pnl_pct": np.random.default_rng(0).normal(0, 1, 20),
        },
        index=idx,
    )
    trial = _fake_trial("logistic_full_5d", beats=True, ece=0.05, index=idx)

    result = evaluate_g5(
        best_trial=trial, dataset=dataset, combined_registry_trials=[], pbo=0.9,
        holdout_result={"beats_baseline": True}, same_bar_stable=True,
    )  # fmt: skip
    assert result.criteria["pbo_below_half"] is False
    assert result.passed is False
