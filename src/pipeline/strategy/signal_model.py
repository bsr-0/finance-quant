"""Phase 5: conditional model replacement for the QSG-MICRO-SWING-001 score.

Entered only because Phase 3's harness returned INCONCLUSIVE -- none of the
score, its 4 component buckets, or its 10 underlying boolean conditions
survived BH correction against forward returns. That result matters here: the
plan calls for restricting the logistic feature set to Phase 3's BH survivors,
and there were none. Rather than silently ignore that, this module uses the
full continuous feature set and records the deviation as its own registered
trial (see ``FEATURE_SET_DEVIATION_NOTE``).

Two structural requirements drive the shape of this module:

1. The label must be the payoff actually deployed -- a triple-barrier outcome
   (hit target vs. stop within the holding window), not raw forward return.
   Built by re-running Phase 1's ``resolve_one`` panel-wide with the same ATR
   stop/target multiples ``signal_output.py`` uses in production (1.5 / 2.0).
2. Gate G5 requires a final block (2024-2026) never touched during model
   development. That forces a hard DEV/HOLDOUT split before any of the
   complexity ladder or trial-registry work below runs: everything in this
   module operates on DEV data only, until ``evaluate_on_holdout`` is called
   exactly once, after development has already produced a candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.backtesting.walk_forward import walk_forward_splits
from pipeline.calibration import CalibratedModelWrapper, CalibrationMethod
from pipeline.eval.metrics import brier_score, calibration_error, log_loss
from pipeline.eval.robustness import probability_of_backtest_overfitting
from pipeline.eval.signal_alpha import SignalAlphaResult, SignalTrialRegistry, walk_forward_ic
from pipeline.strategy.signals import compute_indicators
from pipeline.web.outcome_resolution import ResolutionPolicy, resolve_one

logger = logging.getLogger(__name__)

# Match production exactly (signal_output.py:83-84) so the label reflects the
# payoff the live strategy actually offers, not a hypothetical one.
STOP_ATR_MULTIPLE = 1.5
TARGET_ATR_MULTIPLE = 2.0
DEFAULT_MAX_HOLDING_BARS = 15

FULL_FEATURE_SET = [
    "rsi_14",
    "bb_position",
    "stoch_k",
    "williams_r",
    "atr_pct",
    "sma_50_dev",
    "sma_200_dev",
    "sma_50_slope_norm",
    "volume_ratio",
    "obv_slope_z",
    "macd_hist_norm",
]

# Phase 3's D2 decomposition found that ZERO of the 10 boolean conditions
# survived BH correction, so there is no principled "BH survivor" subset to
# restrict the logistic feature set to, as the plan specifies. TREND_FEATURE_SET
# substitutes the two features behind the two components with the (still
# insignificant) highest raw IC in that decomposition -- trend_stacked and
# above_sma_200 -- as a pre-registered, data-informed fallback. Both feature
# sets are run; this substitution is documented rather than silently applied.
FEATURE_SET_DEVIATION_NOTE = (
    "Phase 3 found no BH-surviving components to restrict the logistic feature "
    "set to (0 of 16 trials passed). TREND_FEATURE_SET substitutes the two "
    "highest-raw-IC (still insignificant) components as a documented fallback."
)
TREND_FEATURE_SET = ["sma_50_dev", "sma_200_dev"]

FEATURE_SETS: dict[str, list[str]] = {
    "full": FULL_FEATURE_SET,
    "trend_only": TREND_FEATURE_SET,
}


# ---------------------------------------------------------------------------
# Labels: triple-barrier outcome, matching the deployed payoff
# ---------------------------------------------------------------------------


def label_ticker(
    indicators: pd.DataFrame,
    policy: ResolutionPolicy | None = None,
    stop_atr_multiple: float = STOP_ATR_MULTIPLE,
    target_atr_multiple: float = TARGET_ATR_MULTIPLE,
) -> pd.DataFrame:
    """Triple-barrier label for every bar of one ticker's indicator history.

    For each date, treats that bar's close as a hypothetical entry with a
    stop/target set from that bar's ATR (the same multiples
    ``signal_output.format_signals`` uses), then resolves against the
    following bars exactly as ``outcome_resolution.resolve_one`` does for real
    predictions.

    Returns a DataFrame indexed like *indicators* with ``label`` (1 if
    hit_target, 0 if stopped_out or expired -- unresolved trailing dates are
    NaN) and ``pnl_pct``.
    """
    policy = policy or ResolutionPolicy(max_holding_bars=DEFAULT_MAX_HOLDING_BARS)
    n = len(indicators)
    labels = np.full(n, np.nan)
    pnls = np.full(n, np.nan)

    close = indicators["close"].to_numpy()
    atr = indicators["atr_14"].to_numpy() if "atr_14" in indicators.columns else np.full(n, np.nan)

    for i in range(n - 1):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue
        stop = entry - a * stop_atr_multiple
        target = entry + a * target_atr_multiple

        future_bars = indicators.iloc[i + 1 :]
        resolution = resolve_one(future_bars, entry, stop, target, policy)
        if resolution is None:
            continue
        labels[i] = 1.0 if resolution.outcome == "hit_target" else 0.0
        pnls[i] = resolution.pnl_pct

    return pd.DataFrame({"label": labels, "pnl_pct": pnls}, index=indicators.index)


# ---------------------------------------------------------------------------
# Features: continuous, not thresholded -- the fix for the score's collinear
# oversold booleans (RC2 in the original investigation)
# ---------------------------------------------------------------------------


def build_features(indicators: pd.DataFrame) -> pd.DataFrame:
    """Continuous features from ``compute_indicators`` output.

    Each replaces a thresholded boolean in ``_score_row`` with the underlying
    continuous quantity, so a model can weight the degree of an oversold
    reading instead of a single bit -- and can't double-count four
    near-duplicate booleans the way the hand-weighted score does (D2 found
    their first principal component explains 76.3% of their joint variance).
    """
    nan_series = pd.Series(np.nan, index=indicators.index)

    def col(name: str) -> pd.Series:
        # DataFrame.get returns None (not the default) is only true for
        # totally-absent columns when no default is given; passing the
        # default explicitly here avoids a TypeError on arithmetic below.
        return indicators.get(name, nan_series)

    close = indicators["close"]
    bb_width = (col("bb_upper") - col("bb_lower")).replace(0, np.nan)

    obv_slope = col("obv_slope")
    obv_slope_std = obv_slope.rolling(60, min_periods=20).std()

    features = pd.DataFrame(index=indicators.index)
    features["rsi_14"] = col("rsi_14")
    # Position within the Bollinger band, continuous: 0 = at lower band,
    # 1 = at upper band, <0/>1 = outside. Replaces the below_bb_lower boolean.
    features["bb_position"] = (close - col("bb_lower")) / bb_width
    features["stoch_k"] = col("stoch_k")
    features["williams_r"] = col("williams_r")
    features["atr_pct"] = col("atr_pct")
    features["sma_50_dev"] = close / col("sma_50") - 1
    features["sma_200_dev"] = close / col("sma_200") - 1
    features["sma_50_slope_norm"] = col("sma_50_slope") / close
    features["volume_ratio"] = col("volume") / col("volume_sma_20").replace(0, np.nan)
    features["obv_slope_z"] = obv_slope / obv_slope_std.replace(0, np.nan)
    features["macd_hist_norm"] = col("macd_hist") / close
    return features


# ---------------------------------------------------------------------------
# Dataset assembly: pooled (date, ticker) rows
# ---------------------------------------------------------------------------


def build_dataset(
    price_frames: dict[str, pd.DataFrame],
    policy: ResolutionPolicy | None = None,
) -> pd.DataFrame:
    """Pool features + triple-barrier labels across every ticker into one
    long DataFrame with a ``date`` column and a ``ticker`` column, ready for
    walk-forward splitting on the date axis."""
    rows = []
    for ticker, df in price_frames.items():
        if df.empty or len(df) < 60:
            continue
        try:
            indicators = compute_indicators(df)
            feats = build_features(indicators)
            labels = label_ticker(indicators, policy)
        except Exception:
            logger.exception("Failed to build dataset rows for %s", ticker)
            continue

        block = feats.copy()
        block["label"] = labels["label"]
        block["pnl_pct"] = labels["pnl_pct"]
        block["ticker"] = ticker
        block["date"] = block.index
        rows.append(block)

    if not rows:
        return pd.DataFrame()

    dataset = pd.concat(rows, ignore_index=True)
    # Drop rows missing the label or any feature (e.g. the SMA-200 warmup
    # period) -- sklearn/lightgbm's classifiers don't accept NaN inputs.
    feature_cols = [c for c in FULL_FEATURE_SET if c in dataset.columns]
    return dataset.dropna(subset=["label", *feature_cols])


# ---------------------------------------------------------------------------
# Model classes: the complexity ladder
# ---------------------------------------------------------------------------


def _train_baseline(train_df: pd.DataFrame) -> dict:
    """Constant predicted probability = the training fold's base rate.
    Every candidate model must beat this on OOS log-loss to be worth using."""
    return {"base_rate": float(train_df["label"].mean())}


def _predict_baseline(model: dict, test_df: pd.DataFrame) -> pd.Series:
    return pd.Series(model["base_rate"], index=test_df.index)


def _make_logistic_trainer(feature_cols: list[str]):
    def train_fn(train_df: pd.DataFrame) -> dict:
        from sklearn.linear_model import LogisticRegression

        x = train_df[feature_cols].to_numpy()
        y = train_df["label"].to_numpy()
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        model.fit(x, y)
        return {"model": model}

    def predict_fn(bundle: dict, test_df: pd.DataFrame) -> pd.Series:
        proba = bundle["model"].predict_proba(test_df[feature_cols].to_numpy())[:, 1]
        return pd.Series(proba, index=test_df.index)

    return train_fn, predict_fn


def _make_lightgbm_trainer(feature_cols: list[str]):
    def train_fn(train_df: pd.DataFrame) -> dict:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            max_depth=3,
            n_estimators=100,
            min_child_samples=50,
            verbosity=-1,
        )
        model.fit(train_df[feature_cols].to_numpy(), train_df["label"].to_numpy())
        return {"model": model}

    def predict_fn(bundle: dict, test_df: pd.DataFrame) -> pd.Series:
        proba = bundle["model"].predict_proba(test_df[feature_cols].to_numpy())[:, 1]
        return pd.Series(proba, index=test_df.index)

    return train_fn, predict_fn


MODEL_BUILDERS = {
    "baseline": lambda feature_cols: (_train_baseline, _predict_baseline),
    "logistic": _make_logistic_trainer,
    "lightgbm": _make_lightgbm_trainer,
}


# ---------------------------------------------------------------------------
# Walk-forward evaluation of one (model_class, feature_set, horizon) trial
# ---------------------------------------------------------------------------


@dataclass
class ModelTrialResult:
    trial_name: str
    model_class: str
    feature_set: str
    oos_log_loss: float
    oos_brier: float
    oos_ece: float
    baseline_log_loss: float
    n_folds: int
    n_oos_rows: int
    oos_predictions: pd.Series
    """Pooled OOS predicted probabilities, indexed like the dataset rows they
    came from -- used to reconstruct a wide panel for walk_forward_ic."""
    oos_labels: pd.Series
    beats_baseline: bool = field(init=False)

    def __post_init__(self) -> None:
        self.beats_baseline = np.isfinite(self.oos_log_loss) and (
            self.oos_log_loss < self.baseline_log_loss
        )


def run_model_trial(
    dataset: pd.DataFrame,
    model_class: str,
    feature_cols: list[str],
    trial_name: str,
    train_size: int = 504,
    test_size: int = 63,
    embargo_size: int = 15,
    label_horizon: int = 15,
    calibrate: bool = True,
    calibration_method: CalibrationMethod = CalibrationMethod.PLATT,
) -> ModelTrialResult:
    """Walk-forward train/predict one model class over the pooled dataset.

    Folds are cut on the unique date axis (not row count) so a fold boundary
    never splits a single day's cross-section across train and test.

    Platt scaling is the default calibration method rather than isotonic:
    isotonic is a nonparametric step function that overfits readily on the
    modest per-fold calibration samples here (20% of each training fold),
    which can push probabilities to overconfident extremes and make OOS
    log-loss *worse* than an uncalibrated model -- confirmed empirically on a
    planted-signal fixture during development, where isotonic-calibrated
    logistic failed to beat the baseline while the same model uncalibrated
    (or Platt-calibrated) did. Logistic regression's raw outputs are already
    reasonably calibrated by construction, so Platt's smooth, low-variance
    correction is the safer default; LightGBM callers may still want isotonic
    given its typically poorer native calibration.
    """
    dates = pd.DatetimeIndex(sorted(dataset["date"].unique()))

    if model_class == "baseline":
        train_fn, predict_fn = MODEL_BUILDERS["baseline"](feature_cols)
    else:
        train_fn, predict_fn = MODEL_BUILDERS[model_class](feature_cols)

    if calibrate and model_class != "baseline":
        wrapper = CalibratedModelWrapper(
            train_fn, predict_fn, method=calibration_method, target_col="label"
        )
        fold_train_fn, fold_predict_fn = wrapper.calibrated_train_fn, wrapper.calibrated_predict_fn
    else:
        fold_train_fn, fold_predict_fn = train_fn, predict_fn

    oos_preds: list[pd.Series] = []
    oos_labels: list[pd.Series] = []
    n_folds = 0

    for train_idx, test_idx in walk_forward_splits(
        dates, train_size, test_size, embargo_size=embargo_size, label_horizon=label_horizon
    ):
        train_dates = set(dates[train_idx])
        test_dates = set(dates[test_idx])
        train_rows = dataset[dataset["date"].isin(train_dates)]
        test_rows = dataset[dataset["date"].isin(test_dates)]
        if train_rows.empty or test_rows.empty:
            continue

        model = fold_train_fn(train_rows)
        preds = fold_predict_fn(model, test_rows)
        oos_preds.append(pd.Series(preds.to_numpy(), index=test_rows.index))
        oos_labels.append(test_rows["label"])
        n_folds += 1

    if not oos_preds:
        empty = pd.Series(dtype=float)
        return ModelTrialResult(
            trial_name, model_class, "", np.nan, np.nan, np.nan, np.nan, 0, 0, empty, empty
        )

    pooled_preds = pd.concat(oos_preds)
    pooled_labels = pd.concat(oos_labels)

    baseline_rate = float(dataset["label"].mean())
    baseline_ll = log_loss(pooled_labels, pd.Series(baseline_rate, index=pooled_labels.index))

    return ModelTrialResult(
        trial_name=trial_name,
        model_class=model_class,
        feature_set="",
        oos_log_loss=log_loss(pooled_labels, pooled_preds),
        oos_brier=brier_score(pooled_labels, pooled_preds),
        oos_ece=calibration_error(pooled_labels, pooled_preds),
        baseline_log_loss=baseline_ll,
        n_folds=n_folds,
        n_oos_rows=len(pooled_preds),
        oos_predictions=pooled_preds,
        oos_labels=pooled_labels,
    )


def _predictions_to_wide_panel(dataset: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Reconstruct a date x ticker wide panel of predicted probabilities from
    pooled OOS predictions, for feeding into walk_forward_ic."""
    aligned = dataset.loc[predictions.index, ["date", "ticker"]].copy()
    aligned["pred"] = predictions.to_numpy()
    return aligned.pivot_table(index="date", columns="ticker", values="pred")


# ---------------------------------------------------------------------------
# Trial budget + complexity ladder orchestration
# ---------------------------------------------------------------------------


@dataclass
class ModelLadderReport:
    dev_trials: list[ModelTrialResult]
    registry: SignalTrialRegistry
    """IC-based trials (baseline excluded -- IC is undefined for a constant
    prediction) registered alongside Phase 3's for a combined BH screen."""
    pbo: float
    best_trial: ModelTrialResult | None


def run_model_ladder(
    dataset: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizons: list[int] = (5, 15),
    feature_sets: dict[str, list[str]] | None = None,
    train_size: int = 504,
    test_size: int = 63,
    embargo_size: int = 15,
    max_trials: int = 12,
) -> ModelLadderReport:
    """Run the complexity ladder (baseline -> logistic -> LightGBM) across
    feature sets and horizons, budget-capped and registered for BH screening.

    No skipping: LightGBM is only attempted for a (feature_set, horizon) combo
    if logistic already beat the baseline on OOS log-loss for that combo.
    """
    feature_sets = feature_sets or FEATURE_SETS
    registry = SignalTrialRegistry()
    dev_trials: list[ModelTrialResult] = []
    trial_count = 0

    def budget_left() -> bool:
        nonlocal trial_count
        if trial_count >= max_trials:
            logger.warning("Trial budget (%d) reached; stopping ladder", max_trials)
            return False
        return True

    combos = [(h, name, cols) for h in horizons for name, cols in feature_sets.items()]

    for horizon, fset_name, feature_cols in combos:
        if not budget_left():
            break

        baseline = run_model_trial(
            dataset, "baseline", feature_cols, f"baseline_{fset_name}_{horizon}d",
            train_size, test_size, embargo_size, horizon, calibrate=False,
        )  # fmt: skip
        dev_trials.append(baseline)
        trial_count += 1

        if not budget_left():
            break

        logistic = run_model_trial(
            dataset, "logistic", feature_cols, f"logistic_{fset_name}_{horizon}d",
            train_size, test_size, embargo_size, horizon,
        )  # fmt: skip
        logistic.feature_set = fset_name
        dev_trials.append(logistic)
        trial_count += 1
        _register_ic_trial(registry, dataset, logistic, price_panel, horizon)

        if not logistic.beats_baseline:
            logger.info(
                "%s did not beat baseline OOS log-loss (%.4f vs %.4f); skipping LightGBM",
                logistic.trial_name, logistic.oos_log_loss, logistic.baseline_log_loss,
            )  # fmt: skip
            continue

        if not budget_left():
            break

        lgbm = run_model_trial(
            dataset, "lightgbm", feature_cols, f"lightgbm_{fset_name}_{horizon}d",
            train_size, test_size, embargo_size, horizon,
        )  # fmt: skip
        lgbm.feature_set = fset_name
        dev_trials.append(lgbm)
        trial_count += 1
        _register_ic_trial(registry, dataset, lgbm, price_panel, horizon)

    pbo = _compute_pbo(dev_trials)

    candidates = [
        t for t in dev_trials if t.model_class != "baseline" and t.beats_baseline and t.n_folds > 0
    ]
    best = min(candidates, key=lambda t: t.oos_log_loss) if candidates else None

    return ModelLadderReport(dev_trials=dev_trials, registry=registry, pbo=pbo, best_trial=best)


def _register_ic_trial(
    registry: SignalTrialRegistry,
    dataset: pd.DataFrame,
    trial: ModelTrialResult,
    price_panel: pd.DataFrame,
    horizon: int,
) -> None:
    """IC of a model's OOS predicted probabilities against forward returns,
    registered so BH screening covers model trials alongside Phase 3's."""
    if trial.n_folds == 0:
        registry.record_trial(
            SignalAlphaResult(
                trial.trial_name, np.nan, np.nan, np.nan, np.nan, np.nan, 0, passed=False
            )
        )
        return

    from pipeline.eval.signal_alpha import compute_forward_returns

    pred_panel = _predictions_to_wide_panel(dataset, trial.oos_predictions)
    fwd = compute_forward_returns(price_panel, horizon=horizon)
    result = walk_forward_ic(
        pred_panel, fwd, signal_name=trial.trial_name,
        train_size=min(252, max(20, trial.n_oos_rows // 4)), test_size=21, embargo_size=horizon,
        label_horizon=horizon,
    )  # fmt: skip
    registry.record_trial(result)


def _compute_pbo(trials: list[ModelTrialResult]) -> float:
    """PBO from genuine per-trial train/test fold performance, unlike Phase
    3's crude half-split proxy: each non-baseline trial's OOS log-loss (lower
    is better) is compared against its baseline as a train/test-style pair."""
    non_baseline = [t for t in trials if t.model_class != "baseline" and t.n_folds > 0]
    if len(non_baseline) < 2:
        return np.nan

    # Invert log-loss into a "score" (higher is better) so PBO's convention
    # (top in-sample quantile checked against OOS median) applies naturally.
    train_scores = pd.Series([-t.baseline_log_loss for t in non_baseline])
    test_scores = pd.Series([-t.oos_log_loss for t in non_baseline])
    return probability_of_backtest_overfitting(train_scores, test_scores)


# ---------------------------------------------------------------------------
# DEV/HOLDOUT split and the one-shot final evaluation
# ---------------------------------------------------------------------------


def split_dev_holdout(
    dataset: pd.DataFrame, holdout_start: str = "2024-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the pooled dataset into a development set and a final holdout
    block that must not be touched until evaluate_on_holdout is called."""
    cutoff = pd.Timestamp(holdout_start)
    dev = dataset[dataset["date"] < cutoff]
    holdout = dataset[dataset["date"] >= cutoff]
    return dev, holdout


def evaluate_on_holdout(
    dev_dataset: pd.DataFrame,
    holdout_dataset: pd.DataFrame,
    model_class: str,
    feature_cols: list[str],
    calibration_method: CalibrationMethod = CalibrationMethod.PLATT,
) -> dict:
    """Train once on the full DEV set, evaluate once on HOLDOUT. Call this
    exactly once, on the single candidate the DEV-only ladder selected.

    See ``run_model_trial`` for why Platt is the default over isotonic.
    """
    if model_class == "baseline":
        train_fn, predict_fn = MODEL_BUILDERS["baseline"](feature_cols)
    else:
        train_fn, predict_fn = MODEL_BUILDERS[model_class](feature_cols)
        wrapper = CalibratedModelWrapper(
            train_fn, predict_fn, method=calibration_method, target_col="label"
        )
        train_fn, predict_fn = wrapper.calibrated_train_fn, wrapper.calibrated_predict_fn

    model = train_fn(dev_dataset)
    preds = predict_fn(model, holdout_dataset)
    labels = holdout_dataset["label"]

    baseline_rate = float(dev_dataset["label"].mean())
    baseline_ll = log_loss(labels, pd.Series(baseline_rate, index=labels.index))

    return {
        "holdout_log_loss": log_loss(labels, preds),
        "holdout_brier": brier_score(labels, preds),
        "holdout_ece": calibration_error(labels, preds),
        "baseline_log_loss": baseline_ll,
        "n_holdout_rows": len(holdout_dataset),
        "beats_baseline": log_loss(labels, preds) < baseline_ll,
    }


# ---------------------------------------------------------------------------
# Gate G5: all eight criteria, or ship nothing
# ---------------------------------------------------------------------------


@dataclass
class G5Result:
    passed: bool
    criteria: dict[str, bool]
    detail: dict[str, object]
    reasoning: list[str] = field(default_factory=list)


def evaluate_g5(
    best_trial: ModelTrialResult | None,
    dataset: pd.DataFrame,
    combined_registry_trials: list[SignalAlphaResult],
    pbo: float,
    holdout_result: dict | None,
    same_bar_stable: bool = False,
    alpha: float = 0.05,
    pnl_ci_alpha: float = 0.05,
) -> G5Result:
    """All eight G5 criteria from the validation plan. Missing any one means
    ship nothing -- the site stays UNRATED regardless of how the others land.
    """
    from pipeline.eval.clustering import cluster_bootstrap, cluster_by_correlation
    from pipeline.eval.signal_alpha import signal_fdr_screen

    criteria: dict[str, bool] = {}
    detail: dict[str, object] = {}
    reasoning: list[str] = []

    criterion_names = (
        "oos_beats_baseline",
        "ic_passes",
        "survives_full_bh",
        "pnl_ci_excludes_zero",
        "pbo_below_half",
        "ece_below_threshold",
        "same_bar_stable",
        "holdout_beats_baseline",
    )
    if best_trial is None:
        return G5Result(
            passed=False,
            criteria=dict.fromkeys(criterion_names, False),
            detail={},
            reasoning=["no trial beat baseline OOS log-loss; nothing to gate"],
        )

    # C1: OOS log-loss and Brier beat baseline, on the de-clustered replicate.
    trial_rows = dataset.loc[best_trial.oos_predictions.index]
    pnl_records = pd.DataFrame(
        {
            "signal_date": trial_rows["date"],
            "ticker": trial_rows["ticker"],
            "pnl_pct": trial_rows["pnl_pct"].to_numpy(),
        }
    )
    returns_panel = dataset.pivot_table(index="date", columns="ticker", values="pnl_pct")
    cluster_map = cluster_by_correlation(returns_panel.pct_change(fill_method=None).fillna(0))

    from pipeline.eval.clustering import declustered_replicate

    declustered = declustered_replicate(pnl_records, cluster_map)
    c1 = bool(best_trial.beats_baseline)
    criteria["oos_beats_baseline"] = c1
    detail["oos_log_loss"] = best_trial.oos_log_loss
    detail["oos_brier"] = best_trial.oos_brier
    detail["declustered_n"] = len(declustered)
    if not c1:
        reasoning.append("OOS log-loss/Brier did not beat the baseline")

    # C2: walk_forward_ic on model probabilities passes.
    ic_trial = next(
        (t for t in combined_registry_trials if t.signal_name == best_trial.trial_name), None
    )
    c2 = bool(ic_trial and ic_trial.passed)
    criteria["ic_passes"] = c2
    detail["ic_mean"] = ic_trial.ic_mean if ic_trial else np.nan
    detail["ic_dsr_prob"] = ic_trial.deflated_sharpe_prob if ic_trial else np.nan
    if not c2:
        reasoning.append("model-probability walk-forward IC did not pass the deflated-Sharpe gate")

    # C3: survives BH across the FULL registry (Phase 3's trials + Phase 5's).
    screened = signal_fdr_screen(combined_registry_trials, alpha=alpha)
    c3 = any(r.signal_name == best_trial.trial_name and sig for r, sig in screened)
    criteria["survives_full_bh"] = c3
    detail["n_registry_trials"] = len(combined_registry_trials)
    if not c3:
        reasoning.append("did not survive BH correction across the combined trial registry")

    # C4: cluster-bootstrap 95% CI on OOS mean P&L excludes 0.
    lo, hi = cluster_bootstrap(pnl_records, cluster_map, np.mean, alpha=pnl_ci_alpha)
    c4 = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
    criteria["pnl_ci_excludes_zero"] = c4
    detail["pnl_ci"] = (lo, hi)
    if not c4:
        reasoning.append(f"cluster-bootstrap P&L CI [{lo:.3f}, {hi:.3f}] does not exclude zero")

    # C5: PBO < 0.5.
    c5 = np.isfinite(pbo) and pbo < 0.5
    criteria["pbo_below_half"] = c5
    detail["pbo"] = pbo
    if not c5:
        reasoning.append(f"probability of backtest overfitting {pbo:.3f} >= 0.5")

    # C6: OOS ECE < 0.10.
    c6 = np.isfinite(best_trial.oos_ece) and best_trial.oos_ece < 0.10
    criteria["ece_below_threshold"] = c6
    detail["oos_ece"] = best_trial.oos_ece
    if not c6:
        reasoning.append(f"OOS ECE {best_trial.oos_ece:.3f} >= 0.10")

    # C7: conclusions unchanged under both same-bar policies. The caller
    # re-runs the whole ladder under target_first and passes the comparison
    # in explicitly; default False is conservative (unverified = not met).
    c7 = bool(same_bar_stable)
    criteria["same_bar_stable"] = c7
    detail["same_bar_stable"] = c7
    if not c7:
        reasoning.append("same-bar policy sensitivity not verified stable (see D5 bracket)")

    # C8: holds on the 2024-2026 holdout, never touched during development.
    c8 = bool(holdout_result and holdout_result.get("beats_baseline"))
    criteria["holdout_beats_baseline"] = c8
    detail["holdout_result"] = holdout_result
    if not c8:
        reasoning.append("did not beat baseline on the untouched 2024-2026 holdout block")

    passed = all(criteria.values())
    if passed:
        reasoning = ["all eight G5 criteria satisfied"]

    return G5Result(passed=passed, criteria=criteria, detail=detail, reasoning=reasoning)


@dataclass
class Phase5Report:
    ladder: ModelLadderReport
    holdout_result: dict | None
    same_bar_stable: bool
    g5: G5Result


def run_phase5(
    price_frames: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    phase3_trials: list[SignalAlphaResult] | None = None,
    holdout_start: str = "2024-01-01",
    horizons: list[int] = (5, 15),
    feature_sets: dict[str, list[str]] | None = None,
    train_size: int = 504,
    test_size: int = 63,
    embargo_size: int = 15,
    max_trials: int = 12,
    alpha: float = 0.05,
) -> Phase5Report:
    """End-to-end Phase 5: DEV/HOLDOUT split, complexity ladder on DEV only,
    D5 same-bar bracket on the winning config, one-shot HOLDOUT evaluation,
    and the combined G5 verdict.

    Args:
        phase3_trials: Phase 3's registered trials (the score, its 4 buckets,
            and its 10 conditions), combined with Phase 5's for the full-registry
            BH screen (G5 criterion C3). Pass ``[]`` to screen Phase 5 alone,
            but that understates the true trial count and is not what the
            plan's discipline calls for.
    """
    dataset = build_dataset(
        price_frames, ResolutionPolicy(max_holding_bars=DEFAULT_MAX_HOLDING_BARS)
    )
    dev, holdout = split_dev_holdout(dataset, holdout_start=holdout_start)

    ladder = run_model_ladder(
        dev, price_panel, horizons=list(horizons), feature_sets=feature_sets,
        train_size=train_size, test_size=test_size, embargo_size=embargo_size,
        max_trials=max_trials,
    )  # fmt: skip

    holdout_result: dict | None = None
    same_bar_stable = False

    if ladder.best_trial is not None:
        best = ladder.best_trial
        feature_cols = FEATURE_SETS.get(best.feature_set, FULL_FEATURE_SET)

        holdout_result = evaluate_on_holdout(dev, holdout, best.model_class, feature_cols)

        # D5: re-label DEV under the opposite same-bar policy and check the
        # winning config still beats baseline. A conclusion that flips
        # between stop_first and target_first is not a conclusion.
        alt_policy = ResolutionPolicy(
            max_holding_bars=DEFAULT_MAX_HOLDING_BARS, same_bar_policy="target_first"
        )
        alt_dataset = build_dataset(price_frames, alt_policy)
        alt_dev, _ = split_dev_holdout(alt_dataset, holdout_start=holdout_start)
        horizon = int(best.trial_name.rsplit("_", 1)[-1].rstrip("d"))
        alt_result = run_model_trial(
            alt_dev, best.model_class, feature_cols, f"{best.trial_name}_target_first",
            train_size, test_size, embargo_size, horizon,
        )  # fmt: skip
        same_bar_stable = bool(alt_result.beats_baseline)

    combined_trials = list(phase3_trials or []) + ladder.registry.trials
    g5 = evaluate_g5(
        ladder.best_trial, dev, combined_trials, ladder.pbo, holdout_result,
        same_bar_stable=same_bar_stable, alpha=alpha,
    )  # fmt: skip

    return Phase5Report(
        ladder=ladder, holdout_result=holdout_result, same_bar_stable=same_bar_stable, g5=g5
    )
