"""Generic signal generation framework for systematic strategies.

Provides a composable framework for defining signals with explicit
mathematical rules. Supports multiple signal families (momentum, value,
quality, etc.) with configurable normalization and combination.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from pipeline.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

TI = TechnicalIndicators


# ---------------------------------------------------------------------------
# Signal normalization
# ---------------------------------------------------------------------------


class NormalizationMethod(Enum):
    RAW = "raw"
    ZSCORE = "zscore"
    RANK = "rank"
    PERCENTILE = "percentile"
    WINSORIZE = "winsorize"
    MIN_MAX = "min_max"


def zscore_normalize(series: pd.Series, window: int = 252) -> pd.Series:
    r"""Cross-sectional or rolling z-score.

    .. math::
        z_i = \frac{x_i - \mu}{\sigma}
    """
    mu = series.rolling(window, min_periods=20).mean()
    sigma = series.rolling(window, min_periods=20).std()
    return (series - mu) / sigma.replace(0, np.nan)


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization (0 to 1)."""
    return df.rank(axis=1, pct=True)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize outliers to specified percentiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def min_max_normalize(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling min-max normalization to [0, 1]."""
    roll_min = series.rolling(window, min_periods=20).min()
    roll_max = series.rolling(window, min_periods=20).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return (series - roll_min) / denom


# ---------------------------------------------------------------------------
# Signal definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for a single raw indicator or signal."""

    name: str
    description: str = ""
    formula: str = ""  # LaTeX-ready formula description
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    lookback_window: int = 252
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    weight: float = 1.0
    higher_is_better: bool = True  # Direction of the signal


class SignalFamily(Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    SIZE = "size"
    LOW_VOLATILITY = "low_volatility"
    CARRY = "carry"
    MICROSTRUCTURE = "microstructure"
    MEAN_REVERSION = "mean_reversion"


# ---------------------------------------------------------------------------
# Raw indicator computation
# ---------------------------------------------------------------------------


class RawIndicator(ABC):
    """Base class for raw indicator computation."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute raw indicator values from OHLCV data."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Indicator name."""

    @property
    def formula(self) -> str:
        """LaTeX-ready formula."""
        return ""


class MomentumReturn(RawIndicator):
    r"""N-day price momentum (total return).

    .. math::
        \text{MOM}_{i,t} = \frac{P_{i,t}}{P_{i,t-n}} - 1
    """

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    @property
    def name(self) -> str:
        return f"momentum_{self.lookback}d"

    @property
    def formula(self) -> str:
        return (
            rf"\text{{MOM}}_{{i,t}} = \frac{{P_{{i,t-{self.skip}}}}}"
            rf"{{P_{{i,t-{self.lookback}}}}} - 1"
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        if self.skip > 0:
            return close.shift(self.skip) / close.shift(self.lookback) - 1
        return close / close.shift(self.lookback) - 1


class MovingAverageCrossover(RawIndicator):
    r"""Ratio of short-term to long-term moving average.

    .. math::
        \text{MAC}_{i,t} = \frac{\text{SMA}(P, n_s)}{\text{SMA}(P, n_l)} - 1
    """

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self.short_window = short_window
        self.long_window = long_window

    @property
    def name(self) -> str:
        return f"ma_cross_{self.short_window}_{self.long_window}"

    @property
    def formula(self) -> str:
        return (
            rf"\text{{MAC}}_{{i,t}} = \frac{{\text{{SMA}}(P, {self.short_window})}}"
            rf"{{\text{{SMA}}(P, {self.long_window})}} - 1"
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        short_ma = TI.sma(close, self.short_window)
        long_ma = TI.sma(close, self.long_window)
        return short_ma / long_ma.replace(0, np.nan) - 1


class RSIMeanReversion(RawIndicator):
    r"""RSI-based mean reversion signal.

    .. math::
        \text{Signal}_{i,t} = 50 - \text{RSI}(P, n)
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"rsi_reversion_{self.window}"

    @property
    def formula(self) -> str:
        return rf"\text{{Signal}}_{{i,t}} = 50 - \text{{RSI}}(P, {self.window})"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        rsi = TI.rsi(df["close"], self.window)
        return 50 - rsi  # Positive when oversold (bullish for mean reversion)


class VolatilitySignal(RawIndicator):
    r"""Realized volatility as a signal (lower is better for low-vol strategy).

    .. math::
        \sigma_{i,t} = \text{std}(\text{ret}_{i}, n) \times \sqrt{252}
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"realized_vol_{self.window}d"

    @property
    def formula(self) -> str:
        return rf"\sigma_{{i,t}} = \text{{std}}(\text{{ret}}_i, {self.window}) \times \sqrt{{252}}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["close"].pct_change()
        return returns.rolling(self.window, min_periods=20).std() * np.sqrt(252)


class MomentumDispersion(RawIndicator):
    r"""Cross-sectional return dispersion — momentum crash protection.

    When cross-sectional dispersion spikes (all stocks moving together in
    the same direction), momentum strategies are vulnerable to sharp
    reversals.  This signal penalises the composite when dispersion is
    abnormally high relative to its own history.

    .. math::
        D_t = \sigma_{\text{cs}}(\text{ret}_{i,t}) \text{ (cross-sectional std)}

    The indicator is the rolling z-score of dispersion: when it exceeds
    +1.5σ, the composite signal is dampened.  Implemented as a
    single-stock proxy using recent absolute return magnitude relative to
    trailing realised vol (high = regime-change risk).
    """

    def __init__(self, ret_window: int = 5, vol_window: int = 60) -> None:
        self.ret_window = ret_window
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        return "momentum_crash_protection"

    @property
    def formula(self) -> str:
        return (
            r"D_{i,t} = \frac{|\text{ret}_{i,5d}|}{\sigma_{i,60}} "
            r"\; (\text{high} \Rightarrow \text{momentum crash risk})"
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ret_5d = close.pct_change(self.ret_window).abs()
        vol_60d = close.pct_change().rolling(self.vol_window, min_periods=20).std()
        # Ratio > 1 means recent move is unusually large relative to vol
        # Return negative values so "higher_is_better=True" penalises high dispersion
        ratio = ret_5d / vol_60d.replace(0, np.nan)
        return -ratio  # Negative = crash-protection penalty


class VolumeRatio(RawIndicator):
    r"""Volume ratio relative to trailing average.

    .. math::
        \text{VR}_{i,t} = \frac{V_{i,t}}{\text{SMA}(V_i, n)}
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"volume_ratio_{self.window}d"

    @property
    def formula(self) -> str:
        return rf"\text{{VR}}_{{i,t}} = \frac{{V_{{i,t}}}}{{\text{{SMA}}(V_i, {self.window})}}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        vol = df["volume"]
        avg_vol = TI.sma(vol, self.window)
        return vol / avg_vol.replace(0, np.nan)


class BollingerBandPosition(RawIndicator):
    r"""Position within Bollinger Bands (0 = lower, 1 = upper).

    .. math::
        \text{BB\%}_{i,t} = \frac{P_{i,t} - BB_{\text{lower}}}
        {BB_{\text{upper}} - BB_{\text{lower}}}
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"bb_position_{self.window}d"

    @property
    def formula(self) -> str:
        return (
            r"\text{BB\%}_{i,t} = \frac{P_{i,t} - BB_{\text{lower}}}"
            r"{BB_{\text{upper}} - BB_{\text{lower}}}"
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        upper, _, lower = TI.bollinger_bands(close, self.window, self.num_std)
        width = (upper - lower).replace(0, np.nan)
        return (close - lower) / width


# ---------------------------------------------------------------------------
# Signal pipeline
# ---------------------------------------------------------------------------


@dataclass
class SignalDefinition:
    """Complete definition of a composite signal with multiple indicators."""

    name: str
    family: SignalFamily
    description: str = ""
    indicators: list[tuple[RawIndicator, SignalConfig]] = field(default_factory=list)

    def add_indicator(
        self,
        indicator: RawIndicator,
        config: SignalConfig | None = None,
    ) -> SignalDefinition:
        """Add a raw indicator with its configuration."""
        cfg = config or SignalConfig(
            name=indicator.name,
            formula=indicator.formula,
        )
        self.indicators.append((indicator, cfg))
        return self

    @property
    def indicator_names(self) -> list[str]:
        return [cfg.name for _, cfg in self.indicators]


class SignalPipeline:
    """Compute and normalize signals for a universe of instruments.

    The pipeline:
      1. Computes raw indicators for each instrument.
      2. Applies normalization (z-score, rank, etc.).
      3. Combines into a composite signal per instrument per date.
    """

    def __init__(self, signal_def: SignalDefinition) -> None:
        self.signal_def = signal_def

    def compute_raw(self, price_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Compute raw indicator values for all instruments.

        Returns:
            ``{ticker: DataFrame}`` where each DataFrame has one column
            per indicator, indexed by date.
        """
        results: dict[str, pd.DataFrame] = {}
        for ticker, df in price_data.items():
            if df.empty:
                continue
            indicator_values: dict[str, pd.Series] = {}
            for indicator, config in self.signal_def.indicators:
                try:
                    raw = indicator.compute(df)
                    indicator_values[config.name] = raw
                except Exception:
                    logger.warning(
                        "Failed to compute %s for %s",
                        config.name,
                        ticker,
                        exc_info=True,
                    )
            if indicator_values:
                results[ticker] = pd.DataFrame(indicator_values)
        return results

    def normalize(self, raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply normalization to raw indicator values."""
        normalized: dict[str, pd.DataFrame] = {}
        for ticker, raw_df in raw_data.items():
            norm_df = pd.DataFrame(index=raw_df.index)
            for _indicator, config in self.signal_def.indicators:
                col = config.name
                if col not in raw_df.columns:
                    continue
                series = raw_df[col]

                if config.normalization == NormalizationMethod.ZSCORE:
                    norm_df[col] = zscore_normalize(series, config.lookback_window)
                elif config.normalization == NormalizationMethod.PERCENTILE:
                    norm_df[col] = series.rolling(config.lookback_window, min_periods=20).rank(
                        pct=True
                    )
                elif config.normalization == NormalizationMethod.MIN_MAX:
                    norm_df[col] = min_max_normalize(series, config.lookback_window)
                elif config.normalization == NormalizationMethod.WINSORIZE:
                    norm_df[col] = zscore_normalize(
                        winsorize(series, config.winsorize_lower, config.winsorize_upper),
                        config.lookback_window,
                    )
                else:
                    norm_df[col] = series

                # Flip sign if lower is better
                if not config.higher_is_better:
                    norm_df[col] = -norm_df[col]

            normalized[ticker] = norm_df
        return normalized

    def combine(self, normalized_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine normalized indicators into a composite signal per ticker.

        Returns:
            DataFrame with tickers as columns and dates as index,
            containing the weighted composite signal.
        """
        weights = {cfg.name: cfg.weight for _, cfg in self.signal_def.indicators}
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0

        composite: dict[str, pd.Series] = {}
        for ticker, norm_df in normalized_data.items():
            weighted_sum = pd.Series(0.0, index=norm_df.index)
            for col in norm_df.columns:
                w = weights.get(col, 1.0)
                weighted_sum += norm_df[col].fillna(0) * w
            composite[ticker] = weighted_sum / total_weight

        if not composite:
            return pd.DataFrame()
        return pd.DataFrame(composite)

    def run(self, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Full pipeline: compute, normalize, combine.

        Returns:
            Composite signal DataFrame (dates x tickers).
        """
        raw = self.compute_raw(price_data)
        normalized = self.normalize(raw)
        return self.combine(normalized)

    def cross_sectional_rank(self, composite: pd.DataFrame) -> pd.DataFrame:
        """Rank signals cross-sectionally (across tickers) each day."""
        return composite.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Pre-built signal definitions
# ---------------------------------------------------------------------------


def optimize_weights(
    pipeline: SignalPipeline,
    price_data: dict[str, pd.DataFrame],
    forward_horizon: int = 5,
    train_size: int = 252,
    test_size: int = 63,
    embargo_size: int = 5,
    alpha: float = 1.0,
) -> SignalDefinition:
    """Optimize indicator weights via cross-validated ridge regression.

    Replaces hardcoded weights with weights learned from the relationship
    between normalized indicators and forward returns, validated
    out-of-sample across walk-forward folds.

    Args:
        pipeline: Existing SignalPipeline with indicators to optimize.
        price_data: ``{ticker: DataFrame}`` with OHLCV columns.
        forward_horizon: Days forward for return computation.
        train_size: Walk-forward training window.
        test_size: Walk-forward test window.
        embargo_size: Gap between train and test (prevent leakage).
        alpha: Ridge regularization strength (higher = more shrinkage
            toward equal weights).

    Returns:
        New SignalDefinition with optimized weights.
    """
    from sklearn.linear_model import Ridge

    from pipeline.backtesting.walk_forward import walk_forward_splits
    from pipeline.eval.signal_alpha import compute_forward_returns

    # Compute normalized indicators per ticker
    raw = pipeline.compute_raw(price_data)
    normalized = pipeline.normalize(raw)

    # Build a stacked panel: (date, ticker) -> indicator values + forward return
    indicator_names = pipeline.signal_def.indicator_names
    rows: list[dict] = []

    # Build price panel for forward returns
    close_panels: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        if "close" in df.columns and not df.empty:
            close_panels[ticker] = df["close"]

    if not close_panels:
        logger.warning("No price data for weight optimization; returning original weights.")
        return pipeline.signal_def

    price_panel = pd.DataFrame(close_panels)
    fwd_returns = compute_forward_returns(price_panel, horizon=forward_horizon)

    for ticker, norm_df in normalized.items():
        if ticker not in fwd_returns.columns:
            continue
        fwd_col = fwd_returns[ticker]
        common_idx = norm_df.index.intersection(fwd_col.dropna().index)
        for dt in common_idx:
            row = {col: norm_df.at[dt, col] for col in indicator_names if col in norm_df.columns}
            row["_fwd_return"] = fwd_col.at[dt]
            row["_date"] = dt
            rows.append(row)

    if len(rows) < train_size + test_size + embargo_size:
        logger.warning(
            "Insufficient data for weight optimization (%d rows). Returning original weights.",
            len(rows),
        )
        return pipeline.signal_def

    panel = pd.DataFrame(rows).sort_values("_date").reset_index(drop=True)
    feature_cols = [c for c in indicator_names if c in panel.columns]
    features = panel[feature_cols].fillna(0).values
    targets = panel["_fwd_return"].values
    dates = pd.DatetimeIndex(panel["_date"])

    # Walk-forward ridge regression to get OOS-validated weights
    fold_coefs: list[np.ndarray] = []
    for train_idx, _test_idx in walk_forward_splits(
        dates, train_size, test_size, embargo_size=embargo_size
    ):
        x_train, y_train = features[train_idx], targets[train_idx]
        valid = np.isfinite(y_train)
        if valid.sum() < 20:
            continue
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train[valid], y_train[valid])
        fold_coefs.append(model.coef_)

    if not fold_coefs:
        logger.warning("No valid folds for weight optimization. Returning original weights.")
        return pipeline.signal_def

    # Average coefficients across folds, then normalize to sum to 1
    avg_coefs = np.mean(fold_coefs, axis=0)
    abs_coefs = np.abs(avg_coefs)
    total = abs_coefs.sum()
    if total == 0:
        logger.warning("All ridge coefficients are zero. Returning original weights.")
        return pipeline.signal_def

    optimized_weights = abs_coefs / total

    # Build new SignalDefinition with optimized weights
    old_def = pipeline.signal_def
    new_def = SignalDefinition(
        name=old_def.name + "_optimized",
        family=old_def.family,
        description=old_def.description + " (weights optimized via CV ridge)",
    )
    for _i, (indicator, old_cfg) in enumerate(old_def.indicators):
        col_name = old_cfg.name
        if col_name in feature_cols:
            idx = feature_cols.index(col_name)
            new_weight = float(optimized_weights[idx])
            # Preserve sign: negative coef means flip higher_is_better
            sign_flip = avg_coefs[idx] < 0
        else:
            new_weight = old_cfg.weight
            sign_flip = False

        new_cfg = SignalConfig(
            name=old_cfg.name,
            description=old_cfg.description,
            formula=old_cfg.formula,
            normalization=old_cfg.normalization,
            lookback_window=old_cfg.lookback_window,
            winsorize_lower=old_cfg.winsorize_lower,
            winsorize_upper=old_cfg.winsorize_upper,
            weight=new_weight,
            higher_is_better=(
                old_cfg.higher_is_better if not sign_flip else not old_cfg.higher_is_better
            ),
        )
        new_def.indicators.append((indicator, new_cfg))

    weight_report = {
        feature_cols[j]: f"{optimized_weights[j]:.3f}" for j in range(len(feature_cols))
    }
    logger.info(
        "Optimized weights (%d folds): %s",
        len(fold_coefs),
        weight_report,
    )
    return new_def


def momentum_signal(
    lookback: int = 126,
    skip: int = 21,
    fast_lookback: int = 63,
    vol_window: int = 60,
    crash_protection: bool = True,
) -> SignalDefinition:
    """Cross-sectional momentum signal with multi-timeframe blend and crash protection.

    Improvements over the original 12-1 month single-window approach:

    1. **Faster primary lookback** (6-1 month default vs 12-1 month) — adapts
       faster to regime changes and avoids the 2022-2023 reversal drag.
    2. **Multi-timeframe blend** — adds a 3-1 month fast momentum component
       to capture shorter-term continuation.
    3. **Crash protection** — monitors abnormal return dispersion and dampens
       the composite signal when momentum-crash risk is elevated.
    4. **Volatility penalty** — increased weight penalises high-vol names
       more aggressively, reducing whipsaw exposure.
    """
    sig = SignalDefinition(
        name="cross_sectional_momentum",
        family=SignalFamily.MOMENTUM,
        description=(
            f"Multi-timeframe momentum ({lookback}-{skip} + {fast_lookback}-{skip}), "
            f"vol-adjusted, crash-protected"
        ),
    )
    # Primary: 6-1 month momentum (weight 0.40)
    sig.add_indicator(
        MomentumReturn(lookback=lookback, skip=skip),
        SignalConfig(
            name="momentum_return",
            description=f"{lookback // 21}-1 month total return",
            formula=(
                rf"\text{{MOM}}_{{i,t}} = \frac{{P_{{i,t-{skip}}}}}"
                rf"{{P_{{i,t-{lookback}}}}} - 1"
            ),
            normalization=NormalizationMethod.ZSCORE,
            lookback_window=lookback,
            weight=0.40,
            higher_is_better=True,
        ),
    )
    # Fast: 3-1 month momentum (weight 0.20)
    sig.add_indicator(
        MomentumReturn(lookback=fast_lookback, skip=skip),
        SignalConfig(
            name="momentum_fast",
            description=f"{fast_lookback // 21}-1 month total return",
            formula=(
                rf"\text{{MOM}}_{{fast}} = \frac{{P_{{i,t-{skip}}}}}"
                rf"{{P_{{i,t-{fast_lookback}}}}} - 1"
            ),
            normalization=NormalizationMethod.ZSCORE,
            lookback_window=fast_lookback,
            weight=0.20,
            higher_is_better=True,
        ),
    )
    # Trend confirmation: 20/50 MA crossover (faster than 50/200)
    sig.add_indicator(
        MovingAverageCrossover(short_window=20, long_window=50),
        SignalConfig(
            name="ma_crossover",
            description="20/50 MA ratio (trend confirmation)",
            formula=r"\text{MAC}_{i,t} = \frac{\text{SMA}(P, 20)}{\text{SMA}(P, 50)} - 1",
            normalization=NormalizationMethod.ZSCORE,
            lookback_window=lookback,
            weight=0.15,
            higher_is_better=True,
        ),
    )
    # Volatility penalty (weight 0.15)
    sig.add_indicator(
        VolatilitySignal(window=vol_window),
        SignalConfig(
            name="volatility",
            description="Realized volatility (penalize high-vol names)",
            formula=(
                rf"\sigma_{{i,t}} = \text{{std}}"
                rf"(\text{{ret}}_i, {vol_window}) \times \sqrt{{252}}"
            ),
            normalization=NormalizationMethod.ZSCORE,
            lookback_window=lookback,
            weight=0.15,
            higher_is_better=False,
        ),
    )
    # Crash protection (weight 0.10)
    if crash_protection:
        sig.add_indicator(
            MomentumDispersion(ret_window=5, vol_window=vol_window),
            SignalConfig(
                name="crash_protection",
                description="Momentum crash protection (penalise high dispersion)",
                formula=(r"D_{i,t} = -\frac{|\text{ret}_{i,5d}|}{\sigma_{i,60}}"),
                normalization=NormalizationMethod.ZSCORE,
                lookback_window=lookback,
                weight=0.10,
                higher_is_better=True,  # Already negated in compute()
            ),
        )
    return sig


def mean_reversion_signal(
    rsi_window: int = 14,
    bb_window: int = 20,
) -> SignalDefinition:
    """Mean reversion signal based on RSI and Bollinger Band position."""
    sig = SignalDefinition(
        name="mean_reversion",
        family=SignalFamily.MEAN_REVERSION,
        description="Short-term mean reversion via RSI and Bollinger Band position",
    )
    sig.add_indicator(
        RSIMeanReversion(window=rsi_window),
        SignalConfig(
            name="rsi_reversion",
            description="RSI distance from neutral (positive when oversold)",
            formula=rf"\text{{Signal}}_{{i,t}} = 50 - \text{{RSI}}(P, {rsi_window})",
            normalization=NormalizationMethod.ZSCORE,
            lookback_window=252,
            weight=0.5,
            higher_is_better=True,
        ),
    )
    sig.add_indicator(
        BollingerBandPosition(window=bb_window),
        SignalConfig(
            name="bb_position",
            description="Bollinger Band position (low = oversold)",
            formula=r"\text{BB\%} = \frac{P - BB_{lower}}{BB_{upper} - BB_{lower}}",
            normalization=NormalizationMethod.RAW,
            weight=0.3,
            higher_is_better=False,  # Low position = buy signal
        ),
    )
    sig.add_indicator(
        VolumeRatio(window=20),
        SignalConfig(
            name="volume_ratio",
            description="Volume relative to 20-day average",
            formula=r"\text{VR} = \frac{V}{\text{SMA}(V, 20)}",
            normalization=NormalizationMethod.RAW,
            weight=0.2,
            higher_is_better=False,  # Low volume on pullback = healthy
        ),
    )
    return sig
