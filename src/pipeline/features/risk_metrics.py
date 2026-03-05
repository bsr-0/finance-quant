"""Risk metrics for quantitative analysis.

Provides institutional-grade risk measures: realized volatility estimators,
Value-at-Risk, drawdown analysis, and risk-adjusted performance ratios.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualisation constants
_TRADING_DAYS_PER_YEAR = 252
_SQRT_252 = np.sqrt(_TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Realized Volatility Estimators
# ---------------------------------------------------------------------------

def close_to_close_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Classical close-to-close realized volatility (annualised)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=2).std() * _SQRT_252


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson (1980) high-low volatility estimator (annualised).

    More efficient than close-to-close because it uses intraday range.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2))
    daily_var = factor * log_hl ** 2
    return np.sqrt(daily_var.rolling(window, min_periods=2).mean() * _TRADING_DAYS_PER_YEAR)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass (1980) volatility estimator (annualised).

    Uses OHLC data for a more efficient estimate than Parkinson.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    daily_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(daily_var.rolling(window, min_periods=2).mean() * _TRADING_DAYS_PER_YEAR)


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Yang-Zhang (2000) volatility estimator (annualised).

    Combines overnight and intraday components; handles opening jumps.
    """
    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    # Rogers-Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    overnight_var = log_oc.rolling(window, min_periods=2).var()
    close_var = log_co.rolling(window, min_periods=2).var()
    rs_var = rs.rolling(window, min_periods=2).mean()

    yz_var = overnight_var + k * close_var + (1 - k) * rs_var
    return np.sqrt(yz_var.clip(lower=0) * _TRADING_DAYS_PER_YEAR)


def ewma_vol(close: pd.Series, span: int = 60) -> pd.Series:
    """Exponentially-weighted volatility (RiskMetrics style, annualised)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span, min_periods=2).std() * _SQRT_252


# ---------------------------------------------------------------------------
# Value-at-Risk / Conditional VaR
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """Rolling historical Value-at-Risk.

    Returns the loss threshold such that losses exceed it only
    ``(1 - confidence)`` fraction of the time.
    """
    quantile = 1 - confidence
    return returns.rolling(window, min_periods=20).quantile(quantile)


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """Rolling Conditional Value-at-Risk (Expected Shortfall).

    Mean of returns below the VaR threshold.
    """
    quantile = 1 - confidence

    def _es(x: pd.Series) -> float:
        threshold = np.nanpercentile(x, quantile * 100)
        tail = x[x <= threshold]
        return tail.mean() if len(tail) > 0 else np.nan

    return returns.rolling(window, min_periods=20).apply(_es, raw=False)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 60,
) -> pd.Series:
    """Parametric (Gaussian) Value-at-Risk.

    .. warning:: Assumes Gaussian returns.  For fat-tailed distributions
       use :func:`cornish_fisher_var` instead.
    """
    from scipy.stats import norm

    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window, min_periods=10).mean()
    sigma = returns.rolling(window, min_periods=10).std()
    return mu + z * sigma


def cornish_fisher_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """Cornish-Fisher VaR adjusting for skewness and kurtosis.

    Provides a more accurate VaR estimate for non-Gaussian returns
    by expanding the quantile function to account for the third and
    fourth moments of the return distribution.
    """
    from scipy.stats import norm

    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window, min_periods=60).mean()
    sigma = returns.rolling(window, min_periods=60).std()
    skew = returns.rolling(window, min_periods=60).skew()
    kurt = returns.rolling(window, min_periods=60).kurt()  # excess kurtosis

    # Cornish-Fisher expansion
    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)

    return mu + z_cf * sigma


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------

def drawdown_series(prices: pd.Series) -> pd.Series:
    """Compute running drawdown from peak (as a negative fraction)."""
    peak = prices.expanding().max()
    return (prices - peak) / peak


def max_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown over *window* periods."""
    dd = drawdown_series(prices)
    return dd.rolling(window, min_periods=2).min()


def drawdown_duration(prices: pd.Series) -> pd.Series:
    """Number of periods since the last equity high."""
    peak = prices.expanding().max()
    is_at_peak = prices >= peak
    groups = is_at_peak.cumsum()
    duration = prices.groupby(groups).cumcount()
    return duration


# ---------------------------------------------------------------------------
# Risk-adjusted Performance Ratios
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 252,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    excess = returns - risk_free_rate / _TRADING_DAYS_PER_YEAR
    mu = excess.rolling(window, min_periods=20).mean()
    sigma = excess.rolling(window, min_periods=20).std()
    return (mu / sigma) * _SQRT_252


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 252,
) -> pd.Series:
    """Rolling annualised Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / _TRADING_DAYS_PER_YEAR

    def _downside_std(x: pd.Series) -> float:
        neg = x[x < 0]
        return np.sqrt((neg ** 2).mean()) if len(neg) > 0 else np.nan

    mu = excess.rolling(window, min_periods=20).mean()
    ds = excess.rolling(window, min_periods=20).apply(_downside_std, raw=False)
    return (mu / ds) * _SQRT_252


def calmar_ratio(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling Calmar ratio (annualised return / max drawdown)."""
    log_ret = np.log(prices / prices.shift(1))
    ann_return = log_ret.rolling(window, min_periods=20).mean() * _TRADING_DAYS_PER_YEAR
    mdd = max_drawdown(prices, window).abs().replace(0, np.nan)
    return ann_return / mdd


# ---------------------------------------------------------------------------
# Higher Moments
# ---------------------------------------------------------------------------

def rolling_skewness(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling skewness of returns.

    Default window is 60 observations.  Sample skewness from fewer
    observations has standard error > 0.5; callers should use
    ``window >= 60`` for meaningful estimates.
    """
    return returns.rolling(window, min_periods=window).skew()


def rolling_kurtosis(returns: pd.Series, window: int = 120) -> pd.Series:
    """Rolling excess kurtosis of returns.

    Default window is 120 observations.  Sample kurtosis is extremely
    noisy — from 60 observations the standard error exceeds 1.0, making
    point estimates unreliable for risk decisions.  Callers should use
    ``window >= 120`` for meaningful estimates.
    """
    return returns.rolling(window, min_periods=window).kurt()


# ---------------------------------------------------------------------------
# Trend / Mean-Reversion Detection
# ---------------------------------------------------------------------------

def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent via the aggregated variance method.

    H < 0.5  → mean-reverting
    H ≈ 0.5  → random walk
    H > 0.5  → trending
    """
    ts = np.asarray(prices.dropna(), dtype=float)
    if len(ts) < max_lag + 1:
        return np.nan

    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        tau.append(np.std(diff))

    tau = np.asarray(tau, dtype=float)
    valid = np.isfinite(tau) & (tau > 0)
    if valid.sum() < 5:
        return np.nan

    slope, _ = np.polyfit(np.log(np.asarray(list(lags))[valid]), np.log(tau[valid]), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Convenience: calculate all risk metrics for a price DataFrame
# ---------------------------------------------------------------------------

def calculate_risk_metrics(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    volume_col: str = "volume",
    window: int = 60,
) -> pd.DataFrame:
    """Append a comprehensive set of risk-metric columns to *df*."""
    result = df.copy()
    prices = df[price_col]
    returns = np.log(prices / prices.shift(1))

    # Volatility estimators
    result["vol_cc_20"] = close_to_close_vol(prices, 20)
    result["vol_cc_60"] = close_to_close_vol(prices, 60)
    result["vol_ewma_60"] = ewma_vol(prices, 60)
    if high_col in df.columns and low_col in df.columns:
        result["vol_parkinson_20"] = parkinson_vol(df[high_col], df[low_col], 20)
        if open_col in df.columns:
            result["vol_gk_20"] = garman_klass_vol(
                df[open_col], df[high_col], df[low_col], prices, 20
            )
            result["vol_yz_20"] = yang_zhang_vol(
                df[open_col], df[high_col], df[low_col], prices, 20
            )

    # VaR / CVaR
    result["var_95_60d"] = historical_var(returns, 0.95, window)
    result["cvar_95_60d"] = historical_cvar(returns, 0.95, window)

    # Drawdown
    result["drawdown"] = drawdown_series(prices)
    result["max_drawdown_252d"] = max_drawdown(prices, 252)
    result["drawdown_duration"] = drawdown_duration(prices)

    # Performance ratios
    result["sharpe_252d"] = sharpe_ratio(returns, window=252)
    result["sortino_252d"] = sortino_ratio(returns, window=252)

    # Higher moments
    result["skewness_60d"] = rolling_skewness(returns, window)
    result["kurtosis_60d"] = rolling_kurtosis(returns, window)

    return result
