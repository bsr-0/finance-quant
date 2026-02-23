"""Robust statistical estimators for financial data.

Provides Winsorization, Median Absolute Deviation, robust covariance
estimation (Ledoit-Wolf shrinkage), and outlier-resistant rolling
statistics that are essential for processing noisy market data.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Clip extreme values to the given percentiles."""
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


def rolling_winsorize(
    series: pd.Series,
    window: int = 60,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Winsorize each value against the trailing *window* distribution."""
    lo = series.rolling(window, min_periods=10).quantile(lower_pct)
    hi = series.rolling(window, min_periods=10).quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# Median Absolute Deviation (MAD)
# ---------------------------------------------------------------------------

def mad(series: pd.Series) -> float:
    """Median Absolute Deviation (population)."""
    median = series.median()
    return (series - median).abs().median()


def rolling_mad(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Median Absolute Deviation."""
    return series.rolling(window, min_periods=10).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )


def mad_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Modified Z-score using MAD instead of std.

    More robust to outliers than the classical Z-score.
    ``score = 0.6745 * (x - median) / MAD``
    """
    med = series.rolling(window, min_periods=10).median()
    m = rolling_mad(series, window)
    # 0.6745 is the 0.75th quantile of the standard normal distribution,
    # used to make the MAD comparable to std for normal data.
    return 0.6745 * (series - med) / m.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Robust Rolling Statistics
# ---------------------------------------------------------------------------

def robust_mean(series: pd.Series, window: int = 60, trim_pct: float = 0.05) -> pd.Series:
    """Trimmed mean: remove top/bottom *trim_pct* before averaging."""
    def _trimmed_mean(x: np.ndarray) -> float:
        s = np.sort(x)
        n = len(s)
        lo = int(n * trim_pct)
        hi = n - lo
        if hi <= lo:
            return np.mean(s)
        return np.mean(s[lo:hi])

    return series.rolling(window, min_periods=10).apply(_trimmed_mean, raw=True)


def robust_std(series: pd.Series, window: int = 60) -> pd.Series:
    """Robust std estimated as ``1.4826 * MAD`` (consistent for normal data)."""
    return 1.4826 * rolling_mad(series, window)


def iqr(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Inter-Quartile Range."""
    q75 = series.rolling(window, min_periods=10).quantile(0.75)
    q25 = series.rolling(window, min_periods=10).quantile(0.25)
    return q75 - q25


# ---------------------------------------------------------------------------
# Outlier Detection
# ---------------------------------------------------------------------------

def detect_outliers_zscore(
    series: pd.Series,
    window: int = 60,
    threshold: float = 3.0,
) -> pd.Series:
    """Return boolean mask of outliers using classical rolling Z-score."""
    mu = series.rolling(window, min_periods=10).mean()
    sigma = series.rolling(window, min_periods=10).std()
    z = ((series - mu) / sigma.replace(0, np.nan)).abs()
    return z > threshold


def detect_outliers_mad(
    series: pd.Series,
    window: int = 60,
    threshold: float = 3.0,
) -> pd.Series:
    """Return boolean mask of outliers using MAD-based Z-score."""
    z = mad_zscore(series, window).abs()
    return z > threshold


def detect_outliers_iqr(
    series: pd.Series,
    window: int = 60,
    multiplier: float = 1.5,
) -> pd.Series:
    """Return boolean mask of outliers using IQR fences."""
    q75 = series.rolling(window, min_periods=10).quantile(0.75)
    q25 = series.rolling(window, min_periods=10).quantile(0.25)
    iq = q75 - q25
    lower = q25 - multiplier * iq
    upper = q75 + multiplier * iq
    return (series < lower) | (series > upper)


# ---------------------------------------------------------------------------
# Covariance Estimation
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage estimator for the covariance matrix.

    Shrinks the sample covariance toward a structured target (scaled
    identity) to reduce estimation error, which is critical when the
    number of assets is comparable to the number of observations.

    Returns:
        (shrunk_cov, shrinkage_intensity)
    """
    X = returns.dropna().values
    n, p = X.shape
    if n < 2 or p < 1:
        return pd.DataFrame(np.eye(p), index=returns.columns, columns=returns.columns), 1.0

    # Sample covariance
    X_centered = X - X.mean(axis=0)
    S = (X_centered.T @ X_centered) / n

    # Target: scaled identity
    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # Optimal shrinkage intensity (Ledoit & Wolf 2004)
    delta = S - F
    sum_sq = (delta ** 2).sum()

    X2 = X_centered ** 2
    phi_mat = (X2.T @ X2) / n - S ** 2
    phi = phi_mat.sum()

    kappa = (phi / sum_sq) if sum_sq > 0 else 1.0
    shrinkage = max(0.0, min(1.0, kappa / n))

    sigma = (1 - shrinkage) * S + shrinkage * F
    cov_df = pd.DataFrame(sigma, index=returns.columns, columns=returns.columns)
    return cov_df, float(shrinkage)


def ewm_correlation(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Exponentially-weighted correlation matrix (latest observation)."""
    return returns.ewm(span=span, min_periods=10).corr().iloc[-len(returns.columns):]


# ---------------------------------------------------------------------------
# Convenience: clean a return series before feeding to models
# ---------------------------------------------------------------------------

def clean_returns(
    returns: pd.Series,
    window: int = 60,
    winsor_pct: Tuple[float, float] = (0.01, 0.99),
    outlier_threshold: float = 5.0,
) -> pd.Series:
    """Winsorize and replace extreme outliers with NaN.

    Use this before computing features to prevent a single fat-finger
    trade from contaminating rolling statistics.
    """
    cleaned = winsorize(returns.copy(), winsor_pct[0], winsor_pct[1])
    outliers = detect_outliers_mad(cleaned, window, outlier_threshold)
    cleaned[outliers] = np.nan
    return cleaned
