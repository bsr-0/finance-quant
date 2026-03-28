"""Evaluation metrics for model performance and risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as z_dist
from scipy.stats import t as t_dist

_TRADING_DAYS = 252


def information_ratio(returns: pd.Series, benchmark: pd.Series | None = None) -> float:
    """Annualized information ratio vs benchmark (or zero if None)."""
    if benchmark is None:
        active = returns
    else:
        aligned_r, aligned_b = returns.align(benchmark, join="inner")
        active = aligned_r - aligned_b
    active = active.dropna()
    if active.empty or active.std() == 0:
        return np.nan
    return float(active.mean() / active.std() * np.sqrt(_TRADING_DAYS))


def sharpe_confidence_interval(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Sharpe ratio with confidence interval using Lo (2002) adjustment.

    Accounts for skewness and kurtosis of the return distribution when
    computing the standard error of the Sharpe ratio, rather than assuming
    IID Gaussian returns.

    Returns:
        Dict with sharpe, se, ci_lower, ci_upper, p_value.
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 10:
        return {
            "sharpe": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }
    excess = returns - risk_free_rate / _TRADING_DAYS
    mu = float(excess.mean())
    sigma = float(excess.std(ddof=1))
    if sigma == 0:
        return {
            "sharpe": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }

    sr_daily = mu / sigma
    sr_annual = sr_daily * np.sqrt(_TRADING_DAYS)

    # Lo (2002) standard error adjusted for skewness/kurtosis
    skew = float(excess.skew())
    kurt = float(excess.kurtosis())  # excess kurtosis
    se_daily = np.sqrt((1 + 0.5 * sr_daily**2 - skew * sr_daily + (kurt / 4) * sr_daily**2) / n)
    se_annual = se_daily * np.sqrt(_TRADING_DAYS)

    z = z_dist.ppf(1 - (1 - confidence) / 2)
    ci_lower = sr_annual - z * se_annual
    ci_upper = sr_annual + z * se_annual
    p_value = float(2 * z_dist.sf(abs(sr_annual / se_annual))) if se_annual > 0 else np.nan

    return {
        "sharpe": float(sr_annual),
        "se": float(se_annual),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": p_value,
    }


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Directional accuracy (fraction of correct sign)."""
    y_true, y_pred = y_true.align(y_pred, join="inner")
    if y_true.empty:
        return np.nan
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def sharpe_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> tuple[float, float]:
    """Annualized Sharpe and Sortino ratios."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan, np.nan
    excess = returns - risk_free_rate / _TRADING_DAYS
    mu = excess.mean()
    sigma = excess.std()
    sharpe = np.nan if sigma == 0 else mu / sigma * np.sqrt(_TRADING_DAYS)
    downside = excess[excess < 0]
    ds = np.sqrt((downside**2).mean()) if len(downside) > 0 else 0.0
    sortino = sharpe if ds == 0 else mu / ds * np.sqrt(_TRADING_DAYS)
    return float(sharpe), float(sortino)


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown from cumulative returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if not dd.isna().all() else 0.0


def drawdown_recovery_time(returns: pd.Series) -> int | float:
    """Max drawdown recovery time in periods (days)."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    is_at_peak = equity >= peak
    groups = is_at_peak.cumsum()
    durations = equity.groupby(groups).cumcount()
    return int(durations.max()) if len(durations) > 0 else np.nan


def turnover(positions: pd.DataFrame) -> pd.Series:
    """Daily turnover from position matrix (sum abs changes / 2)."""
    if positions.empty:
        return pd.Series(dtype=float)
    changes = positions.diff().abs().sum(axis=1)
    return changes / 2.0


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Brier score for probabilistic forecasts (binary)."""
    y_true, y_prob = y_true.align(y_prob, join="inner")
    if y_true.empty:
        return np.nan
    return float(((y_prob - y_true) ** 2).mean())


def log_loss(y_true: pd.Series, y_prob: pd.Series, eps: float = 1e-12) -> float:
    """Log loss for probabilistic forecasts (binary)."""
    y_true, y_prob = y_true.align(y_prob, join="inner")
    if y_true.empty:
        return np.nan
    p = y_prob.clip(eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    """Expected calibration error (ECE)."""
    y_true, y_prob = y_true.align(y_prob, join="inner")
    if y_true.empty:
        return np.nan
    df = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if df.empty:
        return np.nan
    df["bin"] = pd.cut(df["p"], bins=bins, labels=False, include_lowest=True)
    ece = 0.0
    total = len(df)
    for _, grp in df.groupby("bin"):
        if grp.empty:
            continue
        acc = grp["y"].mean()
        conf = grp["p"].mean()
        ece += abs(acc - conf) * len(grp) / total
    return float(ece)


def _newey_west_cov(x_mat: np.ndarray, resid: np.ndarray, n_lags: int | None = None) -> np.ndarray:
    """Newey-West HAC covariance estimator for OLS coefficients.

    Produces heteroskedasticity- and autocorrelation-consistent (HAC)
    standard errors, which are essential for financial time-series
    regressions where residuals are typically both heteroskedastic
    and serially correlated.
    """
    n, k = x_mat.shape
    if n_lags is None:
        n_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # Meat of the sandwich: S = sum of weighted autocovariance matrices
    u = resid.reshape(-1, 1)
    xu = x_mat * u  # n x k
    s_mat = xu.T @ xu / n  # lag-0

    for lag in range(1, n_lags + 1):
        weight = 1 - lag / (n_lags + 1)  # Bartlett kernel
        gamma = xu[lag:].T @ xu[:-lag] / n
        s_mat += weight * (gamma + gamma.T)

    # Sandwich: (X'X)^{-1} S (X'X)^{-1} * n
    xtx_inv = np.linalg.inv(x_mat.T @ x_mat / n)
    return xtx_inv @ s_mat @ xtx_inv / n


def regression_stats(x: pd.DataFrame, y: pd.Series, hac: bool = True) -> dict:
    """OLS regression stats for factor exposures.

    Args:
        x: Regressor DataFrame (factors).
        y: Dependent variable (returns).
        hac: If True, use Newey-West HAC standard errors (recommended
             for financial time-series). If False, use classical OLS SEs.
    """
    x = x.dropna()
    y = y.loc[x.index].dropna()
    x = x.loc[y.index]
    if x.empty or y.empty:
        return {
            "betas": {},
            "t_stats": {},
            "p_values": {},
            "r2": np.nan,
        }
    x_mat = np.column_stack([np.ones(len(x)), x.values])
    y_vec = y.values
    beta = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
    preds = x_mat @ beta
    resid = y_vec - preds
    dof = max(len(y_vec) - x_mat.shape[1], 1)

    if hac and len(y_vec) > 20:
        cov = _newey_west_cov(x_mat, resid)
    else:
        s2 = (resid @ resid) / dof
        cov = s2 * np.linalg.inv(x_mat.T @ x_mat)

    se = np.sqrt(np.abs(np.diag(cov)))
    t_stats = beta / se
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), dof))
    ss_tot = ((y_vec - y_vec.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    columns = ["intercept"] + list(x.columns)
    betas = dict(zip(columns, beta, strict=False))
    t_dict = dict(zip(columns, t_stats, strict=False))
    p_dict = dict(zip(columns, p_vals, strict=False))

    return {
        "betas": betas,
        "t_stats": t_dict,
        "p_values": p_dict,
        "r2": float(r2),
        "residuals": pd.Series(resid, index=x.index),
    }
