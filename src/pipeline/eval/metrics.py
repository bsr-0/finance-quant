"""Evaluation metrics for model performance and risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    ds = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
    sortino = sharpe if ds == 0 else mu / ds * np.sqrt(_TRADING_DAYS)
    return float(sharpe), float(sortino)


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown from cumulative returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


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


def regression_stats(x: pd.DataFrame, y: pd.Series) -> dict:
    """OLS regression stats for factor exposures."""
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
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
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
