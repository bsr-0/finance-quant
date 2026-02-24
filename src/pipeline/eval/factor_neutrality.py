"""Factor neutrality analysis using FF5 + Momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.eval.metrics import regression_stats


def compute_factor_exposures(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """OLS regression of returns on factor returns.

    factor_returns columns: mkt_rf, smb, hml, rmw, cma, mom, rf
    returns should be excess returns if possible.
    """
    factors = factor_returns.copy()
    if "rf" in factors.columns:
        excess = returns - factors["rf"]
        factors = factors.drop(columns=["rf"])
    else:
        excess = returns

    return regression_stats(factors, excess)


def factor_correlation_gate(
    residual_returns: pd.Series,
    factor_returns: pd.DataFrame,
    threshold: float = 0.2,
) -> tuple[bool, dict[str, float]]:
    """Check abs correlation of residuals with each factor is below threshold."""
    factors = factor_returns.copy()
    if "rf" in factors.columns:
        factors = factors.drop(columns=["rf"])
    residual_returns = residual_returns.dropna()
    corr = {}
    passed = True
    for col in factors.columns:
        aligned = residual_returns.align(factors[col], join="inner")
        if aligned[0].empty:
            corr[col] = np.nan
            continue
        val = float(aligned[0].corr(aligned[1]))
        corr[col] = val
        if np.isfinite(val) and abs(val) >= threshold:
            passed = False
    return passed, corr
