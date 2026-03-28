"""Factor neutrality analysis using FF5 + Momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from pipeline.eval.metrics import regression_stats
from pipeline.eval.robustness import holm_bonferroni


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
    alpha: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    """Check abs correlation of residuals with each factor is below threshold.

    Applies Holm-Bonferroni correction to control the family-wise error
    rate when testing multiple factor correlations simultaneously.

    Args:
        residual_returns: Strategy residual returns after factor regression.
        factor_returns: Factor return DataFrame.
        threshold: Absolute correlation threshold.
        alpha: Significance level after multiple testing correction.

    Returns:
        (passed, correlations) where passed is True only if no factor
        has both |correlation| >= threshold AND a statistically
        significant p-value after Holm-Bonferroni correction.
    """
    factors = factor_returns.copy()
    if "rf" in factors.columns:
        factors = factors.drop(columns=["rf"])
    residual_returns = residual_returns.dropna()
    corr: dict[str, float] = {}
    raw_pvals: list[float] = []
    factor_names: list[str] = []
    passed = True

    for col in factors.columns:
        aligned_r, aligned_f = residual_returns.align(factors[col], join="inner")
        aligned_r = aligned_r.dropna()
        aligned_f = aligned_f.loc[aligned_r.index].dropna()
        aligned_r = aligned_r.loc[aligned_f.index]
        if len(aligned_r) < 10:
            corr[col] = np.nan
            continue
        r, p = pearsonr(aligned_r.values, aligned_f.values)
        corr[col] = float(r)
        raw_pvals.append(float(p))
        factor_names.append(col)

    # Apply Holm-Bonferroni correction for multiple comparisons
    if raw_pvals:
        adjusted_pvals = holm_bonferroni(raw_pvals)
        for name, adj_p in zip(factor_names, adjusted_pvals, strict=False):
            r_val = corr[name]
            if np.isfinite(r_val) and abs(r_val) >= threshold and adj_p < alpha:
                passed = False

    return passed, corr
