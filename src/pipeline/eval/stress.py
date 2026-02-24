"""Stress testing utilities, including EVT tail risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass
class StressScenario:
    name: str
    start: str
    end: str


DEFAULT_SCENARIOS = [
    StressScenario("GFC_2008", "2008-09-01", "2009-06-30"),
    StressScenario("COVID_2020", "2020-02-15", "2020-05-31"),
]


def scenario_metrics(returns: pd.Series, scenario: StressScenario) -> dict[str, float]:
    """Compute stress metrics for a given scenario."""
    window = returns.loc[scenario.start : scenario.end].dropna()
    if window.empty:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "max_dd": float("nan"),
            "recovery_days": float("nan"),
        }
    var = np.percentile(window, 5)
    es = window[window <= var].mean() if (window <= var).any() else float("nan")
    equity = (1 + window).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    is_at_peak = equity >= peak
    groups = is_at_peak.cumsum()
    duration = equity.groupby(groups).cumcount()
    recovery_days = int(duration.max()) if len(duration) > 0 else float("nan")
    return {
        "var": float(var),
        "es": float(es),
        "max_dd": float(dd.min()),
        "recovery_days": recovery_days,
    }


def evt_tail_risk(returns: pd.Series, threshold_quantile: float = 0.95) -> dict[str, float]:
    """Fit a GPD to tail losses and compute tail VaR/ES."""
    returns = returns.dropna()
    if returns.empty:
        return {"tail_var": float("nan"), "tail_es": float("nan")}

    losses = -returns
    threshold = np.quantile(losses, threshold_quantile)
    tail = losses[losses > threshold] - threshold
    if len(tail) < 20:
        return {"tail_var": float("nan"), "tail_es": float("nan")}

    c, loc, scale = genpareto.fit(tail, floc=0)
    var = threshold + genpareto.ppf(0.99, c, loc=loc, scale=scale)
    es = threshold + (scale + c * var) / (1 - c) if c < 1 else float("nan")
    return {"tail_var": float(-var), "tail_es": float(-es)}
