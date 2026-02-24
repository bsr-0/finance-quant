import numpy as np
import pandas as pd

from pipeline.eval.factor_neutrality import compute_factor_exposures


def test_factor_regression_recovers_beta():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=200)
    factors = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0, 0.01, size=len(dates)),
            "smb": rng.normal(0, 0.01, size=len(dates)),
            "hml": rng.normal(0, 0.01, size=len(dates)),
            "rmw": rng.normal(0, 0.01, size=len(dates)),
            "cma": rng.normal(0, 0.01, size=len(dates)),
            "mom": rng.normal(0, 0.01, size=len(dates)),
            "rf": np.full(len(dates), 0.0001),
        },
        index=dates,
    )

    returns = 0.5 * factors["mkt_rf"] + 0.2 * factors["mom"] + rng.normal(0, 0.005, len(dates))

    result = compute_factor_exposures(returns, factors)
    beta_mkt = result["betas"].get("mkt_rf")
    beta_mom = result["betas"].get("mom")
    assert beta_mkt is not None
    assert beta_mom is not None
    assert 0.2 < beta_mkt < 0.8
    assert 0.0 < beta_mom < 0.5
