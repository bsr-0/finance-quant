import numpy as np
import pandas as pd

from pipeline.eval.metrics import (
    brier_score,
    calibration_error,
    hit_rate,
    information_ratio,
    log_loss,
    max_drawdown,
    sharpe_sortino,
)


def test_information_ratio_zero_benchmark():
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    ir = information_ratio(returns)
    assert np.isfinite(ir)


def test_hit_rate_directional():
    y_true = pd.Series([0.01, -0.02, 0.03, -0.01])
    y_pred = pd.Series([0.1, -0.1, -0.2, -0.3])
    assert hit_rate(y_true, y_pred) == 0.75


def test_sharpe_sortino():
    returns = pd.Series([0.01] * 100)
    sharpe, sortino = sharpe_sortino(returns)
    assert sharpe > 0
    assert sortino > 0


def test_max_drawdown():
    returns = pd.Series([0.05, -0.1, 0.02, -0.02, 0.01])
    dd = max_drawdown(returns)
    assert dd <= 0


def test_prob_metrics():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = pd.Series([0.9, 0.2, 0.7, 0.1])
    assert brier_score(y_true, y_prob) >= 0
    assert log_loss(y_true, y_prob) >= 0
    assert calibration_error(y_true, y_prob) >= 0
