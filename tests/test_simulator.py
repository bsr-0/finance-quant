import numpy as np
import pandas as pd

from pipeline.backtesting.simulator import PortfolioSimulator, SimulatorConfig


def test_simulate_equity_basic():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"SPY": [100, 101, 102, 101, 103]}, index=dates)
    positions = pd.DataFrame({"SPY": [0, 10, 10, 0, 0]}, index=dates)

    sim = PortfolioSimulator(SimulatorConfig(capital=10_000, max_leverage=2.0))
    result = sim.simulate_equity(positions, prices)

    assert not result.empty
    assert "net_return" in result.columns


def test_simulate_prediction_market_basic():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"c1": [0.5, 0.6, 0.55]}, index=dates)
    positions = pd.DataFrame({"c1": [0, 100, 100]}, index=dates)

    sim = PortfolioSimulator(SimulatorConfig(capital=1_000, fee_bps=10.0))
    result = sim.simulate_prediction_market(positions, prices)

    assert not result.empty
    assert np.isfinite(result["net_return"]).all()
