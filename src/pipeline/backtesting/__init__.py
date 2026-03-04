"""Backtesting framework for point-in-time validated strategy evaluation.

Modules:
    simulator      - Portfolio simulation with cash, leverage, and costs.
    walk_forward   - Walk-forward and purged k-fold cross-validation.
    transaction_costs - Pluggable transaction cost models.
    capacity       - Strategy capacity and parameter sensitivity analysis.
    event_engine   - Event-driven backtesting engine.
    monte_carlo    - Monte Carlo simulation for PnL path analysis.
    survivorship   - Survivorship bias handling and point-in-time universes.
    bias_checks    - Look-ahead bias detection and prevention.
"""

from pipeline.backtesting.simulator import PortfolioSimulator, SimulatorConfig

__all__ = ["PortfolioSimulator", "SimulatorConfig"]
