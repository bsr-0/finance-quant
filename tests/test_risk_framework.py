"""Tests for the institutional risk framework: correlation monitor,
risk dashboard, and enhanced stress testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.eval.stress import (
    DEFAULT_SCENARIOS,
    HypotheticalShock,
    apply_hypothetical_shock,
    run_all_stress_tests,
)
from pipeline.infrastructure.correlation_monitor import (
    CorrelationConfig,
    CorrelationMonitor,
)
from pipeline.infrastructure.risk_dashboard import (
    DailyRiskDashboard,
    DashboardConfig,
    format_dashboard_text,
    report_to_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def correlated_returns():
    """Returns for instruments with known correlation structure."""
    np.random.seed(42)
    n = 100
    idx = pd.bdate_range("2024-01-01", periods=n)
    market = np.random.normal(0.001, 0.01, n)

    return pd.DataFrame({
        "AAPL": market + np.random.normal(0, 0.003, n),  # High corr with market
        "MSFT": market + np.random.normal(0, 0.003, n),  # High corr with market
        "GOLD": np.random.normal(0.0005, 0.008, n),      # Low corr
        "BTC": np.random.normal(0.002, 0.03, n),          # Low corr
    }, index=idx)


@pytest.fixture
def sample_portfolio_returns():
    np.random.seed(42)
    n = 252
    idx = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(np.random.normal(0.0003, 0.01, n), index=idx)


# ---------------------------------------------------------------------------
# Correlation Monitor Tests
# ---------------------------------------------------------------------------


class TestCorrelationMonitor:
    def test_compute_correlation_matrix(self, correlated_returns):
        monitor = CorrelationMonitor()
        monitor.update_returns(correlated_returns)
        corr = monitor.compute_correlation_matrix()
        assert corr.shape == (4, 4)
        # Diagonal should be 1
        for col in corr.columns:
            assert corr.loc[col, col] == pytest.approx(1.0)

    def test_high_corr_clustered(self, correlated_returns):
        cfg = CorrelationConfig(high_corr_threshold=0.5, rolling_window=100)
        monitor = CorrelationMonitor(cfg)
        monitor.update_returns(correlated_returns)
        clusters = monitor.get_clusters()
        assert len(clusters) >= 1
        # AAPL and MSFT should be in the same cluster (both follow market)
        for cluster in clusters:
            if "AAPL" in cluster.members:
                assert "MSFT" in cluster.members
                break

    def test_cluster_limit_check(self, correlated_returns):
        cfg = CorrelationConfig(
            high_corr_threshold=0.5,
            max_cluster_notional=100_000,
        )
        monitor = CorrelationMonitor(cfg)
        monitor.update_returns(correlated_returns)

        clusters = monitor.check_cluster_limits(
            positions={"AAPL": 1000, "MSFT": 500, "GOLD": 200},
            prices={"AAPL": 190, "MSFT": 420, "GOLD": 2000},
        )
        # AAPL + MSFT cluster should breach (190k + 210k = 400k > 100k)
        breached = [c for c in clusters if not c.within_limit]
        assert len(breached) >= 1

    def test_factor_exposures(self, correlated_returns):
        monitor = CorrelationMonitor()
        portfolio = correlated_returns.mean(axis=1)
        factor_returns = pd.DataFrame({
            "market": correlated_returns.mean(axis=1),
        }, index=correlated_returns.index)
        exposures = monitor.compute_factor_exposures(
            portfolio, factor_returns, nav=1_000_000
        )
        assert len(exposures) >= 1
        assert exposures[0].factor == "market"

    def test_summary(self, correlated_returns):
        monitor = CorrelationMonitor()
        monitor.update_returns(correlated_returns)
        monitor.get_clusters()
        summary = monitor.summary()
        assert "num_clusters" in summary
        assert "return_history_length" in summary


# ---------------------------------------------------------------------------
# Risk Dashboard Tests
# ---------------------------------------------------------------------------


class TestDailyRiskDashboard:
    def test_generate_report(self, sample_portfolio_returns):
        cfg = DashboardConfig(
            nav=6_000_000_000,
            max_daily_loss=30_000_000,
            max_gross_exposure=12_000_000_000,
        )
        dashboard = DailyRiskDashboard(cfg)

        positions = {"AAPL": 100_000, "MSFT": 50_000}
        prices = {"AAPL": 190, "MSFT": 420}

        report = dashboard.generate_report(
            date=pd.Timestamp("2024-06-15"),
            positions=positions,
            prices=prices,
            returns_history=sample_portfolio_returns,
        )
        assert report.nav == 6_000_000_000
        assert report.gross_exposure > 0
        assert len(report.limit_utilizations) >= 4
        assert len(report.pre_open_checks) >= 3

    def test_limit_utilizations(self):
        cfg = DashboardConfig(
            nav=1_000_000,
            max_gross_exposure=2_000_000,
        )
        dashboard = DailyRiskDashboard(cfg)
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"X": 10_000},
            prices={"X": 100},  # 1M gross
        )
        # Should be at 50% of 2M limit
        gross_limit = [lu for lu in report.limit_utilizations if lu.limit_name == "gross_exposure"]
        assert len(gross_limit) == 1
        assert gross_limit[0].utilization_pct == pytest.approx(50.0)
        assert gross_limit[0].status == "green"

    def test_stress_results_populated(self):
        dashboard = DailyRiskDashboard()
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"AAPL": 1000},
            prices={"AAPL": 190},
        )
        assert len(report.stress_results) >= 1

    def test_pre_open_checks(self, sample_portfolio_returns):
        dashboard = DailyRiskDashboard()
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"AAPL": 100},
            prices={"AAPL": 190},
            returns_history=sample_portfolio_returns,
        )
        assert report.all_checks_passed

    def test_missing_prices_fails_check(self):
        dashboard = DailyRiskDashboard()
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"AAPL": 100},
            prices={},  # No prices!
        )
        price_check = [c for c in report.pre_open_checks if c[0] == "prices_available"]
        assert len(price_check) == 1
        assert not price_check[0][1]  # Failed

    def test_report_to_dict(self, sample_portfolio_returns):
        dashboard = DailyRiskDashboard()
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"X": 100},
            prices={"X": 50},
            returns_history=sample_portfolio_returns,
        )
        d = report_to_dict(report)
        assert "pnl" in d
        assert "exposure" in d
        assert "risk" in d
        assert "limits" in d

    def test_format_text(self, sample_portfolio_returns):
        dashboard = DailyRiskDashboard()
        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions={"X": 100},
            prices={"X": 50},
            returns_history=sample_portfolio_returns,
        )
        text = format_dashboard_text(report)
        assert "DAILY RISK DASHBOARD" in text
        assert "NAV" in text


# ---------------------------------------------------------------------------
# Enhanced Stress Testing Tests
# ---------------------------------------------------------------------------


class TestEnhancedStress:
    def test_default_scenarios_extended(self):
        assert len(DEFAULT_SCENARIOS) >= 4  # More than the original 2

    def test_hypothetical_shock(self):
        positions = {"AAPL": 10_000, "MSFT": 5_000}
        prices = {"AAPL": 190, "MSFT": 420}
        shock = HypotheticalShock("crash", price_shock_pct=-0.10)
        result = apply_hypothetical_shock(positions, prices, shock)
        # Net long portfolio loses money in a crash
        assert result["pnl_impact"] < 0
        assert "AAPL" in result["per_symbol"]

    def test_spread_stress(self):
        positions = {"AAPL": 1000}
        prices = {"AAPL": 190}
        spreads = {"AAPL": 5.0}
        shock = HypotheticalShock("spread_blowout", spread_multiplier=5.0)
        result = apply_hypothetical_shock(positions, prices, shock, spreads)
        assert result["spread_cost_increase"] > 0

    def test_run_all_stress_tests(self, sample_portfolio_returns):
        results = run_all_stress_tests(
            sample_portfolio_returns,
            positions={"AAPL": 1000},
            prices={"AAPL": 190},
        )
        # Should have historical + EVT + hypothetical results
        assert "EVT_TAIL" in results
        assert any(k.startswith("hypo_") for k in results)

    def test_combined_stress(self):
        positions = {"AAPL": 10_000}
        prices = {"AAPL": 190}
        combined = HypotheticalShock(
            "combined",
            price_shock_pct=-0.15,
            spread_multiplier=3.0,
            volatility_multiplier=2.5,
        )
        result = apply_hypothetical_shock(positions, prices, combined)
        assert result["pnl_impact"] < 0  # Price crash hurts
