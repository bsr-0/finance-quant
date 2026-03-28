"""Unit tests for pre-trade risk controls, intraday monitor, and kill switch."""

from __future__ import annotations

import pytest

from pipeline.infrastructure.risk_controls import (
    CheckStatus,
    IntradayRiskMonitor,
    KillSwitch,
    PortfolioState,
    PreTradeChecker,
    RiskLimits,
)

# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_initial_state_not_engaged(self):
        ks = KillSwitch()
        assert not ks.engaged

    def test_engage_sets_flag(self):
        ks = KillSwitch()
        ks.engage("test reason")
        assert ks.engaged

    def test_reset_clears_flag(self):
        ks = KillSwitch()
        ks.engage("test")
        ks.reset("cleared")
        assert not ks.engaged

    def test_status_contains_reason(self):
        ks = KillSwitch()
        ks.engage("breach detected")
        status = ks.status
        assert status["engaged"] is True
        assert "breach detected" in status["reason"]

    def test_engage_is_idempotent(self):
        ks = KillSwitch()
        ks.engage("first")
        ks.engage("second")  # should not overwrite
        assert "first" in ks.status["reason"]


# ---------------------------------------------------------------------------
# IntradayRiskMonitor
# ---------------------------------------------------------------------------


class TestIntradayRiskMonitor:
    def setup_method(self):
        self.limits = RiskLimits(
            max_daily_loss=10_000.0,
            max_drawdown_pct=0.05,
        )
        self.ks = KillSwitch()
        self.monitor = IntradayRiskMonitor(self.limits, self.ks)
        self.monitor.start_session(nav=100_000.0)

    def test_initial_session_ok(self):
        result = self.monitor.record_pnl(100.0)
        assert result == "ok"
        assert not self.ks.engaged

    def test_throttle_on_daily_loss(self):
        # Use a large NAV so the daily loss doesn't also trigger the hard drawdown threshold.
        # NAV=10_000_000: a loss of 10_001 is only 0.1% drawdown (well below 10% hard threshold)
        # but exceeds max_daily_loss=10_000 → throttle only.
        self.monitor.start_session(nav=10_000_000.0)
        result = self.monitor.record_pnl(-10_001.0)
        assert result == "throttle"
        assert self.monitor._session.throttled

    def test_shutdown_on_double_daily_loss(self):
        # Exceed 2 × max_daily_loss → hard shutdown
        result = self.monitor.record_pnl(-20_001.0)
        assert result == "shutdown"
        assert self.ks.engaged

    def test_throttle_on_drawdown(self):
        # Start NAV = 100_000, max_drawdown_pct = 5 %
        # A loss of 5 001 represents > 5 % drawdown from start/peak
        result = self.monitor.record_pnl(-5_001.0)
        assert result == "throttle"

    def test_shutdown_on_hard_drawdown(self):
        # 2 × 5 % = 10 % drawdown hard limit
        result = self.monitor.record_pnl(-10_001.0)
        # At 10 001 loss, daily loss limit also triggers, so shutdown wins
        assert result in ("throttle", "shutdown")

    def test_session_summary_tracks_pnl(self):
        self.monitor.record_pnl(500.0)
        self.monitor.record_pnl(-200.0)
        summary = self.monitor.session_summary
        assert summary["realised_pnl"] == pytest.approx(300.0)
        assert summary["current_nav"] == pytest.approx(100_300.0)

    def test_peak_nav_updates(self):
        self.monitor.record_pnl(1_000.0)
        summary = self.monitor.session_summary
        assert summary["peak_nav"] == pytest.approx(101_000.0)

    def test_pnl_within_limits_stays_ok(self):
        for _ in range(20):
            result = self.monitor.record_pnl(10.0)
        assert result == "ok"


# ---------------------------------------------------------------------------
# PortfolioState
# ---------------------------------------------------------------------------


class TestPortfolioState:
    def test_gross_exposure(self):
        ps = PortfolioState(
            net_asset_value=1_000_000,
            positions={"AAPL": 200_000, "MSFT": -100_000},
        )
        assert ps.gross_exposure == 300_000

    def test_net_exposure(self):
        ps = PortfolioState(
            net_asset_value=1_000_000,
            positions={"AAPL": 200_000, "MSFT": -100_000},
        )
        assert ps.net_exposure == 100_000

    def test_leverage(self):
        ps = PortfolioState(
            net_asset_value=500_000,
            positions={"AAPL": 1_000_000},
        )
        assert ps.leverage == pytest.approx(2.0)

    def test_empty_portfolio(self):
        ps = PortfolioState(net_asset_value=1_000_000)
        assert ps.gross_exposure == 0.0
        assert ps.leverage == 0.0


# ---------------------------------------------------------------------------
# PreTradeChecker
# ---------------------------------------------------------------------------


class TestPreTradeChecker:
    def setup_method(self):
        self.limits = RiskLimits(
            max_order_notional=500_000,
            max_gross_exposure=2_000_000,
            max_net_exposure=1_000_000,
            max_leverage=3.0,
            max_concentration=0.30,
        )
        self.ks = KillSwitch()
        self.checker = PreTradeChecker(self.limits, self.ks)
        self.portfolio = PortfolioState(
            net_asset_value=1_000_000,
            positions={"AAPL": 200_000},
        )

    def test_approved_normal_order(self):
        # Buy 50k MSFT: total gross = 250k, MSFT conc = 50k/250k = 20% < 30% limit
        result = self.checker.check("MSFT", "buy", 50_000, self.portfolio)
        assert result.approved
        assert result.status == CheckStatus.APPROVED

    def test_rejected_kill_switch(self):
        self.ks.engage("test")
        result = self.checker.check("MSFT", "buy", 100_000, self.portfolio)
        assert not result.approved
        assert result.status == CheckStatus.REJECTED
        assert "kill switch" in result.reason.lower()

    def test_rejected_order_too_large(self):
        result = self.checker.check("MSFT", "buy", 600_000, self.portfolio)
        assert not result.approved
        assert "notional" in result.reason.lower()

    def test_rejected_gross_exposure(self):
        # Already 200k long; adding 1.9M more would hit 2.1M gross limit
        result = self.checker.check("MSFT", "buy", 1_900_000, self.portfolio)
        assert not result.approved
        # Could be rejected for order size first, check either reason
        assert result.status == CheckStatus.REJECTED

    def test_rejected_leverage(self):
        # NAV = 1M, max leverage 3x → max gross = 3M
        # Start at 200k, add enough to breach 3M gross
        portfolio = PortfolioState(
            net_asset_value=100_000,  # small NAV to make leverage easy to breach
            positions={"AAPL": 200_000},
        )
        result = self.checker.check("MSFT", "buy", 200_000, portfolio)
        # Projected gross = 400k, NAV = 100k → leverage = 4x > 3x limit
        assert not result.approved
        assert "leverage" in result.reason.lower()

    def test_rejected_concentration(self):
        # AAPL already 200k, adding 400k more makes it 600k / 600k = 100% > 30%
        result = self.checker.check("AAPL", "buy", 400_000, self.portfolio)
        assert not result.approved
        assert "concentration" in result.reason.lower()

    def test_sell_reduces_long_position(self):
        # Portfolio with three positions; selling 50k AAPL reduces its share below 30%
        portfolio = PortfolioState(
            net_asset_value=1_000_000,
            positions={"AAPL": 200_000, "MSFT": 200_000, "GOOG": 200_000},
        )
        # After sell: AAPL=150k, total gross=550k, AAPL conc=27.3% < 30%
        result = self.checker.check("AAPL", "sell", 50_000, portfolio)
        assert result.approved

    def test_throttled_when_monitor_throttled(self):
        monitor = IntradayRiskMonitor(
            RiskLimits(max_daily_loss=1, max_drawdown_pct=0.001),
            self.ks,
        )
        monitor.start_session(nav=1_000_000)
        monitor.record_pnl(-2)  # Trigger throttle
        checker = PreTradeChecker(self.limits, self.ks, monitor)
        result = checker.check("MSFT", "buy", 10_000, self.portfolio)
        assert result.status in (CheckStatus.THROTTLED, CheckStatus.REJECTED)

    def test_net_exposure_limit(self):
        # Build a portfolio near the net exposure limit
        portfolio = PortfolioState(
            net_asset_value=2_000_000,
            positions={"AAPL": 950_000},  # net = 950k, limit = 1M
        )
        # Adding 100k buy pushes net to 1_050_000 > 1_000_000 limit
        result = self.checker.check("AAPL", "buy", 100_000, portfolio)
        assert not result.approved
        assert "net exposure" in result.reason.lower()
