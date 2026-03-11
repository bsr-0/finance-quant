"""Tests for the capital guard QAQC system.

Verifies that every check independently blocks orders that would exceed
capital limits, and that legitimate orders pass through.
"""

from __future__ import annotations

import pytest

from pipeline.execution.capital_guard import (
    AccountSnapshot,
    CapitalGuard,
    CapitalGuardConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockAccountProvider:
    """Provides a configurable account snapshot for testing."""

    def __init__(
        self,
        equity: float = 500.0,
        cash: float = 500.0,
        buying_power: float = 500.0,
        positions_market_value: float = 0.0,
        position_count: int = 0,
        is_margin_account: bool = False,
    ):
        self._snapshot = AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            positions_market_value=positions_market_value,
            position_count=position_count,
            is_margin_account=is_margin_account,
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        return self._snapshot


def make_guard(
    max_capital: float = 300.0,
    max_positions: int = 2,
    require_cash_account: bool = True,
    **provider_kwargs,
) -> CapitalGuard:
    config = CapitalGuardConfig(
        max_capital=max_capital,
        max_positions=max_positions,
        require_cash_account=require_cash_account,
    )
    provider = MockAccountProvider(**provider_kwargs)
    return CapitalGuard(config=config, account_provider=provider)


# ---------------------------------------------------------------------------
# Tests: basic approval
# ---------------------------------------------------------------------------

class TestBasicApproval:
    def test_small_order_approved(self):
        guard = make_guard(max_capital=300, equity=500, cash=500, buying_power=500)
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=150.0)
        assert result.approved
        assert "max_capital_check" in result.checks_passed

    def test_sell_always_approved(self):
        guard = make_guard(max_capital=100)
        result = guard.check_order("AAPL", "sell", shares=10, limit_price=150.0)
        assert result.approved

    def test_zero_equity_rejected(self):
        guard = make_guard(max_capital=300, equity=0, cash=0, buying_power=0)
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=150.0)
        assert not result.approved


# ---------------------------------------------------------------------------
# Tests: QAQC-1 margin check
# ---------------------------------------------------------------------------

class TestMarginCheck:
    def test_margin_account_rejected(self):
        guard = make_guard(
            max_capital=300, equity=500, cash=500, buying_power=500,
            is_margin_account=True, require_cash_account=True,
        )
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=100.0)
        assert not result.approved
        assert "margin_check" in result.checks_failed

    def test_margin_allowed_when_not_required(self):
        guard = make_guard(
            max_capital=300, equity=500, cash=500, buying_power=500,
            is_margin_account=True, require_cash_account=False,
        )
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=100.0)
        assert result.approved

    def test_cash_account_passes(self):
        guard = make_guard(
            max_capital=300, equity=500, cash=500, buying_power=500,
            is_margin_account=False,
        )
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=100.0)
        assert "margin_check" in result.checks_passed


# ---------------------------------------------------------------------------
# Tests: QAQC-2 max capital
# ---------------------------------------------------------------------------

class TestMaxCapital:
    def test_order_exceeding_max_capital_rejected(self):
        guard = make_guard(
            max_capital=200, equity=500, cash=500, buying_power=500,
            positions_market_value=100,  # already $100 deployed
        )
        # This $150 order would bring total to $250 > $200 max
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=150.0)
        assert not result.approved
        assert "max_capital_check" in result.checks_failed

    def test_order_within_max_capital_approved(self):
        guard = make_guard(
            max_capital=300, equity=500, cash=500, buying_power=500,
            positions_market_value=100,
        )
        # $100 deployed + $150 order = $250 < $300 max
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=150.0)
        assert "max_capital_check" in result.checks_passed

    def test_max_capital_exactly_at_limit(self):
        guard = make_guard(
            max_capital=200, equity=500, cash=500, buying_power=500,
            positions_market_value=100,
        )
        # $100 + $100 = $200 exactly at limit
        result = guard.check_order("SPY", "buy", shares=1, limit_price=100.0)
        assert "max_capital_check" in result.checks_passed


# ---------------------------------------------------------------------------
# Tests: QAQC-3 buying power
# ---------------------------------------------------------------------------

class TestBuyingPower:
    def test_exceeds_buying_power_rejected(self):
        guard = make_guard(
            max_capital=500, equity=500, cash=100, buying_power=100,
        )
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=200.0)
        assert not result.approved
        assert "buying_power_check" in result.checks_failed

    def test_within_buying_power_approved(self):
        guard = make_guard(
            max_capital=500, equity=500, cash=500, buying_power=500,
        )
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=100.0)
        assert "buying_power_check" in result.checks_passed


# ---------------------------------------------------------------------------
# Tests: QAQC-4 single order size
# ---------------------------------------------------------------------------

class TestSingleOrderLimit:
    def test_huge_single_order_rejected(self):
        config = CapitalGuardConfig(max_capital=500, max_single_order_dollars=100)
        provider = MockAccountProvider(equity=500, cash=500, buying_power=500)
        guard = CapitalGuard(config=config, account_provider=provider)

        result = guard.check_order("AAPL", "buy", shares=1, limit_price=150.0)
        assert not result.approved
        assert "single_order_check" in result.checks_failed


# ---------------------------------------------------------------------------
# Tests: QAQC-5 position count
# ---------------------------------------------------------------------------

class TestPositionCount:
    def test_at_max_positions_rejected(self):
        guard = make_guard(
            max_capital=500, max_positions=2,
            equity=500, cash=300, buying_power=300,
            position_count=2,
        )
        result = guard.check_order("MSFT", "buy", shares=1, limit_price=100.0)
        assert not result.approved
        assert "position_count_check" in result.checks_failed

    def test_below_max_positions_approved(self):
        guard = make_guard(
            max_capital=500, max_positions=2,
            equity=500, cash=300, buying_power=300,
            position_count=1,
        )
        result = guard.check_order("MSFT", "buy", shares=1, limit_price=100.0)
        assert "position_count_check" in result.checks_passed


# ---------------------------------------------------------------------------
# Tests: QAQC-6 cash buffer
# ---------------------------------------------------------------------------

class TestCashBuffer:
    def test_insufficient_cash_buffer_rejected(self):
        """Order that would leave less than 5% buffer in cash."""
        config = CapitalGuardConfig(
            max_capital=500, min_cash_buffer_pct=0.05,
        )
        # Cash = $110, order = $100, leaves $10 < $25 (5% of $500)
        provider = MockAccountProvider(equity=500, cash=110, buying_power=110)
        guard = CapitalGuard(config=config, account_provider=provider)

        result = guard.check_order("SPY", "buy", shares=1, limit_price=100.0)
        assert not result.approved
        assert "cash_buffer_check" in result.checks_failed


# ---------------------------------------------------------------------------
# Tests: QAQC-7 utilization
# ---------------------------------------------------------------------------

class TestUtilization:
    def test_over_utilization_rejected(self):
        """Order that would put >95% of equity in positions."""
        guard = make_guard(
            max_capital=1000,
            equity=500, cash=100, buying_power=100,
            positions_market_value=400,  # 80% utilised
        )
        # Adding $90 = 490/500 = 98% > 95%
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=90.0)
        assert not result.approved
        assert "utilization_check" in result.checks_failed


# ---------------------------------------------------------------------------
# Tests: QAQC-8 daily order limit
# ---------------------------------------------------------------------------

class TestDailyOrderLimit:
    def test_exceeds_daily_limit(self):
        config = CapitalGuardConfig(max_capital=500, max_daily_orders=3)
        provider = MockAccountProvider(equity=500, cash=500, buying_power=500)
        guard = CapitalGuard(config=config, account_provider=provider)

        # Place 3 orders (all approved)
        for _ in range(3):
            result = guard.check_order("SPY", "buy", shares=1, limit_price=50.0)
            assert result.approved

        # 4th should be rejected
        result = guard.check_order("SPY", "buy", shares=1, limit_price=50.0)
        assert not result.approved
        assert "daily_order_limit" in result.checks_failed


# ---------------------------------------------------------------------------
# Tests: multiple failures
# ---------------------------------------------------------------------------

class TestMultipleFailures:
    def test_all_failures_reported(self):
        """When multiple checks fail, all should be listed."""
        guard = make_guard(
            max_capital=100,          # max $100
            max_positions=1,          # max 1 position
            equity=500, cash=50, buying_power=50,
            positions_market_value=80,  # already $80 deployed
            position_count=1,         # already 1 position
            is_margin_account=True,   # margin enabled
            require_cash_account=True,
        )
        # $200 order on a margin account at max positions with $80 deployed
        result = guard.check_order("TSLA", "buy", shares=1, limit_price=200.0)
        assert not result.approved
        # Should fail margin, max_capital, buying_power, position_count, etc.
        assert len(result.checks_failed) >= 3


# ---------------------------------------------------------------------------
# Tests: account fetch failure
# ---------------------------------------------------------------------------

class TestAccountFetchFailure:
    def test_account_error_rejects_order(self):
        class FailingProvider:
            def get_account_snapshot(self):
                raise ConnectionError("Broker API timeout")

        config = CapitalGuardConfig(max_capital=500)
        guard = CapitalGuard(config=config, account_provider=FailingProvider())
        result = guard.check_order("AAPL", "buy", shares=1, limit_price=100.0)
        assert not result.approved
        assert "account_fetch" in result.checks_failed


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_fractional_shares(self):
        guard = make_guard(max_capital=200, equity=200, cash=200, buying_power=200)
        result = guard.check_order("AMZN", "buy", shares=0.5, limit_price=180.0)
        # 0.5 * 180 = $90 < $200 max, < $120 single order limit (60% of 200)
        assert result.approved

    def test_negative_max_capital_raises(self):
        with pytest.raises(ValueError, match="max_capital must be positive"):
            CapitalGuardConfig(max_capital=-100)

    def test_config_auto_derives_single_order_limit(self):
        config = CapitalGuardConfig(max_capital=500)
        # Default 60% of 500 = 300
        assert config.max_single_order_dollars == 300.0
