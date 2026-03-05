"""Tests for gap analysis follow-up items (3, 4, 5).

- Item 3: SEC fundamentals restatement tracking
- Item 4: Historical ADV integration
- Item 5: Market impact feedback loop model
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipeline.backtesting.capacity import capacity_analysis
from pipeline.backtesting.liquidity import compute_historical_adv
from pipeline.backtesting.transaction_costs import (
    FeedbackImpactModel,
    SquareRootImpactModel,
    Trade,
)
from pipeline.extract.sec_fundamentals import (
    SecFundamentalsExtractor,
    point_in_time_fundamentals,
)

# ---------------------------------------------------------------------------
# Item 3: SEC Restatement Tracking
# ---------------------------------------------------------------------------

class TestAmendmentDetection:
    def test_original_filing_not_amendment(self):
        facts_json = {
            "facts": {"us-gaap": {"Revenues": {
                "label": "Revenues",
                "units": {"USD": [
                    {"form": "10-K", "filed": "2023-02-15", "end": "2022-12-31",
                     "val": 100000, "accn": "0001-23-000001", "fy": 2022, "fp": "FY"},
                ]},
            }}},
        }
        rows = SecFundamentalsExtractor._parse_facts(facts_json, "AAPL", 320193)
        assert len(rows) == 1
        assert rows[0]["is_amendment"] is False
        assert rows[0]["form_type"] == "10-K"

    def test_amendment_detected(self):
        facts_json = {
            "facts": {"us-gaap": {"Revenues": {
                "label": "Revenues",
                "units": {"USD": [
                    {"form": "10-K/A", "filed": "2023-04-15", "end": "2022-12-31",
                     "val": 110000, "accn": "0001-23-000002", "fy": 2022, "fp": "FY"},
                ]},
            }}},
        }
        rows = SecFundamentalsExtractor._parse_facts(facts_json, "AAPL", 320193)
        assert len(rows) == 1
        assert rows[0]["is_amendment"] is True
        assert rows[0]["form_type"] == "10-K"  # Normalized to base form
        assert rows[0]["original_form_type"] == "10-K/A"

    def test_10q_amendment(self):
        facts_json = {
            "facts": {"us-gaap": {"NetIncomeLoss": {
                "label": "Net Income",
                "units": {"USD": [
                    {"form": "10-Q/A", "filed": "2023-06-01", "end": "2023-03-31",
                     "val": 50000, "accn": "0001-23-000003", "fy": 2023, "fp": "Q1"},
                ]},
            }}},
        }
        rows = SecFundamentalsExtractor._parse_facts(facts_json, "AAPL", 320193)
        assert len(rows) == 1
        assert rows[0]["is_amendment"] is True
        assert rows[0]["form_type"] == "10-Q"


class TestFilingSequence:
    def test_sequence_assigned(self):
        facts_json = {
            "facts": {"us-gaap": {"Revenues": {
                "label": "Revenues",
                "units": {"USD": [
                    {"form": "10-K", "filed": "2023-02-15", "end": "2022-12-31",
                     "val": 100000, "accn": "0001-23-000001", "fy": 2022, "fp": "FY"},
                    {"form": "10-K/A", "filed": "2023-04-15", "end": "2022-12-31",
                     "val": 110000, "accn": "0001-23-000002", "fy": 2022, "fp": "FY"},
                ]},
            }}},
        }
        rows = SecFundamentalsExtractor._parse_facts(facts_json, "AAPL", 320193)
        rows = SecFundamentalsExtractor._assign_filing_sequence(rows)
        assert len(rows) == 2
        # Original filing should be sequence 1
        original = [r for r in rows if not r["is_amendment"]][0]
        amendment = [r for r in rows if r["is_amendment"]][0]
        assert original["filing_sequence"] == 1
        assert amendment["filing_sequence"] == 2


class TestPointInTime:
    def test_restated_value_not_visible_before_filing(self):
        df = pd.DataFrame([
            {"ticker": "AAPL", "metric_name": "Revenues",
             "fiscal_period_end": date(2022, 12, 31),
             "filing_date": date(2023, 2, 15), "metric_value": 100000},
            {"ticker": "AAPL", "metric_name": "Revenues",
             "fiscal_period_end": date(2022, 12, 31),
             "filing_date": date(2023, 4, 15), "metric_value": 110000},
        ])
        # Before restatement filed
        result = point_in_time_fundamentals(df, as_of=date(2023, 3, 1))
        assert len(result) == 1
        assert result.iloc[0]["metric_value"] == 100000

    def test_restated_value_visible_after_filing(self):
        df = pd.DataFrame([
            {"ticker": "AAPL", "metric_name": "Revenues",
             "fiscal_period_end": date(2022, 12, 31),
             "filing_date": date(2023, 2, 15), "metric_value": 100000},
            {"ticker": "AAPL", "metric_name": "Revenues",
             "fiscal_period_end": date(2022, 12, 31),
             "filing_date": date(2023, 4, 15), "metric_value": 110000},
        ])
        # After restatement filed
        result = point_in_time_fundamentals(df, as_of=date(2023, 5, 1))
        assert len(result) == 1
        assert result.iloc[0]["metric_value"] == 110000

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["ticker", "metric_name", "fiscal_period_end",
                                    "filing_date", "metric_value"])
        result = point_in_time_fundamentals(df, as_of=date(2023, 1, 1))
        assert result.empty


# ---------------------------------------------------------------------------
# Item 4: Historical ADV Integration
# ---------------------------------------------------------------------------

class TestComputeHistoricalADV:
    def test_rolling_mean(self):
        dates = pd.bdate_range("2023-01-01", periods=30)
        volume = pd.DataFrame({"SYM_A": np.ones(30) * 1000}, index=dates)
        adv = compute_historical_adv(volume, window=5, min_periods=1)
        # Constant volume → ADV = 1000 everywhere
        assert adv["SYM_A"].iloc[-1] == pytest.approx(1000.0)

    def test_varying_volume(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        volume = pd.DataFrame({"SYM_A": list(range(1, 11))}, index=dates)
        adv = compute_historical_adv(volume, window=3, min_periods=1)
        # Last 3 values: 8, 9, 10 → mean = 9.0
        assert adv["SYM_A"].iloc[-1] == pytest.approx(9.0)

    def test_min_periods(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        volume = pd.DataFrame({"SYM_A": list(range(1, 11))}, index=dates)
        adv = compute_historical_adv(volume, window=5, min_periods=5)
        # First 4 values should be NaN
        assert adv["SYM_A"].iloc[:4].isna().all()
        assert adv["SYM_A"].iloc[4:].notna().all()


class TestCapacityWithSeriesADV:
    def test_accepts_series_adv(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(252) * 0.01)
        adv_series = pd.Series(np.linspace(50000, 100000, 252))
        result = capacity_analysis(
            returns=returns,
            trades_per_year=50,
            avg_price=50.0,
            adv=adv_series,
        )
        assert result.capacity_estimate >= 0
        assert len(result.net_sharpes) > 0

    def test_scalar_adv_still_works(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(252) * 0.01)
        result = capacity_analysis(
            returns=returns,
            trades_per_year=50,
            avg_price=50.0,
            adv=100000.0,
        )
        assert result.capacity_estimate >= 0


class TestStaticADVWarning:
    def test_warns_on_static_adv(self, caplog):
        from pipeline.backtesting.simulator import PortfolioSimulator, SimulatorConfig

        dates = pd.bdate_range("2023-01-01", periods=10)
        prices = pd.DataFrame({"SYM_A": [100.0] * 10}, index=dates)
        target = pd.DataFrame({"SYM_A": [10.0] * 10}, index=dates)
        # Static ADV: all same value
        adv = pd.DataFrame({"SYM_A": [50000.0] * 10}, index=dates)

        sim = PortfolioSimulator(SimulatorConfig(capital=100000))
        import logging

        with caplog.at_level(logging.WARNING):
            sim.simulate_equity(target, prices, adv=adv)
        assert any("appears static" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Item 5: Feedback Impact Model
# ---------------------------------------------------------------------------

class TestFeedbackImpactModel:
    def test_greater_than_simple_model(self):
        """Feedback model cost >= simple model cost for same trade."""
        trade = Trade(symbol="AAPL", side="buy", quantity=10000, price=150.0, adv=500000)
        simple = SquareRootImpactModel(sigma=0.02, eta=0.25)
        feedback = FeedbackImpactModel(sigma=0.02, eta=0.25, n_slices=10)

        simple_cost = simple.estimate(trade)
        feedback_cost = feedback.estimate(trade)
        assert feedback_cost.market_impact >= simple_cost.market_impact

    def test_single_slice_matches_simple(self):
        """With n_slices=1, feedback model equals simple model."""
        trade = Trade(symbol="AAPL", side="buy", quantity=10000, price=150.0, adv=500000)
        simple = SquareRootImpactModel(sigma=0.02, eta=0.25)
        feedback = FeedbackImpactModel(sigma=0.02, eta=0.25, n_slices=1)

        simple_cost = simple.estimate(trade)
        feedback_cost = feedback.estimate(trade)
        assert feedback_cost.market_impact == pytest.approx(
            simple_cost.market_impact, rel=1e-10,
        )

    def test_cost_scales_superlinearly(self):
        """Larger orders should have disproportionately higher cost per share."""
        small_trade = Trade(symbol="AAPL", side="buy", quantity=1000, price=150.0, adv=100000)
        large_trade = Trade(symbol="AAPL", side="buy", quantity=10000, price=150.0, adv=100000)
        model = FeedbackImpactModel(sigma=0.02, eta=0.25, n_slices=10)

        small_cost = model.estimate(small_trade).market_impact / 1000
        large_cost = model.estimate(large_trade).market_impact / 10000
        # Cost per share should be higher for larger order
        assert large_cost > small_cost

    def test_n_slices_monotonic(self):
        """More slices = more feedback steps = higher total cost."""
        trade = Trade(symbol="AAPL", side="buy", quantity=5000, price=150.0, adv=200000)
        costs = []
        for n_slices in [1, 2, 5, 10, 20]:
            model = FeedbackImpactModel(sigma=0.02, eta=0.25, n_slices=n_slices)
            costs.append(model.estimate(trade).market_impact)
        # Cost should increase monotonically with more slices
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]

    def test_zero_adv_returns_no_impact(self):
        trade = Trade(symbol="AAPL", side="buy", quantity=100, price=150.0, adv=0)
        model = FeedbackImpactModel()
        cost = model.estimate(trade)
        assert cost.market_impact == 0.0
        assert cost.spread_cost > 0  # Still charges spread

    def test_sell_side(self):
        """Sell trades should also compute impact correctly."""
        trade = Trade(symbol="AAPL", side="sell", quantity=5000, price=150.0, adv=200000)
        model = FeedbackImpactModel(sigma=0.02, eta=0.25, n_slices=5)
        cost = model.estimate(trade)
        assert cost.market_impact > 0
