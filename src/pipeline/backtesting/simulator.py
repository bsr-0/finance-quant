"""Institution-grade portfolio simulation with cash, leverage, and costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.backtesting.transaction_costs import CostModel, FixedPlusSpreadModel, Trade


@dataclass
class SimulatorConfig:
    capital: float = 1_000_000.0
    max_leverage: float = 2.0
    max_adv_pct: float = 0.1
    borrow_cost_bps: float = 30.0
    slippage_bps: float = 2.0
    fee_bps: float = 0.0
    allow_partial_fills: bool = True


class PortfolioSimulator:
    """Simulate portfolio evolution under realistic constraints."""

    def __init__(self, config: SimulatorConfig | None = None, cost_model: CostModel | None = None):
        self.config = config or SimulatorConfig()
        self.cost_model = cost_model or FixedPlusSpreadModel()

    def _apply_trade_limits(
        self,
        trades: pd.Series,
        prices: pd.Series,
        adv: pd.Series | None,
    ) -> pd.Series:
        if adv is None:
            return trades

        max_qty = adv * self.config.max_adv_pct
        limited = trades.copy()
        for sym, qty in trades.items():
            limit = max_qty.get(sym, np.inf)
            if np.isnan(limit):
                continue
            if abs(qty) > limit:
                limited[sym] = np.sign(qty) * limit if self.config.allow_partial_fills else 0.0
        return limited

    def simulate_equity(
        self,
        target_positions: pd.DataFrame,
        prices: pd.DataFrame,
        adv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Simulate equity portfolio with cash and leverage constraints."""
        if target_positions.empty or prices.empty:
            return pd.DataFrame()

        prices = prices.sort_index()
        target_positions = target_positions.reindex(prices.index).fillna(0.0)
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        cash = self.config.capital
        prev_value = self.config.capital

        records = []

        current_pos = pd.Series(0.0, index=prices.columns)
        for dt in prices.index:
            px = prices.loc[dt].fillna(0.0)
            desired = target_positions.loc[dt].reindex(px.index).fillna(0.0)

            # Leverage constraint: scale desired exposure if needed
            notional = (desired.abs() * px).sum()
            max_notional = self.config.max_leverage * (cash + (current_pos * px).sum())
            if max_notional > 0 and notional > max_notional:
                scale = max_notional / notional
                desired = desired * scale

            trades = desired - current_pos
            adv_row = adv.loc[dt] if adv is not None and dt in adv.index else None
            trades = self._apply_trade_limits(trades, px, adv_row)

            # Apply trades and costs
            total_cost = 0.0
            slippage_cost = 0.0
            for sym, qty in trades.items():
                if qty == 0:
                    continue
                price = px.get(sym, 0.0)
                if price <= 0:
                    continue
                trade_cost = self.cost_model.estimate(Trade(symbol=sym, side="buy" if qty > 0 else "sell", quantity=qty, price=price))
                total_cost += trade_cost.total
                slippage_cost += abs(qty * price) * (self.config.slippage_bps / 10_000)

                cash -= qty * price

            total_cost += slippage_cost
            cash -= total_cost

            # Borrow cost on shorts (daily)
            short_notional = (current_pos.clip(upper=0).abs() * px).sum()
            borrow_cost = short_notional * (self.config.borrow_cost_bps / 10_000) / 252.0
            cash -= borrow_cost
            total_cost += borrow_cost

            new_pos = current_pos + trades
            positions.loc[dt] = new_pos
            current_pos = new_pos

            portfolio_value = cash + (new_pos * px).sum()
            net_return = 0.0 if prev_value == 0 else (portfolio_value - prev_value) / prev_value
            prev_value = portfolio_value

            records.append(
                {
                    "date": dt,
                    "gross_value": float((new_pos * px).sum()),
                    "cash": float(cash),
                    "net_value": float(portfolio_value),
                    "total_cost": float(total_cost),
                    "net_return": float(net_return),
                }
            )

        return pd.DataFrame(records).set_index("date")

    def simulate_prediction_market(
        self,
        target_positions: pd.DataFrame,
        prices: pd.DataFrame,
        fee_bps: float | None = None,
    ) -> pd.DataFrame:
        """Simulate prediction-market portfolio with fees and cash accounting."""
        if target_positions.empty or prices.empty:
            return pd.DataFrame()

        fee_bps = fee_bps if fee_bps is not None else self.config.fee_bps
        prices = prices.sort_index()
        target_positions = target_positions.reindex(prices.index).fillna(0.0)
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        cash = self.config.capital
        prev_value = self.config.capital
        records = []

        current_pos = pd.Series(0.0, index=prices.columns)
        for dt in prices.index:
            px = prices.loc[dt].fillna(0.0)
            desired = target_positions.loc[dt].reindex(px.index).fillna(0.0)

            trades = desired - current_pos

            total_cost = 0.0
            for sym, qty in trades.items():
                if qty == 0:
                    continue
                price = px.get(sym, 0.0)
                if price <= 0:
                    continue
                notional = abs(qty) * price
                fee = notional * (fee_bps / 10_000)
                total_cost += fee
                cash -= qty * price

            cash -= total_cost
            new_pos = current_pos + trades
            positions.loc[dt] = new_pos
            current_pos = new_pos

            portfolio_value = cash + (new_pos * px).sum()
            net_return = 0.0 if prev_value == 0 else (portfolio_value - prev_value) / prev_value
            prev_value = portfolio_value

            records.append(
                {
                    "date": dt,
                    "gross_value": float((new_pos * px).sum()),
                    "cash": float(cash),
                    "net_value": float(portfolio_value),
                    "total_cost": float(total_cost),
                    "net_return": float(net_return),
                }
            )

        return pd.DataFrame(records).set_index("date")
