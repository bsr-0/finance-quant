"""Daily risk dashboard for institutional portfolio monitoring.

Aggregates PnL, exposures, inventory, VaR, stress results, drawdowns,
limit utilizations, and key microstructure metrics into a single report.
Suitable for review by risk management and trading desks.

Components:
    - ``DailyRiskDashboard``: Orchestrates all risk checks and produces
      a structured report.
    - ``PreOpenChecklist``: Series of critical checks that must pass
      before trading starts.
    - Machine-readable output (dict/JSON) and optional human-focused
      summary tables.

Design:
    All computations are performed on the data provided at call time
    (no internal state between days).  The dashboard is stateless so
    it can be used for both live monitoring and historical analysis.

Assumptions:
    - Position and PnL data are provided by the caller.
    - VaR and stress calculations use the functions from
      ``pipeline.features.risk_metrics`` and ``pipeline.eval.stress``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the risk dashboard.

    Attributes:
        nav: Net asset value (total capital under management).
        max_daily_loss: Maximum allowable daily loss.
        max_drawdown_pct: Maximum drawdown threshold.
        max_gross_exposure: Maximum gross notional exposure.
        max_net_exposure: Maximum net notional exposure.
        max_leverage: Maximum leverage ratio.
        var_confidence: Confidence level for VaR calculation.
        var_horizon_days: Horizon for VaR in trading days.
        stress_scenarios: List of (name, shock_pct) pairs for stress tests.
    """

    nav: float = 6_000_000_000.0  # $6B
    max_daily_loss: float = 30_000_000.0  # $30M
    max_drawdown_pct: float = 0.03
    max_gross_exposure: float = 12_000_000_000.0  # 2x NAV
    max_net_exposure: float = 3_000_000_000.0
    max_leverage: float = 2.0
    var_confidence: float = 0.99
    var_horizon_days: int = 1
    stress_scenarios: list[tuple[str, float]] = field(
        default_factory=lambda: [
            ("market_down_5pct", -0.05),
            ("market_down_10pct", -0.10),
            ("market_down_20pct", -0.20),
            ("rates_up_100bps", -0.03),
            ("vol_spike_2x", -0.07),
            ("liquidity_crisis", -0.15),
        ]
    )


@dataclass
class LimitUtilization:
    """Current utilization of a risk limit."""

    limit_name: str
    current_value: float
    limit_value: float
    utilization_pct: float
    status: str  # "green", "yellow", "red"

    @property
    def within_limit(self) -> bool:
        return self.status != "red"


@dataclass
class DashboardReport:
    """Complete daily risk dashboard report."""

    date: pd.Timestamp
    nav: float
    pnl_realized: float
    pnl_unrealized: float
    pnl_total: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    num_positions: int
    var_1d: float
    var_10d: float
    max_drawdown_current: float
    limit_utilizations: list[LimitUtilization]
    stress_results: dict[str, float]
    pre_open_checks: list[tuple[str, bool, str]]
    all_checks_passed: bool
    warnings: list[str]


class DailyRiskDashboard:
    """Produce daily risk reports for a $6B institutional portfolio.

    Usage::

        dashboard = DailyRiskDashboard(config)

        report = dashboard.generate_report(
            date=pd.Timestamp.today(),
            positions=positions_df,
            prices=prices_df,
            returns_history=returns_df,
            realized_pnl=daily_rpnl,
        )

        # Machine-readable output
        print(report_to_dict(report))
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        self.config = config or DashboardConfig()

    def generate_report(
        self,
        date: pd.Timestamp,
        positions: dict[str, float],
        prices: dict[str, float],
        returns_history: pd.Series | None = None,
        realized_pnl: float = 0.0,
        peak_nav: float | None = None,
    ) -> DashboardReport:
        """Generate a complete risk dashboard report.

        Args:
            date: Report date.
            positions: Symbol → quantity (signed).
            prices: Symbol → current price.
            returns_history: Historical portfolio returns for VaR.
            realized_pnl: Realized PnL for the current day.
            peak_nav: Peak NAV for drawdown calculation.

        Returns:
            ``DashboardReport`` with all risk metrics.
        """
        cfg = self.config
        warnings: list[str] = []

        # Exposure calculations
        long_notional = sum(
            qty * prices.get(sym, 0)
            for sym, qty in positions.items()
            if qty > 0
        )
        short_notional = sum(
            abs(qty) * prices.get(sym, 0)
            for sym, qty in positions.items()
            if qty < 0
        )
        gross = long_notional + short_notional
        net = abs(long_notional - short_notional)
        leverage = gross / cfg.nav if cfg.nav > 0 else 0

        # Unrealized PnL (simplified — would need cost basis in production)
        unrealized_pnl = 0.0  # Placeholder; real system has cost basis
        total_pnl = realized_pnl + unrealized_pnl

        # VaR
        var_1d = np.nan
        var_10d = np.nan
        if returns_history is not None and len(returns_history) >= 60:
            rets = returns_history.dropna().values
            var_1d = float(np.percentile(rets, (1 - cfg.var_confidence) * 100)) * cfg.nav
            var_10d = var_1d * np.sqrt(cfg.var_horizon_days * 10)

        # Drawdown
        peak = peak_nav or cfg.nav
        dd = (cfg.nav + total_pnl - peak) / peak if peak > 0 else 0

        # Limit utilizations
        limits = self._compute_limit_utilizations(
            gross, net, leverage, total_pnl, dd
        )

        # Stress tests
        stress_results = {}
        for name, shock in cfg.stress_scenarios:
            stressed_pnl = gross * shock
            stress_results[name] = float(stressed_pnl)

        # Pre-open checks
        checks = self._pre_open_checks(positions, prices, returns_history)
        all_passed = all(passed for _, passed, _ in checks)

        # Collect warnings
        for lu in limits:
            if lu.status == "yellow":
                warnings.append(f"WARNING: {lu.limit_name} at {lu.utilization_pct:.0f}%")
            elif lu.status == "red":
                warnings.append(f"BREACH: {lu.limit_name} at {lu.utilization_pct:.0f}%")

        if not all_passed:
            warnings.append("PRE-OPEN CHECKS FAILED — review before trading")

        return DashboardReport(
            date=date,
            nav=cfg.nav,
            pnl_realized=realized_pnl,
            pnl_unrealized=unrealized_pnl,
            pnl_total=total_pnl,
            gross_exposure=gross,
            net_exposure=net,
            leverage=leverage,
            num_positions=sum(1 for v in positions.values() if v != 0),
            var_1d=var_1d,
            var_10d=var_10d,
            max_drawdown_current=dd,
            limit_utilizations=limits,
            stress_results=stress_results,
            pre_open_checks=checks,
            all_checks_passed=all_passed,
            warnings=warnings,
        )

    def _compute_limit_utilizations(
        self,
        gross: float,
        net: float,
        leverage: float,
        pnl: float,
        drawdown: float,
    ) -> list[LimitUtilization]:
        """Compute utilization for each risk limit."""
        cfg = self.config
        limits = []

        def _status(util: float) -> str:
            if util >= 1.0:
                return "red"
            if util >= 0.80:
                return "yellow"
            return "green"

        # Gross exposure
        gross_util = gross / cfg.max_gross_exposure if cfg.max_gross_exposure > 0 else 0
        limits.append(LimitUtilization(
            "gross_exposure", gross, cfg.max_gross_exposure,
            gross_util * 100, _status(gross_util),
        ))

        # Net exposure
        net_util = net / cfg.max_net_exposure if cfg.max_net_exposure > 0 else 0
        limits.append(LimitUtilization(
            "net_exposure", net, cfg.max_net_exposure,
            net_util * 100, _status(net_util),
        ))

        # Leverage
        lev_util = leverage / cfg.max_leverage if cfg.max_leverage > 0 else 0
        limits.append(LimitUtilization(
            "leverage", leverage, cfg.max_leverage,
            lev_util * 100, _status(lev_util),
        ))

        # Daily loss
        loss_util = abs(min(pnl, 0)) / cfg.max_daily_loss if cfg.max_daily_loss > 0 else 0
        limits.append(LimitUtilization(
            "daily_loss", pnl, -cfg.max_daily_loss,
            loss_util * 100, _status(loss_util),
        ))

        # Drawdown
        dd_util = abs(min(drawdown, 0)) / cfg.max_drawdown_pct if cfg.max_drawdown_pct > 0 else 0
        limits.append(LimitUtilization(
            "drawdown", drawdown, -cfg.max_drawdown_pct,
            dd_util * 100, _status(dd_util),
        ))

        return limits

    def _pre_open_checks(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        returns: pd.Series | None,
    ) -> list[tuple[str, bool, str]]:
        """Run pre-open checks."""
        checks: list[tuple[str, bool, str]] = []

        # Data sanity: prices available for all positions
        missing_prices = [
            sym for sym in positions
            if positions[sym] != 0 and sym not in prices
        ]
        checks.append((
            "prices_available",
            len(missing_prices) == 0,
            f"Missing: {missing_prices}" if missing_prices else "OK",
        ))

        # Returns history adequate for risk models
        has_returns = returns is not None and len(returns.dropna()) >= 60
        checks.append((
            "returns_history_adequate",
            has_returns,
            f"Length={len(returns.dropna()) if returns is not None else 0}" if not has_returns else "OK",
        ))

        # Limits configured
        cfg = self.config
        limits_ok = cfg.max_daily_loss > 0 and cfg.max_gross_exposure > 0
        checks.append((
            "limits_configured",
            limits_ok,
            f"max_loss={cfg.max_daily_loss:,.0f}" if limits_ok else "MISSING",
        ))

        # NAV is positive and reasonable
        nav_ok = cfg.nav > 0
        checks.append((
            "nav_positive",
            nav_ok,
            f"NAV={cfg.nav:,.0f}" if nav_ok else "NAV <= 0",
        ))

        return checks


def report_to_dict(report: DashboardReport) -> dict[str, Any]:
    """Convert a dashboard report to a machine-readable dict."""
    return {
        "date": str(report.date),
        "nav": report.nav,
        "pnl": {
            "realized": report.pnl_realized,
            "unrealized": report.pnl_unrealized,
            "total": report.pnl_total,
        },
        "exposure": {
            "gross": report.gross_exposure,
            "net": report.net_exposure,
            "leverage": report.leverage,
            "num_positions": report.num_positions,
        },
        "risk": {
            "var_1d": report.var_1d,
            "var_10d": report.var_10d,
            "max_drawdown": report.max_drawdown_current,
        },
        "limits": [
            {
                "name": lu.limit_name,
                "current": lu.current_value,
                "limit": lu.limit_value,
                "utilization_pct": lu.utilization_pct,
                "status": lu.status,
            }
            for lu in report.limit_utilizations
        ],
        "stress": report.stress_results,
        "pre_open_checks": [
            {"check": name, "passed": passed, "detail": detail}
            for name, passed, detail in report.pre_open_checks
        ],
        "all_checks_passed": report.all_checks_passed,
        "warnings": report.warnings,
    }


def format_dashboard_text(report: DashboardReport) -> str:
    """Format a dashboard report as a human-readable text summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  DAILY RISK DASHBOARD — {report.date}")
    lines.append("=" * 70)
    lines.append(f"  NAV:                 ${report.nav:>18,.0f}")
    lines.append(f"  PnL (Total):         ${report.pnl_total:>18,.0f}")
    lines.append(f"    Realized:          ${report.pnl_realized:>18,.0f}")
    lines.append(f"    Unrealized:        ${report.pnl_unrealized:>18,.0f}")
    lines.append("-" * 70)
    lines.append(f"  Gross Exposure:      ${report.gross_exposure:>18,.0f}")
    lines.append(f"  Net Exposure:        ${report.net_exposure:>18,.0f}")
    lines.append(f"  Leverage:             {report.leverage:>18.2f}x")
    lines.append(f"  Positions:            {report.num_positions:>18d}")
    lines.append("-" * 70)
    lines.append(f"  VaR (1d, 99%):       ${report.var_1d:>18,.0f}")
    lines.append(f"  VaR (10d, 99%):      ${report.var_10d:>18,.0f}")
    lines.append(f"  Max Drawdown:         {report.max_drawdown_current:>18.2%}")
    lines.append("-" * 70)
    lines.append("  LIMIT UTILIZATIONS:")
    for lu in report.limit_utilizations:
        status_marker = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}.get(lu.status, "[??]")
        lines.append(f"    {status_marker} {lu.limit_name:<25} {lu.utilization_pct:6.1f}%")
    lines.append("-" * 70)
    lines.append("  STRESS TESTS:")
    for name, pnl in report.stress_results.items():
        lines.append(f"    {name:<30} ${pnl:>15,.0f}")
    lines.append("-" * 70)
    lines.append("  PRE-OPEN CHECKS:")
    for name, passed, detail in report.pre_open_checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"    [{status}] {name}: {detail}")
    if report.warnings:
        lines.append("-" * 70)
        lines.append("  WARNINGS:")
        for w in report.warnings:
            lines.append(f"    {w}")
    lines.append("=" * 70)
    return "\n".join(lines)
