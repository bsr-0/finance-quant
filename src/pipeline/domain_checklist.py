"""Domain-Specific Checklists (Agent Directive V7 — Section 24.5).

Generates the required domain-specific outputs:
- <domain_specific_risk_register>
- <domain_data_quirks_checklist>
- <regulatory_compliance_checklist>

Pre-populated for four domains per V7 Sections 24.1–24.4:
- Financial Markets (Section 24.2)
- Sports Betting (Section 24.1)
- Elections & Political Forecasting (Section 24.3)
- Fantasy Sports & Contests (Section 24.4)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RiskEntry:
    risk_id: str
    category: str
    description: str
    severity: str  # high, medium, low
    mitigation: str
    status: str = "open"  # open, mitigated, accepted


@dataclass
class QuirkEntry:
    quirk_id: str
    data_source: str
    description: str
    impact: str
    handling: str


@dataclass
class RegulatoryEntry:
    requirement_id: str
    regulation: str
    requirement: str
    implementation: str
    status: str = "compliant"  # compliant, partial, non_compliant


# ---------------------------------------------------------------------------
# Financial Markets domain (Section 24.2)
# ---------------------------------------------------------------------------


def financial_risk_register() -> list[RiskEntry]:
    """Pre-populated risk register for financial markets domain."""
    return [
        RiskEntry(
            risk_id="FIN-R01",
            category="data",
            description="Survivorship bias from delisted stocks missing in historical data",
            severity="high",
            mitigation="Use point-in-time universe from survivorship.py; include delisted stocks",
        ),
        RiskEntry(
            risk_id="FIN-R02",
            category="data",
            description="Look-ahead bias from adjusted prices or revised economic data",
            severity="high",
            mitigation="Use as-of-date joining via feature_asof.py; track revision timestamps",
        ),
        RiskEntry(
            risk_id="FIN-R03",
            category="execution",
            description="Slippage and market impact for larger positions",
            severity="medium",
            mitigation="Square-root impact model in transaction_costs.py; capacity analysis",
        ),
        RiskEntry(
            risk_id="FIN-R04",
            category="regime",
            description="Overfitting to single market regime (e.g., bull market only)",
            severity="high",
            mitigation="Regime-aware evaluation via eval/regime.py; multi-regime backtest",
        ),
        RiskEntry(
            risk_id="FIN-R05",
            category="execution",
            description="Execution latency causing fill price divergence from signal price",
            severity="medium",
            mitigation="Latency estimation via historical/latency.py; conservative fill assumptions",
        ),
        RiskEntry(
            risk_id="FIN-R06",
            category="data",
            description="Corporate actions (splits, dividends, mergers) corrupting price series",
            severity="high",
            mitigation="Corporate action handling in data pipeline; raw vs adjusted tracking",
        ),
        RiskEntry(
            risk_id="FIN-R07",
            category="model",
            description="Transaction costs not included in backtests inflating apparent edge",
            severity="high",
            mitigation="All backtests include transaction cost models; multiple cost scenarios",
        ),
        RiskEntry(
            risk_id="FIN-R08",
            category="regulatory",
            description="Algorithmic trading regulation non-compliance (MiFID II, SEC 15c3-5)",
            severity="high",
            mitigation="Pre-trade risk checks in capital_guard.py; position limits; kill switch",
        ),
    ]


def financial_data_quirks() -> list[QuirkEntry]:
    """Pre-populated data quirks checklist for financial markets."""
    return [
        QuirkEntry(
            quirk_id="FIN-Q01",
            data_source="Economic indicators (GDP, CPI)",
            description="Subject to revisions weeks/months after initial release",
            impact="Using revised data in training creates look-ahead bias",
            handling="Use point-in-time database; track release vs revision timestamps",
        ),
        QuirkEntry(
            quirk_id="FIN-Q02",
            data_source="Earnings data",
            description="Earnings restatements can change historical values retroactively",
            impact="Backtests using restated data overstate signal quality",
            handling="Flag restatements in SEC extractor; use original values for training",
        ),
        QuirkEntry(
            quirk_id="FIN-Q03",
            data_source="Price data",
            description="Adjusted vs unadjusted prices: splits/dividends affect both",
            impact="Mixing adjusted/unadjusted creates leakage and incorrect returns",
            handling="Explicit adjustment tracking; consistent use across pipeline",
        ),
        QuirkEntry(
            quirk_id="FIN-Q04",
            data_source="SEC filings (13F, insider)",
            description="Filing delays of up to 45 days for 13F holdings",
            impact="Using filing data before its availability date causes leakage",
            handling="Use filing_date not period_date as availability timestamp",
        ),
        QuirkEntry(
            quirk_id="FIN-Q05",
            data_source="Market data",
            description="Sub-second granularity with exchange-specific rules and dark pools",
            impact="Execution quality varies by venue; simplified models underestimate costs",
            handling="Venue-aware execution assumptions; conservative fill rates",
        ),
    ]


def financial_regulatory_checklist() -> list[RegulatoryEntry]:
    """Pre-populated regulatory compliance checklist for financial markets."""
    return [
        RegulatoryEntry(
            requirement_id="FIN-REG01",
            regulation="SEC Rule 15c3-5 (Market Access Rule)",
            requirement="Pre-trade risk controls for market access",
            implementation="capital_guard.py pre-trade checks; position limits",
        ),
        RegulatoryEntry(
            requirement_id="FIN-REG02",
            regulation="MiFID II Article 17",
            requirement="Algorithmic trading systems must have risk controls and kill switches",
            implementation="Circuit breaker in infrastructure/; position limits in execution/",
        ),
        RegulatoryEntry(
            requirement_id="FIN-REG03",
            regulation="General (all jurisdictions)",
            requirement="Audit trail for all trading decisions and system changes",
            implementation="Governance audit trail; experiment registry; deployment log",
        ),
        RegulatoryEntry(
            requirement_id="FIN-REG04",
            regulation="General (all jurisdictions)",
            requirement="Position limits and exposure controls",
            implementation="capital_guard.py max_position, max_portfolio_exposure constraints",
        ),
    ]


# ---------------------------------------------------------------------------
# Sports Betting domain (Section 24.1)
# ---------------------------------------------------------------------------


def sports_betting_risk_register() -> list[RiskEntry]:
    """Pre-populated risk register for sports betting domain."""
    return [
        RiskEntry(
            risk_id="SPT-R01",
            category="data",
            description="Injury reports have unreliable publication times and frequent revisions",
            severity="high",
            mitigation="Track injury report timestamps; use publication_time not event_time",
        ),
        RiskEntry(
            risk_id="SPT-R02",
            category="data",
            description="Historical odds data may represent closing lines, not available lines",
            severity="high",
            mitigation="Use timestamped odds snapshots; never backtest with closing lines",
        ),
        RiskEntry(
            risk_id="SPT-R03",
            category="model",
            description="Small sample sizes (e.g., NFL has only 272 regular-season games/year)",
            severity="high",
            mitigation="Use deflated Sharpe; require multi-season validation; report CIs",
        ),
        RiskEntry(
            risk_id="SPT-R04",
            category="execution",
            description="Line movement from sharp action before bet placement",
            severity="medium",
            mitigation="Model execution delay; compare signal-time vs fill-time odds",
        ),
        RiskEntry(
            risk_id="SPT-R05",
            category="model",
            description="Survivorship bias in player statistics (only active players in dataset)",
            severity="medium",
            mitigation="Include historical rosters with inactive/injured players",
        ),
        RiskEntry(
            risk_id="SPT-R06",
            category="model",
            description="Confusing closing line value with genuine predictive edge",
            severity="high",
            mitigation="Separate CLV analysis from actual profit/loss tracking",
        ),
    ]


def sports_betting_data_quirks() -> list[QuirkEntry]:
    """Pre-populated data quirks for sports betting."""
    return [
        QuirkEntry(
            quirk_id="SPT-Q01",
            data_source="Injury reports",
            description="Published at varying times; frequently revised before game time",
            impact="Using final injury status in training creates leakage",
            handling="Use injury report timestamp; only use reports available at prediction time",
        ),
        QuirkEntry(
            quirk_id="SPT-Q02",
            data_source="Odds/lines",
            description="Lines move continuously; closing odds differ from opening/available odds",
            impact="Backtesting with closing lines inflates historical edge",
            handling="Use timestamped odds snapshots at decision time, not closing lines",
        ),
        QuirkEntry(
            quirk_id="SPT-Q03",
            data_source="Player statistics",
            description="Rest days, travel, back-to-back effects are material but easy to miscalculate",
            impact="Ignoring rest/travel features misses significant signal",
            handling="Calculate rest days from schedule; track travel distance and time zones",
        ),
        QuirkEntry(
            quirk_id="SPT-Q04",
            data_source="Lineup confirmations",
            description="May arrive minutes before game time; roster changes close to lock",
            impact="Training with final lineups that were unknown at prediction time",
            handling="Use lineup confirmation timestamps; fall back to projected lineups",
        ),
        QuirkEntry(
            quirk_id="SPT-Q05",
            data_source="Parlays/accumulators",
            description="Correlated outcomes within parlays (e.g., same-game parlays)",
            impact="Treating correlated legs as independent overstates expected value",
            handling="Model outcome correlations explicitly; test parlay EV with simulations",
        ),
    ]


def sports_betting_regulatory_checklist() -> list[RegulatoryEntry]:
    """Pre-populated regulatory checklist for sports betting."""
    return [
        RegulatoryEntry(
            requirement_id="SPT-REG01",
            regulation="Jurisdictional gambling laws",
            requirement="Verify legality in operator's jurisdiction for each bet type",
            implementation="Jurisdiction whitelist; block non-legal markets",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="SPT-REG02",
            regulation="Platform terms of service",
            requirement="Comply with sportsbook ToS; avoid detectable bot patterns",
            implementation="Rate limiting; human-like bet timing; ToS review per platform",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="SPT-REG03",
            regulation="Responsible gambling",
            requirement="Self-imposed loss limits and cooling-off periods",
            implementation="Configurable daily/weekly/monthly loss limits; forced stop-loss",
            status="partial",
        ),
    ]


# ---------------------------------------------------------------------------
# Elections & Political Forecasting domain (Section 24.3)
# ---------------------------------------------------------------------------


def elections_risk_register() -> list[RiskEntry]:
    """Pre-populated risk register for elections/political forecasting."""
    return [
        RiskEntry(
            risk_id="ELC-R01",
            category="data",
            description="Polling data has known biases (mode effects, likely voter screens, herding)",
            severity="high",
            mitigation="Weight polls by historical accuracy; model house effects explicitly",
        ),
        RiskEntry(
            risk_id="ELC-R02",
            category="model",
            description="Extremely small historical sample sizes (few comparable elections)",
            severity="high",
            mitigation="Use hierarchical models; borrow strength across geographies; wide CIs",
        ),
        RiskEntry(
            risk_id="ELC-R03",
            category="model",
            description="Correlated errors across geographies (national swing effects)",
            severity="high",
            mitigation="Model state-level correlations; do not treat state polls as independent",
        ),
        RiskEntry(
            risk_id="ELC-R04",
            category="data",
            description="Non-stationary dynamics from event-driven shocks (debates, scandals)",
            severity="medium",
            mitigation="Use time-varying models; recalibrate after major events",
        ),
        RiskEntry(
            risk_id="ELC-R05",
            category="model",
            description="Overconfident calibration due to small historical sample",
            severity="high",
            mitigation="Widen prediction intervals; use t-distribution not normal for uncertainty",
        ),
    ]


def elections_data_quirks() -> list[QuirkEntry]:
    """Pre-populated data quirks for elections/political forecasting."""
    return [
        QuirkEntry(
            quirk_id="ELC-Q01",
            data_source="Polling data",
            description="Uneven polling frequency; clusters near election day",
            impact="Models may be poorly calibrated during low-polling periods",
            handling="Use Bayesian priors during polling deserts; weight recency appropriately",
        ),
        QuirkEntry(
            quirk_id="ELC-Q02",
            data_source="Polling data",
            description="Different pollsters use different likely voter screens and weighting",
            impact="Treating all polls equally introduces systematic bias",
            handling="Model pollster-specific house effects; weight by historical accuracy",
        ),
        QuirkEntry(
            quirk_id="ELC-Q03",
            data_source="National vs state polls",
            description="National polls mask state-level variation critical for outcomes",
            impact="National-only models miss electoral college dynamics",
            handling="Build state-level models; use national polls only as a prior",
        ),
        QuirkEntry(
            quirk_id="ELC-Q04",
            data_source="Demographic data",
            description="Census and voter file data subject to privacy restrictions",
            impact="May not be usable in certain jurisdictions or for certain analyses",
            handling="Verify data usage rights; anonymize where required; document compliance",
        ),
        QuirkEntry(
            quirk_id="ELC-Q05",
            data_source="Election cycles",
            description="Primary vs general election dynamics differ fundamentally",
            impact="Models trained on general elections may not transfer to primaries",
            handling="Separate models by election phase; do not pool primary and general data",
        ),
    ]


def elections_regulatory_checklist() -> list[RegulatoryEntry]:
    """Pre-populated regulatory checklist for elections forecasting."""
    return [
        RegulatoryEntry(
            requirement_id="ELC-REG01",
            regulation="Data privacy (GDPR, state regulations)",
            requirement="Restrict use of personal voter data per applicable laws",
            implementation="Data classification; consent tracking; anonymization pipeline",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="ELC-REG02",
            regulation="Forecast disclosure",
            requirement="Published forecasts may require methodology disclosure",
            implementation="Methodology documentation; model card generation",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="ELC-REG03",
            regulation="Election integrity",
            requirement="Forecasts must not constitute election interference",
            implementation="Review publication timing; avoid suppression framing",
            status="partial",
        ),
    ]


# ---------------------------------------------------------------------------
# Fantasy Sports & Contests domain (Section 24.4)
# ---------------------------------------------------------------------------


def fantasy_risk_register() -> list[RiskEntry]:
    """Pre-populated risk register for fantasy sports/contests."""
    return [
        RiskEntry(
            risk_id="FAN-R01",
            category="data",
            description="Ownership percentages change up to lock time and vary by contest",
            severity="high",
            mitigation="Use ownership snapshots timestamped before lock; model ownership drift",
        ),
        RiskEntry(
            risk_id="FAN-R02",
            category="model",
            description="Optimizing for projection accuracy instead of contest-specific scoring",
            severity="high",
            mitigation="Optimize for contest objective (cash floor vs GPP ceiling); not raw accuracy",
        ),
        RiskEntry(
            risk_id="FAN-R03",
            category="model",
            description="Ignoring ownership leverage in tournaments (contrarian value)",
            severity="high",
            mitigation="Model ownership-adjusted EV; maximize leverage in GPP contests",
        ),
        RiskEntry(
            risk_id="FAN-R04",
            category="data",
            description="Scoring rules vary by platform and contest type",
            severity="medium",
            mitigation="Configure scoring rules per platform; validate against official results",
        ),
        RiskEntry(
            risk_id="FAN-R05",
            category="model",
            description="Ignoring correlation between players on same team (stacking)",
            severity="medium",
            mitigation="Model team-level correlations; use correlation-aware lineup optimization",
        ),
    ]


def fantasy_data_quirks() -> list[QuirkEntry]:
    """Pre-populated data quirks for fantasy sports."""
    return [
        QuirkEntry(
            quirk_id="FAN-Q01",
            data_source="Player projections",
            description="Projections are contest-specific and may change up to lock time",
            impact="Using final projections in training creates leakage",
            handling="Track projection timestamps; use projections available at decision time",
        ),
        QuirkEntry(
            quirk_id="FAN-Q02",
            data_source="Salary data",
            description="Salaries change between slate publication and lock; vary by platform",
            impact="Training with final salaries that differ from decision-time salaries",
            handling="Snapshot salaries at decision time; track salary movements",
        ),
        QuirkEntry(
            quirk_id="FAN-Q03",
            data_source="Ownership percentages",
            description="Not always publicly available; varies significantly by contest size",
            impact="Ownership models trained on large-field data may not transfer to small fields",
            handling="Segment ownership models by contest size; use projected ownership when actual unavailable",
        ),
        QuirkEntry(
            quirk_id="FAN-Q04",
            data_source="Late-swap rules",
            description="Some platforms allow roster changes after games start",
            impact="Creates information asymmetry; late-swap strategies differ from lock strategies",
            handling="Model late-swap separately; track which contests allow late swap",
        ),
        QuirkEntry(
            quirk_id="FAN-Q05",
            data_source="Historical contest results",
            description="May not be publicly available; payout structures vary",
            impact="Cannot properly backtest contest-specific strategies without results data",
            handling="Simulate contest payouts using assumed field distributions",
        ),
    ]


def fantasy_regulatory_checklist() -> list[RegulatoryEntry]:
    """Pre-populated regulatory checklist for fantasy sports."""
    return [
        RegulatoryEntry(
            requirement_id="FAN-REG01",
            regulation="State-level fantasy sports laws",
            requirement="Verify legality of paid contests in operator's state",
            implementation="State legality whitelist; block entries from restricted states",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="FAN-REG02",
            regulation="Platform terms of service",
            requirement="Comply with platform rules on automated entries and multi-accounting",
            implementation="Single-account enforcement; rate-limited entry submission",
            status="partial",
        ),
        RegulatoryEntry(
            requirement_id="FAN-REG03",
            regulation="Responsible gaming",
            requirement="Self-imposed spending limits and loss tracking",
            implementation="Configurable contest entry budget; cumulative loss tracking",
            status="partial",
        ),
    ]


# ---------------------------------------------------------------------------
# Domain registry (maps domain names to their checklist functions)
# ---------------------------------------------------------------------------

_DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "finance": {
        "risk": financial_risk_register,
        "quirks": financial_data_quirks,
        "regulatory": financial_regulatory_checklist,
    },
    "sports_betting": {
        "risk": sports_betting_risk_register,
        "quirks": sports_betting_data_quirks,
        "regulatory": sports_betting_regulatory_checklist,
    },
    "elections": {
        "risk": elections_risk_register,
        "quirks": elections_data_quirks,
        "regulatory": elections_regulatory_checklist,
    },
    "fantasy_sports": {
        "risk": fantasy_risk_register,
        "quirks": fantasy_data_quirks,
        "regulatory": fantasy_regulatory_checklist,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_domains() -> list[str]:
    """Return the list of supported domain names."""
    return list(_DOMAIN_REGISTRY.keys())


def generate_domain_risk_register(domain: str = "finance") -> dict[str, Any]:
    """<domain_specific_risk_register> — Section 24.5 required output."""
    reg = _DOMAIN_REGISTRY.get(domain)
    entries = reg["risk"]() if reg else []

    return {
        "report_type": "domain_specific_risk_register",
        "domain": domain,
        "entries": [asdict(e) for e in entries],
        "total_risks": len(entries),
        "high_severity": len([e for e in entries if e.severity == "high"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_domain_data_quirks(domain: str = "finance") -> dict[str, Any]:
    """<domain_data_quirks_checklist> — Section 24.5 required output."""
    reg = _DOMAIN_REGISTRY.get(domain)
    entries = reg["quirks"]() if reg else []

    return {
        "report_type": "domain_data_quirks_checklist",
        "domain": domain,
        "entries": [asdict(e) for e in entries],
        "total_quirks": len(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_regulatory_checklist(domain: str = "finance") -> dict[str, Any]:
    """<regulatory_compliance_checklist> — Section 24.5 required output."""
    reg = _DOMAIN_REGISTRY.get(domain)
    entries = reg["regulatory"]() if reg else []

    return {
        "report_type": "regulatory_compliance_checklist",
        "domain": domain,
        "entries": [asdict(e) for e in entries],
        "total_requirements": len(entries),
        "compliant": len([e for e in entries if e.status == "compliant"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
