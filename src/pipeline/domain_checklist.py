"""Domain-Specific Checklists (Agent Directive V7 — Section 24.5).

Generates the required domain-specific outputs:
- <domain_specific_risk_register>
- <domain_data_quirks_checklist>
- <regulatory_compliance_checklist>

Pre-populated with Section 24.2 (Financial Markets) content.
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
# Public API
# ---------------------------------------------------------------------------

def generate_domain_risk_register(domain: str = "finance") -> dict[str, Any]:
    """<domain_specific_risk_register> — Section 24.5 required output."""
    if domain == "finance":
        entries = financial_risk_register()
    else:
        entries = []

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
    if domain == "finance":
        entries = financial_data_quirks()
    else:
        entries = []

    return {
        "report_type": "domain_data_quirks_checklist",
        "domain": domain,
        "entries": [asdict(e) for e in entries],
        "total_quirks": len(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_regulatory_checklist(domain: str = "finance") -> dict[str, Any]:
    """<regulatory_compliance_checklist> — Section 24.5 required output."""
    if domain == "finance":
        entries = financial_regulatory_checklist()
    else:
        entries = []

    return {
        "report_type": "regulatory_compliance_checklist",
        "domain": domain,
        "entries": [asdict(e) for e in entries],
        "total_requirements": len(entries),
        "compliant": len([e for e in entries if e.status == "compliant"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
