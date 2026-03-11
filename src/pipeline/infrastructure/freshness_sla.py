"""Data Freshness SLA monitoring (Agent Directive V7 Section 19.3).

Every data source must have a documented freshness SLA that specifies
how stale the data can be before the system takes corrective action.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DataCategory(str, Enum):
    """Data freshness categories per Section 19.3."""

    REALTIME = "realtime"  # seconds to minutes
    DAILY_BATCH = "daily"  # hours
    SLOW_REFERENCE = "slow"  # days to weeks
    HISTORICAL = "historical"  # static, validate checksums


class StalenessAction(str, Enum):
    """What to do when data goes stale."""

    SUSPEND = "suspend_decisions"
    SERVE_STALE = "serve_stale_with_flag"
    ALERT_ONLY = "alert_only"
    INVESTIGATE = "investigate_integrity"


@dataclass
class FreshnessSLA:
    """SLA definition for a single data source."""

    source_name: str
    category: DataCategory
    max_staleness_seconds: int
    staleness_action: StalenessAction
    description: str = ""


@dataclass
class FreshnessViolation:
    """A detected freshness violation."""

    source_name: str
    category: str
    staleness_seconds: float
    max_allowed_seconds: int
    action: str
    detected_at: str = ""


class FreshnessMonitor:
    """Monitor data freshness against configured SLAs.

    Usage::

        monitor = FreshnessMonitor()
        monitor.check("polymarket_odds", last_updated)
        violations = monitor.get_violations()
    """

    def __init__(self, slas: list[FreshnessSLA] | None = None) -> None:
        self._slas: dict[str, FreshnessSLA] = {}
        self._violations: list[FreshnessViolation] = []
        for sla in slas or default_slas():
            self.register(sla)

    def register(self, sla: FreshnessSLA) -> None:
        self._slas[sla.source_name] = sla

    def check(
        self,
        source_name: str,
        last_updated: datetime,
        now: datetime | None = None,
    ) -> FreshnessViolation | None:
        """Check a source against its SLA.

        Returns a ``FreshnessViolation`` if the source is stale, else None.
        """
        if source_name not in self._slas:
            logger.warning("No SLA registered for source '%s'", source_name)
            return None

        sla = self._slas[source_name]
        now = now or datetime.now(UTC)
        staleness = (now - last_updated).total_seconds()

        if staleness > sla.max_staleness_seconds:
            violation = FreshnessViolation(
                source_name=source_name,
                category=sla.category.value,
                staleness_seconds=staleness,
                max_allowed_seconds=sla.max_staleness_seconds,
                action=sla.staleness_action.value,
                detected_at=now.isoformat(),
            )
            self._violations.append(violation)
            logger.warning(
                "Freshness SLA violation: %s is %.0fs stale (max %ds) → %s",
                source_name,
                staleness,
                sla.max_staleness_seconds,
                sla.staleness_action.value,
            )
            return violation
        return None

    def check_all(self, last_updated_map: dict[str, datetime]) -> list[FreshnessViolation]:
        """Check all sources in a single call."""
        violations: list[FreshnessViolation] = []
        for source_name, last_updated in last_updated_map.items():
            v = self.check(source_name, last_updated)
            if v:
                violations.append(v)
        return violations

    def get_violations(self) -> list[FreshnessViolation]:
        return list(self._violations)

    def clear_violations(self) -> None:
        self._violations.clear()

    def export_sla_registry(self) -> dict[str, Any]:
        """<freshness_sla_registry> — Section 19.4 required output."""
        return {
            "report_type": "freshness_sla_registry",
            "slas": [asdict(s) for s in self._slas.values()],
            "total_sources": len(self._slas),
            "violations_recorded": len(self._violations),
            "generated_at": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Default SLAs for known data sources
# ---------------------------------------------------------------------------


def default_slas() -> list[FreshnessSLA]:
    """Pre-populated SLAs matching the repo's data extractors."""
    return [
        FreshnessSLA(
            source_name="polymarket_odds",
            category=DataCategory.REALTIME,
            max_staleness_seconds=600,  # 10 minutes
            staleness_action=StalenessAction.SUSPEND,
            description="Polymarket orderbook snapshots",
        ),
        FreshnessSLA(
            source_name="prices_daily",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=86400,  # 24 hours
            staleness_action=StalenessAction.SERVE_STALE,
            description="Daily OHLCV prices",
        ),
        FreshnessSLA(
            source_name="fred_macro",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=172800,  # 48 hours
            staleness_action=StalenessAction.SERVE_STALE,
            description="FRED economic indicators",
        ),
        FreshnessSLA(
            source_name="gdelt_news",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=86400,
            staleness_action=StalenessAction.SERVE_STALE,
            description="GDELT news/event data",
        ),
        FreshnessSLA(
            source_name="sec_fundamentals",
            category=DataCategory.SLOW_REFERENCE,
            max_staleness_seconds=604800,  # 7 days
            staleness_action=StalenessAction.ALERT_ONLY,
            description="SEC fundamental filings",
        ),
        FreshnessSLA(
            source_name="sec_insider",
            category=DataCategory.SLOW_REFERENCE,
            max_staleness_seconds=604800,
            staleness_action=StalenessAction.ALERT_ONLY,
            description="SEC insider transactions",
        ),
        FreshnessSLA(
            source_name="reddit_sentiment",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=86400,
            staleness_action=StalenessAction.SERVE_STALE,
            description="Reddit sentiment scores",
        ),
        FreshnessSLA(
            source_name="short_interest",
            category=DataCategory.SLOW_REFERENCE,
            max_staleness_seconds=1209600,  # 14 days
            staleness_action=StalenessAction.ALERT_ONLY,
            description="Short interest data",
        ),
        FreshnessSLA(
            source_name="etf_flows",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=172800,
            staleness_action=StalenessAction.SERVE_STALE,
            description="ETF fund flow data",
        ),
        FreshnessSLA(
            source_name="fama_french_factors",
            category=DataCategory.SLOW_REFERENCE,
            max_staleness_seconds=2592000,  # 30 days
            staleness_action=StalenessAction.ALERT_ONLY,
            description="Fama-French factor returns",
        ),
    ]
