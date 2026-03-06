"""Notification system for critical trading events.

Supports multiple channels:
- Console (always on — uses Python logging)
- Slack (via incoming webhook)
- Email (via SMTP)

Usage::

    from pipeline.infrastructure.notifier import notify, AlertSeverity

    # Fire-and-forget — channels configured via settings / env vars
    notify(
        severity=AlertSeverity.CRITICAL,
        title="RED Circuit Breaker",
        message="Drawdown hit 15.2%. All positions closed.",
        context={"equity": 254.50, "peak": 300.00},
    )

The module exposes a singleton ``Notifier`` that is lazily configured on
first use.  All public helpers (``notify``, ``get_notifier``) are safe to
call from any module without circular-import risk because this module has
no dependencies on strategy/execution code.
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from enum import IntEnum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class AlertSeverity(IntEnum):
    """Severity levels for notifications.

    Higher values are more critical.  Channels can filter by minimum
    severity so that, for example, Slack only receives WARNING+ while
    email only receives CRITICAL.
    """

    INFO = 0
    WARNING = 1
    CRITICAL = 2


_SEVERITY_EMOJI = {
    AlertSeverity.INFO: "ℹ️",
    AlertSeverity.WARNING: "⚠️",
    AlertSeverity.CRITICAL: "🚨",
}

_SEVERITY_LABEL = {
    AlertSeverity.INFO: "INFO",
    AlertSeverity.WARNING: "WARNING",
    AlertSeverity.CRITICAL: "CRITICAL",
}


# ---------------------------------------------------------------------------
# Channel configs (plain dataclasses — no pydantic dependency here)
# ---------------------------------------------------------------------------

@dataclass
class SlackConfig:
    webhook_url: str
    min_severity: AlertSeverity = AlertSeverity.WARNING
    timeout_seconds: float = 10.0


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)
    min_severity: AlertSeverity = AlertSeverity.CRITICAL
    timeout_seconds: float = 15.0


@dataclass
class ConsoleConfig:
    """Console channel (Python logger).  Always enabled."""

    min_severity: AlertSeverity = AlertSeverity.INFO


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------

class Notifier:
    """Multi-channel notification dispatcher.

    Thread-safe: sends are protected by a lock so concurrent calls from
    the position monitor or realtime feed threads don't interleave.
    """

    def __init__(
        self,
        slack: SlackConfig | None = None,
        email: EmailConfig | None = None,
        console: ConsoleConfig | None = None,
        enabled: bool = True,
    ) -> None:
        self.slack = slack
        self.email = email
        self.console = console or ConsoleConfig()
        self.enabled = enabled
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []

    # -- public API --------------------------------------------------------

    def send(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Dispatch a notification to all configured channels.

        Delivery is best-effort: channel failures are logged but never
        propagated so callers don't need try/except.
        """
        if not self.enabled:
            return

        now = datetime.now(timezone.utc)
        event = {
            "timestamp": now.isoformat(),
            "severity": _SEVERITY_LABEL[severity],
            "title": title,
            "message": message,
            "context": context or {},
        }

        with self._lock:
            self._history.append(event)

            # Console (always)
            if severity >= self.console.min_severity:
                self._send_console(severity, title, message, context)

            # Slack
            if self.slack and severity >= self.slack.min_severity:
                self._send_slack(severity, title, message, context, now)

            # Email
            if self.email and severity >= self.email.min_severity:
                self._send_email(severity, title, message, context, now)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Recent notification history (for testing / dashboards)."""
        return list(self._history)

    # -- channel implementations -------------------------------------------

    def _send_console(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        context: dict[str, Any] | None,
    ) -> None:
        label = _SEVERITY_LABEL[severity]
        ctx_str = f" | {context}" if context else ""
        log_msg = f"[ALERT:{label}] {title} — {message}{ctx_str}"

        if severity >= AlertSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity >= AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _send_slack(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        context: dict[str, Any] | None,
        ts: datetime,
    ) -> None:
        assert self.slack is not None
        emoji = _SEVERITY_EMOJI.get(severity, "")
        ctx_lines = ""
        if context:
            ctx_lines = "\n".join(f"• *{k}*: {v}" for k, v in context.items())
            ctx_lines = f"\n{ctx_lines}"

        payload = {
            "text": (
                f"{emoji} *[{_SEVERITY_LABEL[severity]}] {title}*\n"
                f"{message}{ctx_lines}\n"
                f"_<{ts.strftime('%Y-%m-%d %H:%M:%S UTC')}>_"
            ),
        }

        try:
            resp = httpx.post(
                self.slack.webhook_url,
                json=payload,
                timeout=self.slack.timeout_seconds,
            )
            if resp.status_code != 200:
                logger.error(
                    "Slack webhook returned %d: %s", resp.status_code, resp.text,
                )
        except Exception:
            logger.exception("Failed to send Slack notification")

    def _send_email(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        context: dict[str, Any] | None,
        ts: datetime,
    ) -> None:
        assert self.email is not None
        if not self.email.to_addrs:
            return

        label = _SEVERITY_LABEL[severity]
        ctx_text = ""
        if context:
            ctx_text = "\n\nContext:\n" + "\n".join(
                f"  {k}: {v}" for k, v in context.items()
            )

        msg = EmailMessage()
        msg["Subject"] = f"[{label}] {title}"
        msg["From"] = self.email.from_addr or self.email.smtp_user
        msg["To"] = ", ".join(self.email.to_addrs)
        msg.set_content(
            f"{label}: {title}\n\n"
            f"{message}{ctx_text}\n\n"
            f"Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        try:
            if self.email.use_tls:
                server = smtplib.SMTP(
                    self.email.smtp_host,
                    self.email.smtp_port,
                    timeout=self.email.timeout_seconds,
                )
                server.starttls()
            else:
                server = smtplib.SMTP(
                    self.email.smtp_host,
                    self.email.smtp_port,
                    timeout=self.email.timeout_seconds,
                )

            if self.email.smtp_user:
                server.login(self.email.smtp_user, self.email.smtp_password)

            server.send_message(msg)
            server.quit()
        except Exception:
            logger.exception("Failed to send email notification")


# ---------------------------------------------------------------------------
# Singleton / global access
# ---------------------------------------------------------------------------

_notifier: Optional[Notifier] = None
_notifier_lock = threading.Lock()


def get_notifier() -> Notifier:
    """Return the global notifier, creating it lazily from settings."""
    global _notifier
    if _notifier is not None:
        return _notifier

    with _notifier_lock:
        if _notifier is not None:
            return _notifier
        _notifier = _build_notifier_from_settings()
        return _notifier


def set_notifier(notifier: Notifier) -> None:
    """Override the global notifier (useful for testing)."""
    global _notifier
    _notifier = notifier


def reset_notifier() -> None:
    """Reset the global notifier (useful for testing)."""
    global _notifier
    _notifier = None


def notify(
    severity: AlertSeverity,
    title: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Convenience: send a notification via the global notifier."""
    get_notifier().send(severity, title, message, context)


# ---------------------------------------------------------------------------
# Build from settings
# ---------------------------------------------------------------------------

def _build_notifier_from_settings() -> Notifier:
    """Construct a Notifier from PipelineSettings.

    Import is deferred to avoid circular dependencies (settings →
    notifier → settings).
    """
    try:
        from pipeline.settings import get_settings

        settings = get_settings()
        ns = getattr(settings, "notifications", None)

        if ns is None:
            return Notifier(enabled=True)

        enabled = getattr(ns, "enabled", True)

        # Slack
        slack_cfg = None
        slack_url = getattr(ns, "slack_webhook_url", "") or ""
        if slack_url:
            slack_cfg = SlackConfig(
                webhook_url=slack_url,
                min_severity=AlertSeverity[
                    getattr(ns, "slack_min_severity", "WARNING").upper()
                ],
            )

        # Email
        email_cfg = None
        smtp_host = getattr(ns, "smtp_host", "") or ""
        if smtp_host:
            to_raw = getattr(ns, "email_to", [])
            to_addrs = to_raw if isinstance(to_raw, list) else [to_raw]
            email_cfg = EmailConfig(
                smtp_host=smtp_host,
                smtp_port=getattr(ns, "smtp_port", 587),
                smtp_user=getattr(ns, "smtp_user", ""),
                smtp_password=getattr(ns, "smtp_password", ""),
                use_tls=getattr(ns, "smtp_use_tls", True),
                from_addr=getattr(ns, "email_from", ""),
                to_addrs=to_addrs,
                min_severity=AlertSeverity[
                    getattr(ns, "email_min_severity", "CRITICAL").upper()
                ],
            )

        # Console
        console_cfg = ConsoleConfig(
            min_severity=AlertSeverity[
                getattr(ns, "console_min_severity", "INFO").upper()
            ],
        )

        return Notifier(
            slack=slack_cfg,
            email=email_cfg,
            console=console_cfg,
            enabled=enabled,
        )

    except Exception:
        logger.exception("Failed to build notifier from settings; using defaults")
        return Notifier(enabled=True)
