"""Tests for domain-specific checklists (V7 Section 24)."""

from __future__ import annotations

import pytest

from pipeline.domain_checklist import (
    available_domains,
    generate_domain_data_quirks,
    generate_domain_risk_register,
    generate_regulatory_checklist,
)

DOMAINS = ["finance", "sports_betting", "elections", "fantasy_sports"]


class TestDomainChecklists:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_risk_register_populated(self, domain):
        report = generate_domain_risk_register(domain)
        assert report["report_type"] == "domain_specific_risk_register"
        assert report["domain"] == domain
        assert report["total_risks"] >= 5
        assert len(report["entries"]) == report["total_risks"]

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_data_quirks_populated(self, domain):
        report = generate_domain_data_quirks(domain)
        assert report["report_type"] == "domain_data_quirks_checklist"
        assert report["domain"] == domain
        assert report["total_quirks"] >= 5
        assert len(report["entries"]) == report["total_quirks"]

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_regulatory_checklist_populated(self, domain):
        report = generate_regulatory_checklist(domain)
        assert report["report_type"] == "regulatory_compliance_checklist"
        assert report["domain"] == domain
        assert report["total_requirements"] >= 3
        assert len(report["entries"]) == report["total_requirements"]

    def test_unknown_domain_returns_empty(self):
        report = generate_domain_risk_register("unknown")
        assert report["total_risks"] == 0

    def test_available_domains(self):
        domains = available_domains()
        assert set(DOMAINS).issubset(set(domains))

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_risk_entries_have_required_fields(self, domain):
        report = generate_domain_risk_register(domain)
        for entry in report["entries"]:
            assert "risk_id" in entry
            assert "category" in entry
            assert "description" in entry
            assert "severity" in entry
            assert "mitigation" in entry

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_quirk_entries_have_required_fields(self, domain):
        report = generate_domain_data_quirks(domain)
        for entry in report["entries"]:
            assert "quirk_id" in entry
            assert "data_source" in entry
            assert "description" in entry
            assert "impact" in entry
            assert "handling" in entry
