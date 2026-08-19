"""Tests for the static site builder.

Focused on the guarantees that must hold while the signal score is unvalidated:
every rendered page carries the experimental banner, and no page colour-codes
scores as if their quality were established.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("jinja2")

from pipeline.web.static_builder import (
    _compute_ticker_stats,
    _load_validation_summary,
    _score_class,
    build_static_site,
)

PAGES = [
    "index.html",
    "history.html",
    "performance.html",
    "pipeline.html",
    "ticker/XLU.html",
]


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Build a site from one signal CSV and one resolved prediction.

    Chdir's into tmp_path so _load_validation_summary's default relative
    "reports" path can't pick up this repo's real reports/ directory --
    without that isolation this fixture's output would depend on whatever
    validation reports happen to exist on disk when the suite runs.
    """
    monkeypatch.chdir(tmp_path)
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-19",
                "ticker": "XLP",
                "direction": "LONG",
                "score": 85,
                "trend_pts": 40,
                "pullback_pts": 25,
                "volume_pts": 10,
                "volatility_pts": 10,
                "entry_price": 100.0,
                "stop_price": 97.0,
                "target_1": 104.0,
                "target_2": 106.0,
                "atr": 2.0,
                "atr_pct": 2.0,
                "regime": "BULL",
                "confidence": "UNRATED",
                "strategy_id": "QSG-MICRO-SWING-001",
            }
        ]
    ).to_csv(signals_dir / "signals_20260519.csv", index=False)

    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "signal_date": "2026-01-05",
                        "ticker": "XLU",
                        "score": 70,
                        "confidence": "UNRATED",
                        "entry_price": 100.0,
                        "stop_price": 97.0,
                        "target_price": 104.0,
                        "regime": "BULL",
                        "direction": "long",
                        "outcome": "hit_target",
                        "resolved_date": "2026-01-09",
                        "resolved_price": 104.0,
                        "pnl_pct": 4.0,
                        "days_held": 4,
                        "bars_held": 4,
                    }
                ],
                "last_updated": "2026-05-19T00:00:00+00:00",
            }
        )
    )

    out = build_static_site(
        output_dir=tmp_path / "site",
        signals_dir=signals_dir,
        history_path=history,
    )
    return out


@pytest.mark.parametrize("page", PAGES)
def test_every_page_carries_the_validation_banner(site, page):
    html = (site / page).read_text()
    assert "validation-banner" in html
    assert "not validated" in html
    assert "Do not trade these signals." in html


def test_pages_do_not_colour_code_scores(site):
    for page in PAGES:
        html = (site / page).read_text()
        assert (
            "score-high" not in html.split("<style>")[-1].split("</style>")[-1]
        ), f"{page} colour-codes a score as high-quality"


def test_score_class_is_neutral_unless_legacy():
    assert _score_class(95) == "score-neutral"
    assert _score_class(45) == "score-neutral"
    assert _score_class(95, mode="legacy") == "score-high"
    assert _score_class(45, mode="legacy") == "score-low"


def test_ticker_win_rate_counts_profitable_not_just_target_hits():
    """A profitable expiry is a win; it is not a target hit."""
    preds = [
        {"outcome": "expired", "pnl_pct": 2.0, "score": 70},
        {"outcome": "stopped_out", "pnl_pct": -3.0, "score": 65},
    ]
    stats = _compute_ticker_stats(preds)

    assert stats["win_rate"] == 50.0
    assert stats["target_hit_rate"] == 0.0


# --- validation summary loading ----------------------------------------------


def _write_g3_report(reports_dir, verdict="INCONCLUSIVE", n_survivors=0):
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "signal_validation.json").write_text(
        json.dumps(
            {
                "score_result": {
                    "ic_mean": 0.0236,
                    "deflated_sharpe_prob": 0.865,
                    "n_folds": 56,
                },
                "monotonicity": {"daily_ic_ci": [-0.0151, 0.0713]},
                "decomposition": {
                    "trials": list(range(16)),
                    "screened": [(i, i < n_survivors) for i in range(16)],
                    "first_pc_variance_ratio": 0.763,
                },
                "verdict": verdict,
                "reasoning": ["some reason"],
            }
        )
    )


def _write_g5_report(reports_dir, passed=False):
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase5_validation.json").write_text(
        json.dumps(
            {
                "ladder": {"dev_trials": list(range(8)), "pbo": 0.5},
                "g5": {
                    "passed": passed,
                    "criteria": {"oos_beats_baseline": passed},
                    "reasoning": ["some reason"],
                },
                "lightgbm_exploration": list(range(4)),
            }
        )
    )


def test_load_validation_summary_returns_none_when_absent(tmp_path):
    assert _load_validation_summary(tmp_path / "nonexistent") is None


def test_load_validation_summary_parses_g3_and_g5(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_g3_report(reports_dir, verdict="INCONCLUSIVE", n_survivors=0)
    _write_g5_report(reports_dir, passed=False)

    summary = _load_validation_summary(reports_dir)

    assert summary["g3"]["verdict"] == "INCONCLUSIVE"
    assert summary["g3"]["n_trials"] == 16
    assert summary["g3"]["n_bh_survivors"] == 0
    assert summary["g5"]["passed"] is False
    assert summary["g5"]["n_dev_trials"] == 8
    assert summary["g5"]["n_lightgbm_explored"] == 4


def test_load_validation_summary_handles_only_g3(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_g3_report(reports_dir)

    summary = _load_validation_summary(reports_dir)
    assert summary["g3"] is not None
    assert summary["g5"] is None


def test_load_validation_summary_tolerates_corrupt_json(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "signal_validation.json").write_text("{not valid json")

    assert _load_validation_summary(reports_dir) is None


def test_load_validation_summary_picks_newest_phase5_report(tmp_path):
    import time

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "phase5_validation_a.json").write_text(
        json.dumps(
            {
                "ladder": {"dev_trials": [1], "pbo": 0.1},
                "g5": {"passed": False, "criteria": {}, "reasoning": []},
                "lightgbm_exploration": [],
            }
        )
    )
    time.sleep(0.01)
    (reports_dir / "phase5_validation_b.json").write_text(
        json.dumps(
            {
                "ladder": {"dev_trials": [1, 2, 3], "pbo": 0.2},
                "g5": {"passed": False, "criteria": {}, "reasoning": []},
                "lightgbm_exploration": [],
            }
        )
    )

    summary = _load_validation_summary(reports_dir)
    assert summary["g5"]["n_dev_trials"] == 3


def test_performance_page_renders_validation_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_g3_report(tmp_path / "reports")
    _write_g5_report(tmp_path / "reports")

    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"predictions": [], "last_updated": ""}))

    out = build_static_site(
        output_dir=tmp_path / "site", signals_dir=signals_dir, history_path=history
    )
    html = (out / "performance.html").read_text()

    assert "verdict-inconclusive" in html
    assert "verdict-shipnothing" in html
    assert "INCONCLUSIVE" in html
    assert "SHIP NOTHING" in html
    assert "not recommended for live or" in html


def test_performance_page_without_reports_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no reports/ dir here

    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"predictions": [], "last_updated": ""}))

    out = build_static_site(
        output_dir=tmp_path / "site", signals_dir=signals_dir, history_path=history
    )
    html = (out / "performance.html").read_text()

    assert "were not found" in html
