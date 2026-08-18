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

from pipeline.web.static_builder import _compute_ticker_stats, _score_class, build_static_site

PAGES = [
    "index.html",
    "history.html",
    "performance.html",
    "pipeline.html",
    "ticker/XLU.html",
]


@pytest.fixture
def site(tmp_path):
    """Build a site from one signal CSV and one resolved prediction."""
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
