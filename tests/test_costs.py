"""The cost vector decides every remaining verdict, so its edges must be pinned."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost import costs  # noqa: E402
from cost.costs import (  # noqa: E402
    CRYPTO_FALLBACK_HALF,
    FALLBACK_HALF,
    FLAT_ASSUMPTION,
    MAX_HALF,
    MIN_HALF,
    Cost,
    cost_table,
    half_spread,
)


def test_measured_symbols_are_labelled_measured_and_are_tight():
    c = half_spread("SPY")
    assert c.source == "measured", c
    # SPY quotes a penny or less on a ~$700 tape; anything above 2 bps is wrong
    assert MIN_HALF <= c.half_spread < 0.0002, c
    assert c.half_spread < FLAT_ASSUMPTION, "SPY must not cost the flat assumption"


def test_predicted_symbols_are_labelled_and_bounded():
    tbl = cost_table(["XLB", "EWG", "IYT"])
    for r in tbl.to_dict("records"):
        assert r["source"] in {"measured", "predicted", "fallback"}
        assert MIN_HALF <= r["half_spread"] <= MAX_HALF, r


def test_unknown_symbol_falls_back_and_says_so():
    c = half_spread("NOT_A_TICKER_XYZ")
    assert c.source == "fallback"
    assert c.half_spread == FALLBACK_HALF


def test_crypto_is_out_of_domain_not_extrapolated():
    """The equity ADV fit returns ~0.2 bps for BTC, which is nonsense."""
    for sym in ("BTC", "BTCUSD"):
        c = half_spread(sym)
        assert c.source == "fallback", f"{sym} took an equity prediction"
        assert c.half_spread == CRYPTO_FALLBACK_HALF


def test_stress_is_never_cheaper_than_median():
    for sym in ("SPY", "GLD", "VIXY", "XLB", "NOT_A_TICKER_XYZ"):
        c_med = half_spread(sym)
        c_str = half_spread(sym, stress=True)
        assert c_str.half_spread >= c_med.half_spread - 1e-12, sym


def test_round_trip_scales_with_legs():
    c = Cost("X", 0.0001, "measured")
    assert c.round_trip(legs=1) == pytest.approx(0.0002)
    assert c.round_trip(legs=2) == pytest.approx(0.0004)


def test_date_argument_is_accepted_but_does_not_change_the_answer():
    """Documented behaviour: the sample is a per-symbol level, not a daily series.

    If this ever starts failing, a time-varying table was added and every caller's
    caveat text needs updating with it.
    """
    from datetime import date
    a = half_spread("SPY", date(2020, 3, 16))     # peak-COVID
    b = half_spread("SPY", date(2026, 7, 15))
    assert a.half_spread == b.half_spread


def test_the_headline_claim_holds():
    """Most of the universe is cheaper than the flat 5 bps -- the finding itself."""
    tbl = cost_table(costs.universe())
    assert len(tbl) > 50
    cheaper = (tbl["vs_flat_5bps"] < 1.0).mean()
    assert cheaper > 0.9, f"only {cheaper:.0%} cheaper than flat -- re-check the pull"
    assert tbl["bps"].median() < 2.0


def test_fit_is_reported_with_its_quality():
    m = costs._model()
    fit = m["fit"]
    assert fit is not None, "no liquidity fit -- predictions would be unlabelled guesses"
    assert fit["n"] >= 6
    assert 0.0 <= fit["r2"] <= 1.0
    assert fit["slope"] < 0, "spread must FALL as dollar volume rises"
    assert fit["resid_sd_log"] > 0
