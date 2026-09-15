"""The measurement engine must find a planted signal and reject noise.

If these fail, every verdict in ANSWERS.md is worthless -- so they check the two
directions that matter: a known edge is recovered, and pure noise is not mistaken
for one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graders.lean_ic import (  # noqa: E402
    forward_returns,
    ic_table,
    newey_west_t,
    summarise,
)

ASSETS = [f"A{i}" for i in range(12)]
N_DAYS = 900


def _prices_and_scores(signal_strength: float, seed: int = 5):
    """Random-walk prices plus scores correlated with the NEXT 5-day return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=N_DAYS)
    rets = rng.standard_normal((N_DAYS, len(ASSETS))) * 0.01
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                          index=dates, columns=ASSETS)
    fwd = forward_returns(prices, 5)
    noise = rng.standard_normal(fwd.shape)
    score = signal_strength * fwd.to_numpy() / 0.02 + noise
    scores = pd.DataFrame(score, index=dates, columns=ASSETS)
    long = (scores.stack(future_stack=True).rename("score").reset_index())
    long.columns = ["date", "asset", "score"]
    return prices, long.dropna()


def test_recovers_a_planted_signal():
    prices, scores = _prices_and_scores(signal_strength=1.0)
    rows = ic_table(scores, prices, horizons=(5,), by_year=False)
    r = rows[0]
    assert r["rank_ic"] > 0.15, r
    assert r["t_nw"] > 4, r
    assert r["hit_rate"] > r["base_rate"], r
    assert r["ls_spread_gross"] > 0, r


def test_rejects_pure_noise():
    prices, scores = _prices_and_scores(signal_strength=0.0, seed=99)
    rows = ic_table(scores, prices, horizons=(5,), by_year=False)
    r = rows[0]
    assert abs(r["rank_ic"]) < 0.03, r
    assert abs(r["t_nw"]) < 2.0, r


def test_reports_clusters_not_just_n():
    """Same-date rows are one draw. n must not be quotable as independent."""
    prices, scores = _prices_and_scores(signal_strength=0.5)
    rows = ic_table(scores, prices, horizons=(5,), by_year=False)
    r = rows[0]
    assert r["n"] > r["n_clusters"] * 5      # 12 assets per date
    assert r["n_clusters"] <= N_DAYS


def test_flags_a_low_cluster_count():
    """Three dates, many assets -> the note must say so in words."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=40)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(
        rng.standard_normal((40, len(ASSETS))) * 0.01, axis=0)),
        index=dates, columns=ASSETS)
    rows_in = []
    for d in dates[:3]:
        for a in ASSETS:
            rows_in.append({"date": d, "asset": a, "score": rng.standard_normal()})
    out = ic_table(pd.DataFrame(rows_in), prices, horizons=(5,), by_year=False)
    note = " ".join(out[0]["notes"])
    assert "distinct dates" in note
    assert out[0]["n_clusters"] == 3


def test_newey_west_widens_the_error_on_autocorrelated_input():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(600) * 0.01
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = 0.8 * x[i - 1] + e[i]        # strongly persistent
    x += 0.002
    t_ols = newey_west_t(x, lags=0)
    t_nw = newey_west_t(x, lags=20)
    assert abs(t_nw) < abs(t_ols)


def test_newey_west_handles_degenerate_input():
    assert newey_west_t(np.array([1.0, 2.0]), lags=1) is None
    assert newey_west_t(np.zeros(50), lags=4) is None


def test_forward_returns_do_not_peek():
    prices = pd.DataFrame({"X": [10.0, 11.0, 12.0, 13.0, 14.0]},
                          index=pd.bdate_range("2020-01-01", periods=5))
    fwd = forward_returns(prices, 2)
    assert fwd["X"].iloc[0] == pytest.approx(12.0 / 10.0 - 1)
    assert np.isnan(fwd["X"].iloc[-1])      # no future to look at
    assert np.isnan(fwd["X"].iloc[-2])


def test_summarise_counts_years_honestly():
    prices, scores = _prices_and_scores(signal_strength=1.0)
    rows = ic_table(scores, prices, horizons=(5,), by_year=True)
    s = summarise(rows, 5)
    assert s["years_graded"] >= 3
    assert s["years_ic_positive"] <= s["years_graded"]
    assert s["years_ic_pos_and_t_gt_2"] <= s["years_ic_positive"]


def test_missing_score_column_is_an_error_not_a_silent_zero():
    prices, scores = _prices_and_scores(signal_strength=0.0)
    with pytest.raises(ValueError):
        ic_table(scores.rename(columns={"score": "lean"}), prices, horizons=(5,))
