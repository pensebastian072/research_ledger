"""Both spread estimators must recover a KNOWN spread from simulated data.

A published worked example would only pin one number. Simulating a price process
with a spread we choose ourselves tests the whole range that matters on this box
(0.5 bps for SPY up to ~20 bps for a thin sector ETF) and checks the two things a
cost layer must not do: overstate a tight spread, or return signal when the true
spread is zero.

Construction mirrors what generates real daily bars: an efficient log-price random
walk sampled intraday, with every observed print pushed to the bid or the ask by a
fair coin. The recorded high/low/close are then taken from the observed prints, so
the bid-ask bounce inflates the observed range exactly as it does in the wild.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost.spread_ohlc import (  # noqa: E402
    FLAT_ASSUMPTION,
    abdi_ranaldo,
    corwin_schultz,
    monthly_spreads,
    per_symbol_summary,
)


def simulate(true_spread: float, n_days: int = 2000, ticks_per_day: int = 90,
             daily_vol: float = 0.012, seed: int = 4) -> pd.DataFrame:
    """Daily OHLC from an efficient walk plus a bid-ask bounce of `true_spread`."""
    rng = np.random.default_rng(seed)
    step = daily_vol / np.sqrt(ticks_per_day)
    n = n_days * ticks_per_day
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * step))
    side = rng.choice([-1.0, 1.0], size=n)          # buy or sell at the touch
    obs = mid * (1.0 + side * true_spread / 2.0)
    obs = obs.reshape(n_days, ticks_per_day)
    return pd.DataFrame(
        {"high": obs.max(axis=1), "low": obs.min(axis=1), "close": obs[:, -1]},
        index=pd.bdate_range("2010-01-01", periods=n_days))


@pytest.mark.parametrize("true_spread", [0.0100, 0.0200])
def test_both_estimators_recover_a_WIDE_known_spread(true_spread):
    """Where these estimators were designed to work -- single equities, 100-200 bps."""
    df = simulate(true_spread)
    for name, est in (("corwin_schultz", corwin_schultz(df)),
                      ("abdi_ranaldo", abdi_ranaldo(df))):
        assert est is not None, name
        assert 0.5 * true_spread < est < 1.5 * true_spread, (
            f"{name}: estimated {est:.6f} for a true spread of {true_spread:.6f}")


@pytest.mark.parametrize("true_spread", [0.0002, 0.0005, 0.0010])
def test_estimators_return_ZERO_at_etf_scale(true_spread):
    """The finding that killed the free-cost plan -- pin it so nobody retries it.

    At the ~1.2% daily vol these ETFs run, a real 0.5-10 bps spread is buried in the
    high-low range: alpha goes negative, the paper's truncation fires, and the answer
    is 0.0 -- not a small number. A cost layer reporting zero cost is worse than the
    flat 5 bps guess it was meant to replace.
    """
    df = simulate(true_spread, daily_vol=0.012)
    assert corwin_schultz(df) == 0.0, "CS now resolves ETF-scale spreads -- re-verify"
    assert abdi_ranaldo(df) == 0.0, "AR now resolves ETF-scale spreads -- re-verify"


def test_resolution_floor_is_reported_and_above_etf_spreads():
    from cost.spread_ohlc import ETF_VOL_TYPICAL, calibrate_floor
    table = [r for r in calibrate_floor(vols=(ETF_VOL_TYPICAL,))
             if r["daily_vol"] == ETF_VOL_TYPICAL]
    resolved = [r["true_spread"] for r in table if r["resolved"]]
    assert resolved, "nothing resolved at all -- estimator is broken, not just blind"
    # the floor must sit far above the 0.5-20 bps range these ETFs actually quote
    assert min(resolved) >= 0.002, min(resolved)


def test_estimators_are_monotonic_in_the_true_spread():
    """The cost layer's whole job is ranking symbols by cost -- so ordering must hold."""
    spreads = [0.0002, 0.0010, 0.0040, 0.0200]
    cs = [corwin_schultz(simulate(s)) for s in spreads]
    ar = [abdi_ranaldo(simulate(s)) for s in spreads]
    assert cs == sorted(cs), cs
    assert ar == sorted(ar), ar


def test_zero_true_spread_does_not_manufacture_cost():
    """With no bounce, both must land near zero -- not at the 5 bps default."""
    df = simulate(0.0)
    cs, ar = corwin_schultz(df), abdi_ranaldo(df)
    assert cs < FLAT_ASSUMPTION, f"corwin_schultz invented {cs:.6f} from no spread"
    assert ar < FLAT_ASSUMPTION, f"abdi_ranaldo invented {ar:.6f} from no spread"


def test_volatility_alone_is_not_read_as_spread():
    """Ten times the volatility, same zero spread: neither may confuse the two."""
    calm = simulate(0.0, daily_vol=0.006, seed=8)
    wild = simulate(0.0, daily_vol=0.060, seed=8)
    for f in (corwin_schultz, abdi_ranaldo):
        assert f(wild) < 10 * max(f(calm), 1e-5), (
            f"{f.__name__} scales with volatility, not spread")


def test_overnight_gap_adjustment_matters():
    """A pure overnight jump must not be priced as spread."""
    df = simulate(0.0005, n_days=600)
    jumped = df.copy()
    jumped.iloc[300:] *= 1.15                      # a clean 15% gap, no spread change
    with_adj = corwin_schultz(jumped, adjust_overnight=True)
    without = corwin_schultz(jumped, adjust_overnight=False)
    assert with_adj <= without + 1e-9
    assert with_adj < 4 * 0.0005


def test_returns_none_on_unusable_input():
    tiny = pd.DataFrame({"high": [1.0, 1.1], "low": [0.9, 1.0], "close": [1.0, 1.05]})
    assert corwin_schultz(tiny) is None
    assert abdi_ranaldo(tiny) is None
    bad = pd.DataFrame({"high": [np.nan] * 5, "low": [np.nan] * 5,
                        "close": [np.nan] * 5})
    assert corwin_schultz(bad) is None
    assert abdi_ranaldo(bad) is None


def test_negatives_are_clipped_not_propagated():
    """Small noisy samples can drive either estimator negative; a cost must be >= 0."""
    rng = np.random.default_rng(2)
    for seed in range(12):
        n = 25
        c = 100 + rng.standard_normal(n)
        df = pd.DataFrame({"high": c + 0.01, "low": c - 0.01, "close": c},
                          index=pd.bdate_range("2020-01-01", periods=n))
        for f in (corwin_schultz, abdi_ranaldo):
            v = f(df)
            assert v is None or v >= 0.0


def test_monthly_panel_and_summary_shape():
    ohlc = {"TIGHT": simulate(0.0002, n_days=500),
            "WIDE": simulate(0.0100, n_days=500, seed=9)}
    monthly = monthly_spreads(ohlc)
    assert set(monthly["symbol"]) == {"TIGHT", "WIDE"}
    assert (monthly["n_days"] >= 12).all()
    s = per_symbol_summary(monthly)
    assert s.loc["TIGHT", "blend_half"] < s.loc["WIDE", "blend_half"]
    # the ratio column is the finding: TIGHT must read as cheaper than the flat 5 bps
    assert s.loc["TIGHT", "vs_flat_5bps"] < 1.0
    assert s.loc["WIDE", "vs_flat_5bps"] > 1.0
    assert (s["disagreement"] >= 0).all()
