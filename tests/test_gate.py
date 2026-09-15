"""The corrected DSR must be right, and the inversion must be exact."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate import (  # noqa: E402
    deflated_sharpe,
    deflated_sharpe_fixed,
    expected_max_sr_units,
    legacy_deflated_sharpe_ratio,
    recompute_from_summary,
)


def test_bracket_is_dimensionless_and_grows_with_trials():
    assert expected_max_sr_units(1) == 0.0
    vals = [expected_max_sr_units(n) for n in (2, 8, 42, 132)]
    assert all(0 < v < 4 for v in vals)
    assert vals == sorted(vals)


def test_single_trial_matches_the_canonical_implementation():
    """With n_trials=1 there is no deflation term, so both must agree exactly."""
    rng = np.random.default_rng(7)
    pnl = rng.standard_normal(300) * 0.01 + 0.0004
    old = deflated_sharpe(pnl, n_trials=1)
    new = deflated_sharpe_fixed(pnl, n_trials=1)
    assert old["ratio"] == pytest.approx(new["ratio"], abs=1e-9)


def test_fixed_is_calibrated_on_pure_noise():
    """Best-of-N noise must average ~0.5. This is the whole definition of the fix."""
    rng = np.random.default_rng(11)
    for n_trials in (2, 8):
        probs = []
        for _ in range(400):
            trials = rng.standard_normal((n_trials, 210)) * 0.01
            srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
            best = trials[int(np.argmax(srs))]
            probs.append(deflated_sharpe_fixed(best, n_trials=n_trials)["prob"])
        assert abs(float(np.mean(probs)) - 0.5) < 0.10, f"n_trials={n_trials}"


def test_upstream_is_now_calibrated_on_pure_noise():
    """Upstream was PATCHED on 2026-07-30, so the canonical function must pass too.

    This replaces test_shipped_is_not_calibrated_on_pure_noise, which existed only to
    pin the live defect. If this ever fails, macro_gpu_lab/validate.py regressed.
    """
    rng = np.random.default_rng(13)
    probs = []
    for _ in range(300):
        trials = rng.standard_normal((8, 210)) * 0.01
        srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
        best = trials[int(np.argmax(srs))]
        probs.append(deflated_sharpe(best, n_trials=8)["prob"])
    assert abs(float(np.mean(probs)) - 0.5) < 0.10, float(np.mean(probs))


def test_fixed_has_power_against_a_real_edge():
    rng = np.random.default_rng(17)
    passes = []
    for _ in range(300):
        trials = rng.standard_normal((8, 210)) * 0.01
        trials[0] += 0.30 * 0.01
        srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
        best = trials[int(np.argmax(srs))]
        passes.append(deflated_sharpe_fixed(best, n_trials=8)["ratio"] > 0)
    assert float(np.mean(passes)) > 0.90


def test_cross_mode_needs_trial_srs():
    pnl = np.random.default_rng(3).standard_normal(100) * 0.01
    with pytest.raises(ValueError):
        deflated_sharpe_fixed(pnl, n_trials=8, sigma_sr="cross")
    ok = deflated_sharpe_fixed(pnl, n_trials=8, sigma_sr="cross",
                               trial_srs=[0.01, -0.02, 0.03, 0.0, 0.015, -0.01,
                                          0.02, -0.005])
    assert ok["sigma_sr_mode"] == "cross"
    assert ok["sr_star"] > 0


def test_recompute_from_summary_round_trips_exactly():
    """Invert a shipped scorecard and land on the directly-computed fixed ratio.

    This is the test that licenses correcting 163 stored rows without re-running a
    single backtest: if the inversion is exact on synthetic series, it is exact on
    the committed ones.
    """
    rng = np.random.default_rng(23)
    for n_trials in (2, 8, 42):
        for size in (60, 210, 413):
            pnl = rng.standard_normal(size) * 0.01 + 0.0006
            # upstream is fixed now, so the LEGACY formula stands in for what actually
            # wrote the historical scorecards this helper is meant to correct
            legacy_ratio = round(legacy_deflated_sharpe_ratio(pnl, n_trials), 4)
            sr = round(deflated_sharpe(pnl, n_trials=n_trials)["sr"], 4)
            new = deflated_sharpe_fixed(pnl, n_trials=n_trials)
            inv = recompute_from_summary(sr, legacy_ratio, size, n_trials)
            assert inv is not None
            # stored sr/ratio are rounded to 4dp, so allow that much slack
            assert inv["fixed_ratio"] == pytest.approx(new["ratio"], abs=5e-3), (
                f"n_trials={n_trials} size={size}")


def test_recompute_returns_none_when_undefined():
    assert recompute_from_summary(0.1, -5.0, 100, 1) is None      # no deflation term
    assert recompute_from_summary(None, -5.0, 100, 8) is None
    assert recompute_from_summary(0.1, 0.0, 100, 8) is None       # division by zero
    assert recompute_from_summary(0.1, -5.0, 1, 8) is None


def test_penalty_inflation_is_about_sqrt_T():
    """The defect's signature: the penalty is scaled by ~sqrt(T-1)."""
    rng = np.random.default_rng(29)
    pnl = rng.standard_normal(210) * 0.01 + 0.0005
    sr = round(deflated_sharpe(pnl, n_trials=8)["sr"], 4)
    legacy_ratio = round(legacy_deflated_sharpe_ratio(pnl, 8), 4)
    inv = recompute_from_summary(sr, legacy_ratio, 210, 8)
    assert inv["penalty_inflation"] == pytest.approx(math.sqrt(209), rel=0.15)


def test_legacy_reference_reproduces_a_real_stored_scorecard():
    """Anchor the legacy formula to a number actually on disk.

    macro surprise_shift :: short_spy_overlay 5d recorded sr 0.1281, ratio -18.3675
    at n=210, n_trials=8. If the legacy reference is faithful, inverting that row
    yields the +0.309 the audit reported.
    """
    inv = recompute_from_summary(0.1281, -18.3675, 210, 8)
    assert inv["fixed_ratio"] == pytest.approx(0.309, abs=0.01)
    assert inv["penalty_inflation"] == pytest.approx(13.8, rel=0.1)
