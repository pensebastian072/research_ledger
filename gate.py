"""The canonical gate, imported -- plus a correct Deflated Sharpe beside it.

`pbo_cscv`, `profit_factor`, `sharpe`, `purged_kfold`, `walk_forward_splits` and
`evaluate_gate` are re-exported verbatim from
`macro_gpu_lab/macro_gpu_lab/validate.py` (hard rule 6). Do not vendor or rewrite
them here; if the import breaks, fix `MACRO_GPU_LAB_DIR`.

`deflated_sharpe` is ALSO re-exported unchanged, so historical numbers stay
reproducible -- but it is provably wrong (see `audit/dsr_audit.py` and CLAUDE.md).
`deflated_sharpe_fixed` is the corrected Bailey & Lopez de Prado (2014) form. Both
are reported side by side; neither silently replaces the other.

The defect, precisely
---------------------
BLdP define

    DSR = PSR(SR*) = Phi[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) ]

    SR* = E[max SR] = sigma_SR * [ (1-gamma)*Phi^-1(1 - 1/N)
                                   + gamma*Phi^-1(1 - 1/(N*e)) ]

`SR*` and `SR` must be in the same units (per-observation SR). The bracketed term
is dimensionless -- a number of sigma_SR's -- so it MUST be multiplied by
sigma_SR before being subtracted from SR. The shipped code subtracts the bracket
itself, then scales the whole difference by sqrt(T-1)/denom, inflating the penalty
by roughly sqrt(T-1). At T=210 that is 14.5x; the term reaches 1.46 in
per-observation SR units, which corresponds to an annualised Sharpe near 23 --
a bar no real series clears, which is why the shipped gate returns prob=0.0000
even on data with a genuine edge.

Estimating sigma_SR
-------------------
sigma_sr="se" (default): sigma_SR is approximated by the SR estimator's own
    standard error, sqrt(denom^2/(T-1)), which makes SR* = bracket*denom/sqrt(T-1)
    and the whole thing collapse to  ratio = SR*sqrt(T-1)/denom - bracket.
    Conservative, and needs nothing but the series itself.
sigma_sr="cross": sigma_SR is the sample std of the SRs actually observed across
    the N trials -- the faithful reading of the paper. Requires `trial_srs`.
    Slightly more permissive here in practice, because the trial SRs on this box
    disperse a little less than the SE approximation assumes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import ledger_paths

# ── import the canonical gate ───────────────────────────────────────
_MACRO = ledger_paths.MACRO_GPU_LAB
if not (_MACRO / "macro_gpu_lab" / "validate.py").exists():
    raise ImportError(
        f"canonical gate not found under {_MACRO}. Set MACRO_GPU_LAB_DIR. "
        "Do NOT vendor the math (CLAUDE.md hard rule 6)."
    )
if str(_MACRO) not in sys.path:
    sys.path.insert(0, str(_MACRO))

from macro_gpu_lab.validate import (  # noqa: E402
    deflated_sharpe,
    evaluate_gate,
    pbo_cscv,
    profit_factor,
    purged_kfold,
    sharpe,
    walk_forward_splits,
)
from macro_gpu_lab.validate import _norm_cdf, _norm_ppf  # noqa: E402

EULER_GAMMA = 0.5772156649015329
# Follows the upstream config, raised 0.0 -> 1.645 on 2026-07-30 alongside the
# deflated_sharpe unit fix. With a correct DSR, ratio>0 is a MEDIAN test that
# best-of-8 pure noise clears 44.8% of the time; 1.645 is prob>0.95.
DEFLATED_SHARPE_MIN = 1.645
PBO_MAX = 0.5

__all__ = [
    "deflated_sharpe", "deflated_sharpe_fixed", "expected_max_sr_units",
    "evaluate_gate", "evaluate_gate_fixed", "pbo_cscv", "profit_factor",
    "purged_kfold", "sharpe", "walk_forward_splits", "recompute_from_summary",
    "DEFLATED_SHARPE_MIN", "PBO_MAX",
]


def expected_max_sr_units(n_trials: int) -> float:
    """The dimensionless BLdP bracket: E[max SR] in units of sigma_SR."""
    n = max(1, int(n_trials))
    if n == 1:
        return 0.0
    return ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n)
            + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n * math.e)))


def _moments(arr: np.ndarray):
    mu, sigma = arr.mean(), arr.std(ddof=1)
    if sigma <= 0:
        return None
    sr = mu / sigma
    skew = float(((arr - mu) ** 3).mean() / sigma ** 3)
    kurt = float(((arr - mu) ** 4).mean() / sigma ** 4)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return sr, skew, kurt, denom


def deflated_sharpe_fixed(pnls, n_trials: int = 1, *, sigma_sr: str = "se",
                          trial_srs=None):
    """Unit-consistent Deflated Sharpe. Same output shape as the canonical one."""
    arr = np.asarray(pnls, dtype=float)
    if arr.size < 8:
        return None
    m = _moments(arr)
    if m is None:
        return None
    sr, skew, kurt, denom = m
    T = arr.size
    n = max(1, int(n_trials))
    bracket = expected_max_sr_units(n)
    z = sr * math.sqrt(T - 1) / denom          # PSR z-score against SR* = 0

    if n == 1:
        ratio, sr_star, sigma_hat = z, 0.0, denom / math.sqrt(T - 1)
    elif sigma_sr == "cross":
        if trial_srs is None or len(trial_srs) < 2:
            raise ValueError('sigma_sr="cross" needs trial_srs (>=2 trial SRs)')
        sigma_hat = float(np.std(np.asarray(trial_srs, dtype=float), ddof=1))
        sr_star = sigma_hat * bracket
        ratio = (sr - sr_star) * math.sqrt(T - 1) / denom
    elif sigma_sr == "se":
        sigma_hat = denom / math.sqrt(T - 1)
        sr_star = sigma_hat * bracket
        ratio = z - bracket                    # algebraically identical, no cancellation
    else:
        raise ValueError(f'sigma_sr must be "se" or "cross", got {sigma_sr!r}')

    return {"sr": round(sr, 4), "ratio": round(ratio, 4),
            "prob": round(_norm_cdf(ratio), 4), "n": T, "n_trials": n,
            "sr_star": round(sr_star, 6), "sigma_sr": round(sigma_hat, 6),
            "bracket": round(bracket, 4), "sigma_sr_mode": sigma_sr,
            "psr_z": round(z, 4), "skew": round(skew, 4), "kurt": round(kurt, 4)}


def evaluate_gate_fixed(pnls, n_trials: int = 1, *, sigma_sr: str = "se",
                        trial_srs=None) -> dict:
    """`evaluate_gate`'s contract, with the corrected DSR. Reports BOTH ratios."""
    arr = np.asarray(pnls, dtype=float)
    ds_old = deflated_sharpe(arr, n_trials=n_trials)
    ds_new = deflated_sharpe_fixed(arr, n_trials=n_trials, sigma_sr=sigma_sr,
                                   trial_srs=trial_srs)
    pbo = pbo_cscv(arr)
    reasons = []
    if ds_new is None:
        reasons.append("deflated_sharpe unavailable (n < 8 or zero variance)")
    elif ds_new["ratio"] <= DEFLATED_SHARPE_MIN:
        reasons.append(f"deflated_sharpe_ratio {ds_new['ratio']} <= {DEFLATED_SHARPE_MIN}")
    if pbo is None:
        reasons.append("pbo unavailable (n < CSCV_N_GROUPS*4)")
    elif pbo >= PBO_MAX:
        reasons.append(f"pbo {pbo} >= {PBO_MAX}")
    return {
        "passes": not reasons,
        "deflated_sharpe": ds_new,
        "deflated_sharpe_shipped": ds_old,
        "pbo": pbo,
        "profit_factor": round(profit_factor(arr), 4),
        "sharpe": sharpe(arr),
        "n_trades": int(arr.size),
        "reasons": reasons,
        "thresholds": {"deflated_sharpe_min": DEFLATED_SHARPE_MIN, "pbo_max": PBO_MAX},
    }


def legacy_deflated_sharpe_ratio(pnls, n_trials: int = 1) -> float | None:
    """The PRE-2026-07-30 formula, kept only as a reference for the inversion below.

    `macro_gpu_lab/validate.py` was patched on 2026-07-30, so the live
    `deflated_sharpe` no longer produces the numbers sitting in the historical
    scorecards. `recompute_from_summary` exists to correct those OLD rows, so it has
    to be validated against the formula that wrote them -- which is this. It is a
    reference implementation, never a gate: nothing should call it to score anything.
    """
    arr = np.asarray(pnls, dtype=float)
    if arr.size < 8:
        return None
    mu, sigma = arr.mean(), arr.std(ddof=1)
    if sigma <= 0:
        return None
    sr = mu / sigma
    T = arr.size
    skew = float(((arr - mu) ** 3).mean() / sigma ** 3)
    kurt = float(((arr - mu) ** 4).mean() / sigma ** 4)
    n = max(1, int(n_trials))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    if n == 1:
        return sr * math.sqrt(T - 1) / denom
    return (sr - expected_max_sr_units(n)) * math.sqrt(T - 1) / denom


def recompute_from_summary(sr: float, shipped_ratio: float, n: int,
                           n_trials: int) -> dict | None:
    """Correct a HISTORICAL scorecard WITHOUT re-running its backtest.

    Applies to rows written BEFORE 2026-07-30, when the upstream formula was
    `ratio = (sr - bracket)*sqrt(n-1)/denom`. Rows written after the fix are already
    correct and must not be passed through here -- `collect.py` distinguishes them by
    the scorecard's date.

    That old formula lets a stored (sr, ratio, n, n_trials) quadruple pin `denom`
    exactly:

        denom = (sr - bracket)*sqrt(n-1)/ratio

    and the corrected ratio is then  sr*sqrt(n-1)/denom - bracket. Pure algebra on
    numbers already on disk -- no data is re-read and no model is re-fit. Returns
    None when the inversion is undefined (n_trials < 2, ratio == 0, n < 2).
    """
    if sr is None or shipped_ratio in (None, 0) or n is None or n < 2:
        return None
    if n_trials is None or int(n_trials) < 2:
        return None      # n_trials==1 has no e_max term; shipped == fixed already
    bracket = expected_max_sr_units(int(n_trials))
    root = math.sqrt(n - 1)
    denom = (sr - bracket) * root / shipped_ratio
    if not math.isfinite(denom) or denom <= 0:
        return None
    ratio = sr * root / denom - bracket
    return {"sr": sr, "shipped_ratio": round(shipped_ratio, 4),
            "fixed_ratio": round(ratio, 4), "prob": round(_norm_cdf(ratio), 4),
            "implied_denom": round(denom, 6), "bracket": round(bracket, 4),
            "n": int(n), "n_trials": int(n_trials),
            "penalty_inflation": round(root / denom, 2)}
