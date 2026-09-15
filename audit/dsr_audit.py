"""Audit the canonical Deflated Sharpe. Verify the ruler before trusting the FAILs.

Every model on this box is SHADOW because `deflated_sharpe_ratio <= 0`, across four
repos, on hundreds of configurations, including on a strategy whose permutation
p-value was 0.0315 with PBO 0.0. A gate that never passes anything is either a
correct gate meeting uniformly worthless models, or a broken ruler. This module
decides which, and the answer is: broken ruler.

Three independent checks, all runnable offline:

1. NULL CALIBRATION -- the decisive one. Generate N independent trials of pure
   zero-mean noise, submit the best. A correct deflated Sharpe returns prob ~= 0.5,
   because deflation is defined to remove exactly the selection advantage and
   nothing more. Anything that returns ~0.0 on noise is over-penalising; ~1.0 is
   under-penalising.
2. POWER -- plant a known true SR in one trial of a best-of-8 search and sweep its
   size. A usable test's pass rate must rise with the planted edge.
3. SCORECARD RECOMPUTATION -- invert every stored (sr, ratio, n, n_trials) on disk
   to recover its `denom`, then re-apply the corrected formula. No backtest is
   re-run and no model is re-fit; this is algebra on numbers already committed.

Nothing here edits `validate.py`. Output is evidence for
`proposals/2026-07-29-dsr-unit-bug.md`, which a human accepts or rejects.

    .venv\\Scripts\\python.exe -m audit.dsr_audit [--mc 3000] [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from gate import (  # noqa: E402
    deflated_sharpe,
    deflated_sharpe_fixed,
    expected_max_sr_units,
    recompute_from_summary,
)
from scorecards import iter_gate_rows  # noqa: E402

SEED = 20260729
NULL_TOLERANCE = 0.10          # |mean prob - 0.5| a correct formula must stay inside


def null_calibration(n_mc: int = 3000, T: int = 210,
                     trials_grid=(2, 8, 42)) -> list[dict]:
    """Best-of-N on pure noise. Correct DSR -> mean prob ~= 0.5."""
    rng = np.random.default_rng(SEED)
    out = []
    for n_trials in trials_grid:
        shipped, fixed_se, fixed_cross = [], [], []
        for _ in range(n_mc):
            trials = rng.standard_normal((n_trials, T)) * 0.01
            srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
            best = trials[int(np.argmax(srs))]
            d_old = deflated_sharpe(best, n_trials=n_trials)
            d_se = deflated_sharpe_fixed(best, n_trials=n_trials, sigma_sr="se")
            d_cr = deflated_sharpe_fixed(best, n_trials=n_trials, sigma_sr="cross",
                                         trial_srs=srs)
            shipped.append(d_old["prob"])
            fixed_se.append(d_se["prob"])
            fixed_cross.append(d_cr["prob"])
        row = {"n_trials": n_trials, "T": T, "n_mc": n_mc,
               "shipped_mean_prob": round(float(np.mean(shipped)), 4),
               "fixed_se_mean_prob": round(float(np.mean(fixed_se)), 4),
               "fixed_cross_mean_prob": round(float(np.mean(fixed_cross)), 4),
               "shipped_pass_rate": round(float(np.mean(np.array(shipped) > 0.5)), 4),
               "fixed_se_pass_rate": round(float(np.mean(np.array(fixed_se) > 0.5)), 4)}
        row["shipped_calibrated"] = abs(row["shipped_mean_prob"] - 0.5) <= NULL_TOLERANCE
        row["fixed_se_calibrated"] = abs(row["fixed_se_mean_prob"] - 0.5) <= NULL_TOLERANCE
        out.append(row)
    return out


def power_curve(n_mc: int = 1500, T: int = 210, n_trials: int = 8,
                true_srs=(0.0, 0.05, 0.15, 0.30)) -> list[dict]:
    """Plant a known per-observation SR in one arm of a best-of-N search."""
    rng = np.random.default_rng(SEED + 1)
    out = []
    for true_sr in true_srs:
        shipped, fixed = [], []
        for _ in range(n_mc):
            trials = rng.standard_normal((n_trials, T)) * 0.01
            trials[0] += true_sr * 0.01
            srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
            best = trials[int(np.argmax(srs))]
            shipped.append(deflated_sharpe(best, n_trials=n_trials)["ratio"] > 0)
            fixed.append(deflated_sharpe_fixed(best, n_trials=n_trials)["ratio"] > 0)
        out.append({"true_per_obs_sr": true_sr,
                    "annualised_equiv": round(true_sr * math.sqrt(252), 2),
                    "shipped_pass_rate": round(float(np.mean(shipped)), 4),
                    "fixed_se_pass_rate": round(float(np.mean(fixed)), 4)})
    return out


def recompute_scorecards() -> list[dict]:
    """Correct every stored gate row that carries enough numbers to invert."""
    rows = []
    for r in iter_gate_rows():
        fix = recompute_from_summary(r.get("sr"), r.get("dsr_ratio"),
                                     r.get("n_trades"), r.get("n_trials"))
        if fix is None:
            continue
        pbo = r.get("pbo")
        flips = (fix["fixed_ratio"] > 0 and pbo is not None and pbo < 0.5
                 and not r.get("passes"))
        rows.append({**r, **fix, "flips_to_pass": flips,
                     "strict_pass_1p645": fix["fixed_ratio"] > 1.645})
    rows.sort(key=lambda x: -x["fixed_ratio"])
    return rows


def _fmt(v, spec=".4f", width=9):
    if v is None:
        return f"{'-':>{width}s}"
    try:
        return f"{v:>{width}{spec}}"
    except (TypeError, ValueError):
        return f"{str(v):>{width}s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=3000, help="Monte-Carlo draws")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    null = null_calibration(n_mc=args.mc)
    power = power_curve()
    scards = recompute_scorecards()

    shipped_broken = not any(r["shipped_calibrated"] for r in null)
    fixed_ok = all(r["fixed_se_calibrated"] for r in null)
    shipped_powerless = all(r["shipped_pass_rate"] == 0.0 for r in null) and \
        all(p["shipped_pass_rate"] == 0.0 for p in power)
    verdict = "BUG" if (shipped_broken and fixed_ok) else "VERIFIED"

    report = {
        "as_of": date.today().isoformat(),
        "verdict": verdict,
        "shipped_is_calibrated_on_noise": not shipped_broken,
        "shipped_never_passes": shipped_powerless,
        "fixed_se_is_calibrated_on_noise": fixed_ok,
        "null_calibration": null,
        "power_curve": power,
        "bracket_by_n_trials": {n: round(expected_max_sr_units(n), 4)
                                for n in (2, 8, 20, 34, 42, 126, 132)},
        "scorecard_recomputation": scards,
        "n_flips_to_pass": sum(1 for r in scards if r["flips_to_pass"]),
        "n_strict_pass_1p645": sum(1 for r in scards if r["strict_pass_1p645"]),
        "note": ("Nothing was patched. deflated_sharpe in macro_gpu_lab/validate.py is "
                 "unchanged; this is evidence for a human decision. DSR ratio > 0 is a "
                 "median test (a coin flip clears it ~46% of the time) -- see the "
                 "strict_pass_1p645 column for the prob>0.95 bar."),
    }

    out = ledger_paths.JOURNAL / f"dsr_audit_{report['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if verdict == "VERIFIED" else 1

    print("=" * 78)
    print(f"DSR AUDIT -- verdict: {verdict}")
    print("=" * 78)
    print("\n1. NULL CALIBRATION (best-of-N on pure noise; correct DSR -> prob ~= 0.50)")
    print(f"   {'n_trials':>8s} {'shipped':>9s} {'fixed-SE':>9s} {'fixed-cross':>12s}"
          f"  {'shipped ok?':>11s}")
    for r in null:
        print(f"   {r['n_trials']:8d} {r['shipped_mean_prob']:9.4f} "
              f"{r['fixed_se_mean_prob']:9.4f} {r['fixed_cross_mean_prob']:12.4f}"
              f"  {str(r['shipped_calibrated']):>11s}")

    print("\n2. POWER (best-of-8, one arm carries a real edge)")
    print(f"   {'true SR/obs':>11s} {'~annual':>8s} {'shipped':>9s} {'fixed-SE':>9s}")
    for p in power:
        print(f"   {p['true_per_obs_sr']:11.2f} {p['annualised_equiv']:8.2f} "
              f"{p['shipped_pass_rate']:9.3f} {p['fixed_se_pass_rate']:9.3f}")

    print("\n3. STORED SCORECARDS RECOMPUTED (algebra on committed numbers, no re-fit)")
    print(f"   {'source':<46s} {'sr':>8s} {'shipped':>9s} {'fixed':>8s} "
          f"{'PBO':>6s} {'x':>5s} flips")
    for r in scards:
        pbo = r.get("pbo")
        print(f"   {r['label'][:46]:<46s} {r['sr']:8.4f} {r['shipped_ratio']:9.3f} "
              f"{r['fixed_ratio']:8.3f} {_fmt(pbo, '.3f', 6)} "
              f"{r['penalty_inflation']:5.1f} "
              f"{'YES' if r['flips_to_pass'] else '-'}")

    print(f"\n   rows recomputed: {len(scards)}"
          f" | flip to PASS under the frozen ratio>0 bar: {report['n_flips_to_pass']}"
          f" | survive the stricter ratio>1.645: {report['n_strict_pass_1p645']}")
    print(f"\nwrote {out}")
    print("\nvalidate.py NOT patched -- see proposals/2026-07-29-dsr-unit-bug.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
