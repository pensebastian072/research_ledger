"""Q-REL-05: does the international-ETF reversal clear the gate as a DEPLOYABLE book?

Supersedes Q-REL-04, which answered a different question than the one that matters.

WHAT CHANGED, AND WHY IT IS THE WHOLE POINT. Q-REL-04 measured +34.1%/yr at t 2.66
PER DEPLOYED LEG -- i.e. the return earned by capital while it was actually in a trade.
The same trades, on the same days, sized as a FIXED-SLOT unlevered book, give
+5.2%/yr at t 1.79. Nothing about the signal changed; only the capital convention did.
Per-deployed-leg silently assumes capital appears exactly when a signal fires and
disappears when it does not, which is leverage. A fixed-slot book holds N slots and
lets unfilled slots sit in cash -- which is what you can actually trade, so it is the
number that goes on the record.

The weighting convention is named in the frozen bar this time, precisely because
Q-REL-04 showed it moves the t-stat from 1.79 to 2.66.

FROZEN BAR (questions.yaml, pre-registered before this grader existed):
    on a FIXED-SLOT, UNLEVERED basket -- alpha against BOTH SPY and the basket's own
    buy-and-hold must be > 0 with Newey-West t > 2, AND the corrected DSR must clear
    1.645 with PBO < 0.5.

The two-benchmark requirement is the fix for Q-GATE-04, the defect that contaminates
almost every near-miss on this box: a strategy that is long an asset under some
condition and flat otherwise books market beta as edge unless something subtracts it.
Here the second leg is stricter than the first -- the basket's own buy-and-hold
returned +10.28%/yr over this window, so beating SPY is not enough; the signal has to
beat simply owning the same nine ETFs.

WHAT IS ALREADY KNOWN, so this grader is not expected to produce a surprise. The beta
controls were run on 2026-07-30 and recorded in questions.yaml: net exposure -0.0002,
beta to SPY +0.0381, R2 0.0063 -- genuinely market-neutral, NOT a beta harvest, which
is the control that killed the qlib batteries. But alpha vs SPY came in at t +1.693 and
alpha vs the basket at t +1.816, against a bar of t > 2, with corrected DSR +1.281
against 1.645. This grader exists to put that FAIL on the record reproducibly rather
than leaving it as prose in a YAML file.

    .venv\\Scripts\\python.exe -m graders.intl_reversal_portfolio [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from cost.costs import half_spread  # noqa: E402
from gate import evaluate_gate_fixed  # noqa: E402
from graders.intl_reversal import FAMILY, VOL_WINDOW, _signals  # noqa: E402
from graders.lean_ic import newey_west_t  # noqa: E402
from graders.qlib_leans import load_prices  # noqa: E402

QUESTION_ID = "Q-REL-05"
SUPERSEDES = "Q-REL-04"

# Pre-registered in questions.yaml as the best Q-REL-04 setting. NOT re-searched here:
# re-opening the threshold grid after seeing which one won is the laundering the
# frozen bar exists to prevent.
Z_THRESHOLD = 1.5
DRIVER = "self"                 # driver=self t 2.77 beat driver=SPY t 1.74

# The deployable convention. One slot per family name, unfilled slots sit in CASH.
# Gross exposure can never exceed 1.0, which is what "unlevered" means here.
N_SLOTS = len(FAMILY)           # 9
BENCH = "SPY"

MIN_YEARS_POSITIVE = 5          # carried over from Q-REL-04's frozen bar
ALPHA_T_BAR = 2.0               # frozen
NW_LAGS = 1


def _leg_returns(rets: pd.DataFrame, family: list[str],
                 costs: dict[str, float]) -> pd.DataFrame:
    """Per-name net daily return of the reversal leg, aligned on a common index.

    A name contributes 0.0 on days it is not signalled -- that is the whole point of
    the fixed-slot convention, and it is why this number is smaller than Q-REL-04's.
    """
    cols = {}
    for s in family:
        drv = s if DRIVER == "self" else BENCH
        df = _signals(rets, s, drv, Z_THRESHOLD)
        if df.empty:
            continue
        sig = df["sig"].to_numpy(float)
        fwd = df["fwd"].to_numpy(float)
        turns = np.abs(np.diff(np.concatenate([[0.0], sig])))
        cols[s] = pd.Series(sig * fwd - turns * costs[s], index=df.index)
    return pd.DataFrame(cols).sort_index()


def _alpha_vs(port: pd.Series, bench: pd.Series) -> dict:
    """Newey-West alpha of `port` against `bench`.

    beta is OLS; alpha is then the mean of the beta-adjusted series, t-stat via the
    same Bartlett HAC estimator every other grader on this box uses. Reported with
    beta and R2 so a near-zero-beta claim can be checked rather than asserted.
    """
    df = pd.concat([port.rename("p"), bench.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return {"n": int(len(df)), "alpha_bps": None, "t_nw": None,
                "beta": None, "r2": None, "ann_pct": None}
    p = df["p"].to_numpy(float)
    b = df["b"].to_numpy(float)
    bc = b - b.mean()
    var = float(bc @ bc)
    beta = float(bc @ (p - p.mean()) / var) if var > 0 else 0.0
    resid = p - beta * b
    t = newey_west_t(resid, lags=NW_LAGS)
    pred = beta * bc + p.mean()
    ss_res = float(((p - pred) ** 2).sum())
    ss_tot = float(((p - p.mean()) ** 2).sum())
    return {"n": int(len(df)),
            "alpha_bps": round(float(resid.mean()) * 1e4, 3),
            "t_nw": None if t is None else round(t, 3),
            "beta": round(beta, 4),
            "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
            "ann_pct": round(float(resid.mean()) * 252 * 100, 2)}


def run() -> dict:
    prices = load_prices()
    rets = prices.pct_change().iloc[1:]
    family = [s for s in FAMILY if s in rets.columns]
    costs = {s: half_spread(s).half_spread for s in family}

    legs = _leg_returns(rets, family, costs)

    # ── the deployable book: 1/N_SLOTS per name, unfilled slots earn cash (0.0) ──
    port = legs.fillna(0.0).sum(axis=1) / N_SLOTS
    # exposure diagnostic: how much of the book is actually working on a given day
    filled = (legs.fillna(0.0) != 0.0).sum(axis=1)

    bench_spy = rets[BENCH].reindex(port.index)
    bench_basket = rets[family].reindex(port.index).mean(axis=1)   # own buy-and-hold

    net = port.to_numpy(float)
    idx = port.index

    per_year, pos = {}, 0
    for y, g in port.groupby(port.index.year):
        if len(g) < 20:
            continue
        tt = newey_west_t(g.to_numpy(float), lags=NW_LAGS)
        per_year[int(y)] = {"n": int(len(g)),
                            "mean_bps": round(float(g.mean()) * 1e4, 3),
                            "ann_pct": round(float(g.mean()) * 252 * 100, 2),
                            "t": None if tt is None else round(tt, 2)}
        pos += int(g.mean() > 0)

    # Concentration check. Q-REL-04 recorded "not a 2020 artifact" -- true for the
    # per-deployed-leg convention, FALSE for the deployable book, which is the whole
    # reason this question exists. Computed rather than asserted.
    tot = sum(v["mean_bps"] * v["n"] for v in per_year.values())
    n_all = sum(v["n"] for v in per_year.values())
    ex = {y: v for y, v in per_year.items() if y != 2020}
    tot_ex = sum(v["mean_bps"] * v["n"] for v in ex.values())
    n_ex = sum(v["n"] for v in ex.values())
    concentration = {
        "ann_pct_all_years": round(tot / n_all * 252 / 100, 2) if n_all else None,
        "ann_pct_excluding_2020": round(tot_ex / n_ex * 252 / 100, 2) if n_ex else None,
        "pnl_share_2020": round((per_year.get(2020, {}).get("mean_bps", 0.0)
                                 * per_year.get(2020, {}).get("n", 0)) / tot, 4)
                          if tot else None,
        "day_share_2020": round(per_year.get(2020, {}).get("n", 0) / n_all, 4)
                          if n_all else None,
        "years_with_t_gt_2": [int(y) for y, v in per_year.items() if (v["t"] or 0) > 2],
        "negative_years": [int(y) for y, v in per_year.items() if v["mean_bps"] < 0],
    }

    # Trial count. Q-REL-04 charged its own pre-registered grid (2 drivers x 4
    # thresholds). Q-REL-05 re-uses ONE frozen setting and adds one capital
    # convention, so it inherits that grid and adds nothing to it.
    n_trials = 2 * 4

    gate = evaluate_gate_fixed(net, n_trials=n_trials)
    a_spy = _alpha_vs(port, bench_spy)
    a_basket = _alpha_vs(port, bench_basket)

    checks = {
        "alpha_vs_spy_positive": bool((a_spy["alpha_bps"] or 0) > 0),
        "alpha_vs_spy_t_gt_2": bool((a_spy["t_nw"] or 0) > ALPHA_T_BAR),
        "alpha_vs_basket_positive": bool((a_basket["alpha_bps"] or 0) > 0),
        "alpha_vs_basket_t_gt_2": bool((a_basket["t_nw"] or 0) > ALPHA_T_BAR),
        "dsr_gt_1645": bool(gate["passes"]),
        "years_positive": bool(pos >= MIN_YEARS_POSITIVE),
    }
    verdict = "YES" if all(checks.values()) else "NO"
    failed = [k for k, v in checks.items() if not v]

    return {
        "question": QUESTION_ID,
        "supersedes": SUPERSEDES,
        "as_of": date.today().isoformat(),
        "family": family,
        "convention": {
            "slots": N_SLOTS, "levered": False, "driver": DRIVER,
            "z_threshold": Z_THRESHOLD,
            "note": "unfilled slots sit in cash; gross exposure never exceeds 1.0",
        },
        "n_days": int(net.size),
        "mean_bps": round(float(net.mean()) * 1e4, 3),
        "ann_pct": round(float(net.mean()) * 252 * 100, 2),
        "t_nw": (lambda t: None if t is None else round(t, 3))(
            newey_west_t(net, lags=NW_LAGS)),
        "hit_rate": round(float((net[net != 0] > 0).mean()), 4) if (net != 0).any() else None,
        "slots_filled": {
            "mean": round(float(filled.mean()), 2),
            "max": int(filled.max()),
            "pct_days_fully_idle": round(float((filled == 0).mean()), 4),
        },
        "benchmarks": {
            "spy_ann_pct": round(float(bench_spy.mean()) * 252 * 100, 2),
            "basket_buy_hold_ann_pct": round(float(bench_basket.mean()) * 252 * 100, 2),
        },
        "alpha_vs_spy": a_spy,
        "alpha_vs_basket_buy_hold": a_basket,
        "per_year": per_year,
        "concentration": concentration,
        "years_positive": pos,
        "years_graded": len(per_year),
        "gate": {
            "passes": gate["passes"],
            "dsr_fixed": (gate["deflated_sharpe"] or {}).get("ratio"),
            "dsr_shipped": (gate["deflated_sharpe_shipped"] or {}).get("ratio"),
            "pbo": gate["pbo"],
            "profit_factor": gate["profit_factor"],
            "n_trials": n_trials,
            "reasons": gate["reasons"],
        },
        "frozen_bar_checks": checks,
        "failed_checks": failed,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()

    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 100)
    print(f"{QUESTION_ID}: is the international-ETF reversal a DEPLOYABLE book? "
          f"(supersedes {SUPERSEDES})")
    print("=" * 100)
    c = rep["convention"]
    print(f"convention: {c['slots']} fixed slots, unlevered, driver={c['driver']}, "
          f"|z| >= {c['z_threshold']}")
    print(f"            {c['note']}")
    print(f"family: {', '.join(rep['family'])}")
    sf = rep["slots_filled"]
    print(f"\nslots filled: mean {sf['mean']} of {c['slots']}, max {sf['max']}, "
          f"fully idle on {sf['pct_days_fully_idle']:.1%} of days")
    print(f"book: {rep['mean_bps']} bps/day = {rep['ann_pct']}%/yr, "
          f"NW t = {rep['t_nw']}, n = {rep['n_days']}")
    b = rep["benchmarks"]
    print(f"benchmarks: SPY {b['spy_ann_pct']}%/yr | "
          f"basket buy-and-hold {b['basket_buy_hold_ann_pct']}%/yr")

    print("\nFROZEN BAR (pre-registered before this grader existed):")
    for label, a in (("vs SPY", rep["alpha_vs_spy"]),
                     ("vs basket buy-and-hold", rep["alpha_vs_basket_buy_hold"])):
        mark = "OK " if (a["t_nw"] or 0) > ALPHA_T_BAR else "FAIL"
        print(f"  [{mark}] alpha {label:24} {a['ann_pct']:+7.2f}%/yr  "
              f"t = {a['t_nw']}  (bar t > {ALPHA_T_BAR})   beta {a['beta']} R2 {a['r2']}")
    g = rep["gate"]
    print(f"  [{'OK ' if g['passes'] else 'FAIL'}] gate  DSR {g['dsr_fixed']} "
          f"(bar > 1.645), PBO {g['pbo']} (bar < 0.5), n_trials {g['n_trials']}")
    print(f"  [{'OK ' if rep['years_positive'] >= MIN_YEARS_POSITIVE else 'FAIL'}] "
          f"years positive {rep['years_positive']}/{rep['years_graded']} "
          f"(bar >= {MIN_YEARS_POSITIVE})")

    print("\nper year:")
    for y, r in sorted(rep["per_year"].items()):
        print(f"    {y}  n={r['n']:4}  {r['mean_bps']:+8.3f} bps/day  "
              f"{r['ann_pct']:+7.2f}%/yr  t={r['t']}")

    print(f"\nVERDICT: {rep['verdict']}")
    if rep["failed_checks"]:
        print("  failed: " + ", ".join(rep["failed_checks"]))
    print("\nNOTE: a fixed-slot book is what can actually be traded. Q-REL-04's "
          "+34.1%/yr at t 2.66\n      was per DEPLOYED LEG -- same signal, same trades, "
          "leverage supplied by the\n      capital convention. This number supersedes it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
