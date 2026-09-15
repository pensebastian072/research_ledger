"""Q-QLIB-03: does the qlib long-short survive the REAL cost?

`Q-QLIB-01` established there is barely any cross-sectional signal: 5d rank IC +0.008
(t_NW 1.07), hit rate 51.6% against a 55.8% base rate. But the diagnostics also
measured the gross top8-bottom8 spread at **+9.0 bps per 5-day rebalance** -- roughly
+4.5%/yr gross -- while `qlib_lab/pipeline.py` charges `COST_PER_SIDE = 0.0005`, a flat
5 bps on all 76 assets. `Q-COST-01` then measured the real number: SPY 0.23 bps, the
median symbol 0.76 bps, 86 of 87 symbols CHEAPER than the flat assumption.

So the FAIL may have been a cost artefact rather than a signal verdict, and that is
worth settling before qlib is written off. Nothing is re-fit: this reuses the
point-in-time `qlib_score_history.parquet` and only changes what a trade is charged.

Three portfolio constructions, because the decile table was non-monotonic (top decile
+6.3 bps excess, bottom -2.1, decile 5 worst at -7.0) -- which says the SHORT leg is
noise and paying to hold it may be the whole problem:

  long_short   top-TOPK minus bottom-TOPK      (what the pipeline does: 2 legs)
  long_rest    top-TOPK minus the rest         (1 concentrated leg + broad hedge)
  long_only    top-TOPK, market-neutral by     (1 leg -- half the crossings)
               subtracting the cross-sectional mean

Each is charged at the measured per-symbol cost, then again at the p90 stress cost.
An edge that only clears at median cost is recorded as NO.

    .venv\\Scripts\\python.exe -m graders.qlib_recost [--json]
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
from cost.costs import FLAT_ASSUMPTION, half_spread  # noqa: E402
from gate import evaluate_gate_fixed  # noqa: E402
from graders.lean_ic import newey_west_t  # noqa: E402
from graders.qlib_leans import load_leans, load_prices  # noqa: E402

QUESTION_ID = "Q-QLIB-03"
TOPK = 8                    # matches qlib_lab config.TOPK
MIN_CLUSTERS = 100          # frozen in questions.yaml
HORIZONS = (5, 21)


def _cost_map(symbols, stress: bool) -> dict[str, float]:
    return {s: half_spread(s, stress=stress).half_spread for s in symbols}


def _cost_sources(symbols) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in symbols:
        src = half_spread(s).source
        out[src] = out.get(src, 0) + 1
    return out


def build_pnls(leans: pd.DataFrame, prices: pd.DataFrame, horizon: int,
               construction: str, costs: dict[str, float]) -> tuple[list, list]:
    """One PnL per rebalance, non-overlapping (step = horizon), cost per crossing."""
    score_col = f"score_{horizon}d"
    fwd = prices.shift(-horizon) / prices - 1.0
    dates = sorted(leans["date"].unique())
    by_date = {d: g for d, g in leans.groupby("date")}

    pnls, stamps = [], []
    i = 0
    # step by `horizon` sessions so consecutive trades never overlap -- same
    # non-overlap rule qlib_lab's oos_portfolio_pnls uses.
    while i < len(dates):
        d = dates[i]
        g = by_date.get(d)
        i += horizon
        if g is None or score_col not in g or len(g) < 2 * TOPK + 1:
            continue
        if d not in fwd.index:
            continue
        row = fwd.loc[d]
        g = g.dropna(subset=[score_col])
        ranked = g.sort_values(score_col, ascending=False)["asset"].tolist()
        top = [a for a in ranked[:TOPK] if a in row.index and np.isfinite(row[a])]
        bot = [a for a in ranked[-TOPK:] if a in row.index and np.isfinite(row[a])]
        rest = [a for a in ranked[TOPK:] if a in row.index and np.isfinite(row[a])]
        if not top:
            continue

        def leg(names):
            return float(np.mean([row[a] for a in names])) if names else 0.0

        def crossing_cost(names):
            # in and out: two crossings per name, averaged over the leg
            return (2.0 * float(np.mean([costs.get(a, FLAT_ASSUMPTION)
                                         for a in names])) if names else 0.0)

        if construction == "long_short":
            if not bot:
                continue
            gross = leg(top) - leg(bot)
            cost = crossing_cost(top) + crossing_cost(bot)
        elif construction == "long_rest":
            if not rest:
                continue
            gross = leg(top) - leg(rest)
            cost = crossing_cost(top) + crossing_cost(rest)
        elif construction == "long_only":
            all_names = [a for a in ranked if a in row.index and np.isfinite(row[a])]
            if len(all_names) < 2 * TOPK:
                continue
            gross = leg(top) - leg(all_names)      # excess over the cross-section
            cost = crossing_cost(top)
        else:
            raise ValueError(construction)

        pnls.append(gross - cost)
        stamps.append(str(pd.Timestamp(d).date()))
    return pnls, stamps


def grade(pnls, stamps, horizon, label) -> dict:
    arr = np.asarray(pnls, dtype=float)
    n = arr.size
    clusters = len(set(stamps))
    out = {"label": label, "horizon": horizon, "n": int(n), "n_clusters": clusters,
           "mean_bps": None, "t_nw": None, "hit_rate": None,
           "gate": None, "verdict": "NO-DATA", "why": ""}
    if n < 8:
        out["why"] = f"only {n} rebalances"
        return out
    out["mean_bps"] = round(float(arr.mean()) * 1e4, 3)
    t = newey_west_t(arr, lags=0)     # rebalances are non-overlapping by construction
    out["t_nw"] = None if t is None else round(t, 3)
    out["hit_rate"] = round(float((arr > 0).mean()), 4)
    g = evaluate_gate_fixed(arr, n_trials=6)   # 3 constructions x 2 horizons
    out["gate"] = {"passes": g["passes"], "dsr_fixed": (g["deflated_sharpe"] or {}).get("ratio"),
                   "dsr_shipped": (g["deflated_sharpe_shipped"] or {}).get("ratio"),
                   "pbo": g["pbo"], "profit_factor": g["profit_factor"]}
    if clusters < MIN_CLUSTERS:
        out["verdict"] = "NO-DATA"
        out["why"] = f"{clusters} clusters, need {MIN_CLUSTERS}"
    elif arr.mean() > 0 and (out["t_nw"] or 0) > 2 and g["passes"]:
        out["verdict"] = "YES"
        out["why"] = f"mean {out['mean_bps']}bps t={out['t_nw']} gate passes"
    else:
        out["verdict"] = "NO"
        bits = []
        if arr.mean() <= 0:
            bits.append("mean <= 0")
        if (out["t_nw"] or 0) <= 2:
            bits.append(f"t={out['t_nw']} <= 2")
        if not g["passes"]:
            bits.append("gate: " + "; ".join(g["reasons"]))
        out["why"] = ", ".join(bits)
    return out


def run() -> dict:
    leans = load_leans()
    prices = load_prices(set(leans["asset"].unique()))
    syms = sorted(set(leans["asset"].unique()) & set(prices.columns))

    rows = []
    for stress in (False, True):
        costs = _cost_map(syms, stress)
        for h in HORIZONS:
            for c in ("long_short", "long_rest", "long_only"):
                pnls, stamps = build_pnls(leans, prices, h, c, costs)
                label = f"{c}{'@stress' if stress else '@median'}"
                rows.append({**grade(pnls, stamps, h, label), "stress": stress,
                             "construction": c})
        # and the flat assumption, for the like-for-like comparison
        if not stress:
            flat = {s: FLAT_ASSUMPTION for s in syms}
            for h in HORIZONS:
                pnls, stamps = build_pnls(leans, prices, h, "long_short", flat)
                rows.append({**grade(pnls, stamps, h, "long_short@flat5bps"),
                             "stress": False, "construction": "long_short"})

    best = max((r for r in rows if r["verdict"] != "NO-DATA"),
               key=lambda r: (r["mean_bps"] or -1e9), default=None)
    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "n_symbols": len(syms),
        "cost_sources": _cost_sources(syms),
        "median_cost_bps": round(float(np.median(
            [half_spread(s).half_spread for s in syms])) * 1e4, 3),
        "rows": rows,
        "best": best,
        "verdict": ("YES" if any(r["verdict"] == "YES" for r in rows) else
                    "NO" if any(r["verdict"] == "NO" for r in rows) else "NO-DATA"),
        "caveats": [
            "no model is re-fit -- this reuses the point-in-time score history and only "
            "changes what a trade is charged.",
            "costs are QUOTED half-spreads with no market impact, so they are a floor; "
            "the @stress rows use the within-session p90 and are the honest test.",
            "n_trials=6 (3 constructions x 2 horizons). The construction choice was "
            "pre-registered in questions.yaml before this ran, driven by the measured "
            "non-monotonic decile table, not by trying variants until one passed.",
            "long_only is market-neutral by subtracting the cross-sectional mean, which "
            "is not the same as being dollar-neutral and would need a hedge in practice.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"qlib_recost_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 96)
    print(f"{QUESTION_ID}: does the qlib long-short survive the REAL cost?")
    print("=" * 96)
    print(f"{rep['n_symbols']} symbols | median measured cost "
          f"{rep['median_cost_bps']} bps/side vs the flat 5.00 bps assumed | "
          f"sources {rep['cost_sources']}")
    print(f"\n  {'construction':<26s} {'h':>3s} {'mean_bps':>9s} {'t':>7s} "
          f"{'hit':>6s} {'PF':>6s} {'DSRfix':>8s} {'PBO':>6s} {'n':>5s} {'clus':>5s}  verdict")
    for r in rep["rows"]:
        g = r["gate"] or {}
        def f(v, spec, w):
            return f"{'-':>{w}s}" if v is None else f"{v:>{w}{spec}}"
        print(f"  {r['label']:<26s} {r['horizon']:>3d} {f(r['mean_bps'],'.2f',9)} "
              f"{f(r['t_nw'],'.2f',7)} {f(r['hit_rate'],'.3f',6)} "
              f"{f(g.get('profit_factor'),'.3f',6)} {f(g.get('dsr_fixed'),'.3f',8)} "
              f"{f(g.get('pbo'),'.3f',6)} {r['n']:>5d} {r['n_clusters']:>5d}  "
              f"{r['verdict']}")
    print(f"\nVERDICT: {rep['verdict']}")
    for r in rep["rows"]:
        if r["why"]:
            print(f"  {r['label']} {r['horizon']}d: {r['why']}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
