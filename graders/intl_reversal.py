"""Q-REL-04: is the international-ETF overnight reversal real and tradable?

Supersedes Q-REL-01, which called this a stale-NAV artifact and stopped. It is not
one ticker -- the same sign runs across the whole family, and INDA's own lag-1
autocorrelation (-0.214) is stronger than any cross-pair:

    INDA own  -0.214 | SPY->INDA -0.165 | SPY->EEM -0.143 | SPY->EWY -0.129
    SPY->FXI  -0.105 | SPY->VGK  -0.104 | SPY->EWJ  -0.095

Mechanism: a US-listed foreign ETF trades all day on US sentiment while its
underlying market is shut, overshoots, and reverts when the home market reprices.
That is microstructure, not a macro relationship -- but microstructure that pays is
still an edge, and Q-REL-01 never checked whether it pays.

Now it can be checked properly, because Q-COST-01 measured what a side costs:
INDA 1.02 bps, EEM 0.87, FXI 1.30 -- not the flat 5 bps that would have killed it.

Three things this separates, which the original miner conflated:

  1. SELF vs LEAD -- trade on the ETF's own prior return, versus on SPY's. If self
     dominates, this is a simpler and more robust effect than any "relationship".
  2. UNCONDITIONAL vs CONDITIONAL -- reverse every day, versus only after a large
     prior move (|z| > 1, 1.5, 2 on trailing vol). Conditioning cuts turnover
     sharply, and turnover is what the cost multiplies. Thresholds were
     pre-registered in questions.yaml before this ran.
  3. PER YEAR -- if the whole edge is 2020 stress, that is a NO regardless of the
     pooled number.

    .venv\\Scripts\\python.exe -m graders.intl_reversal [--json]
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
from graders.lean_ic import newey_west_t  # noqa: E402
from graders.qlib_leans import load_prices  # noqa: E402

QUESTION_ID = "Q-REL-04"
SUPERSEDES = "Q-REL-01"
FAMILY = ["INDA", "EEM", "EWY", "EWJ", "FXI", "VGK", "EWU", "EWG", "EWZ"]
LEAD = "SPY"
Z_THRESHOLDS = (0.0, 1.0, 1.5, 2.0)      # 0.0 = unconditional; pre-registered
VOL_WINDOW = 60
MIN_YEARS_POSITIVE = 5                   # frozen in questions.yaml


def _signals(rets: pd.DataFrame, symbol: str, driver: str, z_thresh: float):
    """Yesterday's move (own or SPY's), gated on |z|, as a reversal signal."""
    r = rets[symbol]
    d = rets[driver]
    vol = d.rolling(VOL_WINDOW).std()
    z = d / vol
    prior = d.shift(1)
    zt = z.shift(1)
    sig = -np.sign(prior)                          # reverse the prior move
    if z_thresh > 0:
        sig = sig.where(zt.abs() >= z_thresh, 0.0)
    df = pd.concat([sig.rename("sig"), r.rename("fwd")], axis=1).dropna()
    return df


def _pnl(df: pd.DataFrame, cost_half: float) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Charge two crossings on every change of position, not on every bar held."""
    sig = df["sig"].to_numpy(float)
    fwd = df["fwd"].to_numpy(float)
    turns = np.abs(np.diff(np.concatenate([[0.0], sig])))
    gross = sig * fwd
    net = gross - turns * cost_half            # each unit of turnover = 1 crossing
    live = sig != 0
    return net[live], df.index[live]


def _grade(net: np.ndarray, idx: pd.DatetimeIndex, label: str,
           n_trials: int) -> dict:
    out = {"label": label, "n": int(net.size),
           "n_clusters": int(len(set(idx.date))) if net.size else 0,
           "mean_bps": None, "t_nw": None, "hit_rate": None, "ann_pct": None,
           "years_positive": None, "years_graded": None, "per_year": {},
           "gate": None, "verdict": "NO-DATA", "why": ""}
    if net.size < 30:
        out["why"] = f"only {net.size} live days"
        return out
    out["mean_bps"] = round(float(net.mean()) * 1e4, 3)
    t = newey_west_t(net, lags=1)
    out["t_nw"] = None if t is None else round(t, 3)
    out["hit_rate"] = round(float((net > 0).mean()), 4)
    out["ann_pct"] = round(float(net.mean()) * 252 * 100, 2)

    s = pd.Series(net, index=idx)
    per_year, pos = {}, 0
    for y, g in s.groupby(s.index.year):
        if len(g) < 20:
            continue
        tt = newey_west_t(g.to_numpy(float), lags=1)
        per_year[int(y)] = {"n": int(len(g)),
                            "mean_bps": round(float(g.mean()) * 1e4, 3),
                            "t": None if tt is None else round(tt, 2)}
        pos += int(g.mean() > 0)
    out["per_year"] = per_year
    out["years_graded"] = len(per_year)
    out["years_positive"] = pos

    g = evaluate_gate_fixed(net, n_trials=n_trials)
    out["gate"] = {"passes": g["passes"],
                   "dsr_fixed": (g["deflated_sharpe"] or {}).get("ratio"),
                   "dsr_shipped": (g["deflated_sharpe_shipped"] or {}).get("ratio"),
                   "pbo": g["pbo"], "profit_factor": g["profit_factor"]}

    ok_years = pos >= MIN_YEARS_POSITIVE
    if net.mean() > 0 and (out["t_nw"] or 0) > 2 and g["passes"] and ok_years:
        out["verdict"] = "YES"
        out["why"] = (f"{out['mean_bps']}bps/day t={out['t_nw']} "
                      f"ann={out['ann_pct']}% years+={pos}/{len(per_year)}")
    else:
        out["verdict"] = "NO"
        bits = []
        if net.mean() <= 0:
            bits.append("mean <= 0")
        if (out["t_nw"] or 0) <= 2:
            bits.append(f"t={out['t_nw']} <= 2")
        if not g["passes"]:
            bits.append("gate: " + "; ".join(g["reasons"]))
        if not ok_years:
            bits.append(f"only {pos}/{len(per_year)} years positive, "
                        f"need {MIN_YEARS_POSITIVE}")
        out["why"] = ", ".join(bits)
    return out


def run() -> dict:
    prices = load_prices()
    rets = prices.pct_change().iloc[1:]
    family = [s for s in FAMILY if s in rets.columns]
    costs = {s: half_spread(s).half_spread for s in family}
    cost_src = {s: half_spread(s).source for s in family}

    # the full pre-registered grid: 2 drivers x 4 thresholds x (basket + per symbol)
    n_trials = 2 * len(Z_THRESHOLDS)

    rows, per_symbol = [], []
    for driver_name in ("self", LEAD):
        for zt in Z_THRESHOLDS:
            # ── basket: equal-weight the family, each charged its own cost ──
            legs = []
            for s in family:
                drv = s if driver_name == "self" else LEAD
                df = _signals(rets, s, drv, zt)
                if df.empty:
                    continue
                net, idx = _pnl(df, costs[s])
                if net.size:
                    legs.append(pd.Series(net, index=idx))
            if not legs:
                continue
            basket = pd.concat(legs, axis=1).mean(axis=1).dropna()
            label = (f"basket driver={driver_name} z>={zt}")
            rows.append({**_grade(basket.to_numpy(float), basket.index, label,
                                  n_trials),
                         "driver": driver_name, "z": zt, "scope": "basket"})

    # per-symbol at the most promising pre-registered setting, for transparency
    for s in family:
        df = _signals(rets, s, s, 1.0)
        if df.empty:
            continue
        net, idx = _pnl(df, costs[s])
        per_symbol.append({**_grade(net, idx, f"{s} self z>=1.0", n_trials),
                           "symbol": s, "cost_bps": round(costs[s] * 1e4, 2),
                           "cost_source": cost_src[s]})

    best = max((r for r in rows if r["mean_bps"] is not None),
               key=lambda r: r["mean_bps"], default=None)
    return {
        "question": QUESTION_ID,
        "supersedes": SUPERSEDES,
        "as_of": date.today().isoformat(),
        "family": family,
        "costs_bps": {s: round(c * 1e4, 2) for s, c in costs.items()},
        "cost_sources": cost_src,
        "n_trials": n_trials,
        "rows": rows,
        "per_symbol": per_symbol,
        "best": best,
        "verdict": ("YES" if any(r["verdict"] == "YES" for r in rows) else
                    "NO" if any(r["verdict"] == "NO" for r in rows) else "NO-DATA"),
        "caveats": [
            "this is a MICROSTRUCTURE effect, not a macro relationship -- it lives in "
            "the ETF's premium/discount to NAV and is the kind of thing authorised "
            "participants arbitrage. Quoted spread is a floor on the cost of competing "
            "with them, and market impact is not modelled at all.",
            "the reversal requires trading at or near the close every day it fires; "
            "closing-auction execution is assumed and never verified here.",
            "costs are a per-symbol level from 12 sampled 2025-2026 sessions applied "
            "across 2016-2026. Spreads were wider earlier and much wider in 2020, so "
            "the early years are undercharged.",
            "the z thresholds and both drivers were pre-registered before this ran; "
            "n_trials charges the whole grid.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"intl_reversal_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 100)
    print(f"{QUESTION_ID}: international-ETF reversal -- real and tradable? "
          f"(supersedes {SUPERSEDES})")
    print("=" * 100)
    print(f"family: {', '.join(rep['family'])}")
    print(f"costs (bps/side): " + ", ".join(
        f"{k} {v}" for k, v in rep["costs_bps"].items()))
    print(f"\n  {'setting':<30s} {'bps/day':>8s} {'ann%':>7s} {'t':>7s} {'hit':>6s} "
          f"{'PF':>6s} {'DSRfix':>8s} {'PBO':>6s} {'yr+':>6s} {'n':>6s}  verdict")
    for r in rep["rows"]:
        g = r["gate"] or {}
        def f(v, spec, w):
            return f"{'-':>{w}s}" if v is None else f"{v:>{w}{spec}}"
        yr = (f"{r['years_positive']}/{r['years_graded']}"
              if r["years_graded"] else "-")
        print(f"  {r['label']:<30s} {f(r['mean_bps'],'.2f',8)} "
              f"{f(r['ann_pct'],'.1f',7)} {f(r['t_nw'],'.2f',7)} "
              f"{f(r['hit_rate'],'.3f',6)} {f(g.get('profit_factor'),'.3f',6)} "
              f"{f(g.get('dsr_fixed'),'.3f',8)} {f(g.get('pbo'),'.3f',6)} "
              f"{yr:>6s} {r['n']:>6d}  {r['verdict']}")

    print(f"\n  per symbol at the pre-registered self / z>=1.0 setting")
    print(f"  {'symbol':<8s} {'cost':>6s} {'src':<10s} {'bps/day':>8s} {'ann%':>7s} "
          f"{'t':>7s} {'yr+':>6s} {'n':>6s}")
    for r in rep["per_symbol"]:
        yr = f"{r['years_positive']}/{r['years_graded']}" if r["years_graded"] else "-"
        print(f"  {r['symbol']:<8s} {r['cost_bps']:>6.2f} {r['cost_source']:<10s} "
              f"{(r['mean_bps'] if r['mean_bps'] is not None else float('nan')):>8.2f} "
              f"{(r['ann_pct'] if r['ann_pct'] is not None else float('nan')):>7.1f} "
              f"{(r['t_nw'] if r['t_nw'] is not None else float('nan')):>7.2f} "
              f"{yr:>6s} {r['n']:>6d}")

    if rep["best"]:
        b = rep["best"]
        print(f"\n  best setting: {b['label']} -> {b['mean_bps']} bps/day "
              f"({b['ann_pct']}%/yr), t={b['t_nw']}, {b['verdict']}")
        if b.get("per_year"):
            print("  per year: " + ", ".join(
                f"{y}:{v['mean_bps']:+.1f}" for y, v in b["per_year"].items()))
    print(f"\nVERDICT: {rep['verdict']}")
    for r in rep["rows"]:
        if r["why"] and r["verdict"] != "NO-DATA":
            print(f"  {r['label']}: {r['why']}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
