"""Q-REL-01: are the 25 lead-lag "survivors" a real relationship or a stale-NAV artifact?

The original miner (`qlib_lab/qlib_lab/experiments.py::mine_lead_lag`) ran 22,800
tests at Bonferroni alpha 2.19e-6 and reported 25 survivors. Two things in that
output say "artifact" rather than "relationship":

  * discovery |corr| ~0.21 collapses to validation |corr| ~0.077 -- a 3x shrink,
    which is what you see when a threshold selects noise;
  * the survivor list is dominated by INDA as the TARGET, with SPY/DIA/XLF/VGK/VLUE
    as leaders at lag 1.

INDA is a US-listed India ETF. Its underlying market closes before the US session
opens, so INDA's US-hours price partly anticipates the next Indian session -- the
documented international-ETF stale-NAV effect. Any US leader will "predict" it at
lag 1. That is microstructure, not a macro relationship, and it is arbitraged
inside the ETF's premium/discount rather than being available as alpha.

This module re-tests the claim with four changes, all pre-registered in
questions.yaml BEFORE the run:

  (a) THIRD SPLIT. Thirds instead of halves: discover / validate / holdout. A pair
      must keep the same sign and clear a magnitude floor in ALL THREE.
  (b) REGION FILTER. Targets whose underlying market is shut during US hours are
      excluded, killing the stale-NAV channel. Both the filtered and unfiltered
      runs are reported so the size of the artifact is visible.
  (c) BLOCK BOOTSTRAP on the holdout correlation (circular, 21-day blocks) instead
      of a Bonferroni z on raw daily correlations, which assumes iid returns.
  (d) PnL OR IT DOES NOT COUNT. Every surviving pair is converted to a holdout
      trading rule and pushed through the gate. A pair that survives the
      correlation screen but fails the gate is recorded as NO.

    .venv\\Scripts\\python.exe -m graders.leadlag_remine [--json] [--lags 1,2,3,5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from gate import deflated_sharpe, deflated_sharpe_fixed, pbo_cscv, profit_factor  # noqa: E402
from graders.qlib_leans import load_prices  # noqa: E402

QUESTION_ID = "Q-REL-01"
LAGS = (1, 2, 3, 5)
MIN_ABS_CORR = 0.05            # same floor the original used on its validation half
BOOT_BLOCKS = 21
BOOT_DRAWS = 2000
COST_PER_SIDE = 0.0005

# Underlying-market region per symbol. "US" = underlying trades during US hours, so
# no stale-NAV channel. Anything else is a foreign or 24h underlying.
NON_US_UNDERLYING = {
    # Asia-Pacific
    "INDA", "EWJ", "EWY", "EWT", "EWH", "EWS", "EWA", "FXI", "KWEB", "VPL", "EEM",
    # Europe / UK
    "VGK", "EWG", "EWU", "EFA",
    # Latin America
    "EWZ", "EWW", "ILF",
    # 24h / FX / crypto
    "BTC", "EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDCNH", "GOLD", "OIL", "COPPER",
}


def _p_from_r(r: float, m: int) -> float:
    """Two-sided p for a correlation, iid assumption -- as the original did."""
    z = abs(r) * math.sqrt(max(1, m))
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def block_bootstrap_corr(a: np.ndarray, b: np.ndarray, *, block: int = BOOT_BLOCKS,
                         draws: int = BOOT_DRAWS, seed: int = 20260729) -> dict | None:
    """Circular block bootstrap CI for corr(a, b); preserves autocorrelation."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 3 * block:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    base = _corr(a, b)
    if base is None:
        return None
    out = np.empty(draws)
    idx_grid = (np.arange(block)[None, :] + rng.integers(0, n, size=(draws, n_blocks))
                [:, :, None]) % n
    idx = idx_grid.reshape(draws, -1)[:, :n]
    for i in range(draws):
        out[i] = np.corrcoef(a[idx[i]], b[idx[i]])[0, 1]
    out = out[np.isfinite(out)]
    if out.size < 100:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"corr": round(base, 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
            "frac_same_sign": round(float((np.sign(out) == np.sign(base)).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mine(rets: pd.DataFrame, lags=LAGS, *, exclude_non_us_targets: bool,
         n_splits: int = 3) -> dict:
    """Thirds-validated lead-lag scan. Returns survivors + the test count."""
    names = list(rets.columns)
    targets = [t for t in names
               if not (exclude_non_us_targets and t in NON_US_UNDERLYING)]
    n = len(rets)
    cut = [int(n * i / n_splits) for i in range(n_splits + 1)]
    parts = [rets.iloc[cut[i]:cut[i + 1]] for i in range(n_splits)]

    n_tests = len(names) * len(targets) * len(lags)
    alpha = 0.05 / max(1, n_tests)
    survivors = []
    for lag in lags:
        shifted = [p.shift(lag).iloc[lag:] for p in parts]
        tgts = [p.iloc[lag:] for p in parts]
        for lead in names:
            corrs = [t.corrwith(s[lead]) for s, t in zip(shifted, tgts)]
            for tgt in targets:
                if tgt == lead:
                    continue
                vals = [c.get(tgt) for c in corrs]
                if any(v is None or pd.isna(v) for v in vals):
                    continue
                vals = [float(v) for v in vals]
                disc = vals[0]
                if _p_from_r(disc, len(parts[0]) - lag) >= alpha:
                    continue
                signs = {int(np.sign(v)) for v in vals}
                if len(signs) != 1 or 0 in signs:
                    continue
                if any(abs(v) < MIN_ABS_CORR for v in vals[1:]):
                    continue
                survivors.append({
                    "leader": lead, "target": tgt, "lag": lag,
                    "corr_discovery": round(vals[0], 4),
                    "corr_validation": round(vals[1], 4),
                    "corr_holdout": round(vals[2], 4),
                    "shrink_disc_to_holdout": round(abs(vals[2]) / abs(vals[0]), 3)
                    if vals[0] else None,
                    "target_non_us_underlying": tgt in NON_US_UNDERLYING,
                })
    survivors.sort(key=lambda d: -abs(d["corr_holdout"]))
    return {"n_tests": n_tests, "bonferroni_alpha": alpha,
            "n_survivors": len(survivors), "survivors": survivors,
            "n_targets": len(targets), "n_leaders": len(names)}


def to_pnl_and_gate(rets: pd.DataFrame, pair: dict, *, holdout_frac: float = 1 / 3,
                    n_trials: int = 1) -> dict:
    """sign(leader lagged) * target return on the HOLDOUT third, then the gate."""
    n = len(rets)
    start = int(n * (1 - holdout_frac))
    ho = rets.iloc[start:]
    lead = ho[pair["leader"]].shift(pair["lag"])
    tgt = ho[pair["target"]]
    df = pd.concat([lead.rename("lead"), tgt.rename("tgt")], axis=1).dropna()
    if df.empty:
        return {"n": 0, "reason": "no overlapping holdout rows"}
    direction = np.sign(df["lead"]) * np.sign(pair["corr_holdout"])
    pnl = (direction * df["tgt"]).to_numpy(float)
    # trading every day the sign flips: charge a round trip on each change
    turns = np.abs(np.diff(np.concatenate([[0.0], direction.to_numpy(float)]))) / 2.0
    pnl_net = pnl - turns * 2.0 * COST_PER_SIDE
    ds_old = deflated_sharpe(pnl_net, n_trials=n_trials)
    ds_new = deflated_sharpe_fixed(pnl_net, n_trials=n_trials)
    pbo = pbo_cscv(pnl_net)
    passes_fixed = (ds_new is not None and ds_new["ratio"] > 0
                    and pbo is not None and pbo < 0.5)
    return {"n": int(pnl_net.size),
            "mean_pnl_net": round(float(pnl_net.mean()), 6),
            "profit_factor_net": round(profit_factor(pnl_net), 4),
            "dsr_shipped": (ds_old or {}).get("ratio"),
            "dsr_fixed": (ds_new or {}).get("ratio"),
            "pbo": pbo,
            "passes_gate_fixed_dsr": bool(passes_fixed)}


def run(lags=LAGS) -> dict:
    prices = load_prices()
    rets = prices.pct_change().iloc[1:]
    rets = rets.dropna(axis=1, thresh=int(0.8 * len(rets)))

    unfiltered = mine(rets, lags, exclude_non_us_targets=False)
    filtered = mine(rets, lags, exclude_non_us_targets=True)

    # how much of the original signal was the stale-NAV channel?
    n_non_us = sum(1 for s in unfiltered["survivors"]
                   if s["target_non_us_underlying"])
    artifact_share = (round(n_non_us / unfiltered["n_survivors"], 4)
                      if unfiltered["n_survivors"] else None)

    # bootstrap + PnL-gate the survivors that pass the region filter
    graded = []
    n_trials = filtered["n_tests"]
    for pair in filtered["survivors"][:25]:
        n = len(rets)
        ho = rets.iloc[int(n * 2 / 3):]
        a = ho[pair["leader"]].shift(pair["lag"]).to_numpy(float)
        b = ho[pair["target"]].to_numpy(float)
        boot = block_bootstrap_corr(a, b)
        gate = to_pnl_and_gate(rets, pair, n_trials=n_trials)
        graded.append({**pair, "bootstrap": boot, "gate": gate})

    survives_all = [g for g in graded
                    if (g.get("bootstrap") or {}).get("excludes_zero")
                    and g["gate"].get("passes_gate_fixed_dsr")]

    verdict = "NO" if not survives_all else "YES"
    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "universe": len(rets.columns),
        "rows": int(len(rets)),
        "date_range": [str(rets.index.min().date()), str(rets.index.max().date())],
        "lags": list(lags),
        "unfiltered": {k: v for k, v in unfiltered.items() if k != "survivors"},
        "filtered": {k: v for k, v in filtered.items() if k != "survivors"},
        "unfiltered_top": unfiltered["survivors"][:10],
        "artifact_share_non_us_targets": artifact_share,
        "n_non_us_targets_in_unfiltered": n_non_us,
        "graded": graded,
        "n_survive_everything": len(survives_all),
        "verdict": verdict,
        "caveats": [
            "the original miner validated on halves with a |corr|>0.05 floor and "
            "no PnL step; this uses thirds, a block bootstrap, and requires the "
            "gate. It is a STRICTER test of the same claim, not a new search.",
            "n_trials for the deflated Sharpe is the full test count of the "
            "filtered scan, so the multiple-comparison cost of mining is charged "
            "to every survivor.",
            "the region filter removes the stale-NAV channel by excluding targets "
            "whose underlying market is shut during US hours. It does not claim "
            "that channel is unreal -- only that it is microstructure, already "
            "arbitraged in the ETF premium/discount, and not a macro relationship.",
            "correlation is measured on returns of a fixed present-day universe; "
            "delisted names are absent, so this inherits a survivorship tilt.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lags", type=str, default="1,2,3,5")
    args = ap.parse_args()
    lags = tuple(int(x) for x in args.lags.split(",") if x.strip())

    rep = run(lags)
    out = ledger_paths.JOURNAL / f"leadlag_remine_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 96)
    print(f"{QUESTION_ID}: lead-lag survivors -- real relationship or stale-NAV artifact?")
    print("=" * 96)
    print(f"{rep['universe']} assets, {rep['rows']} rows, "
          f"{rep['date_range'][0]} .. {rep['date_range'][1]}, lags {rep['lags']}")
    u, f = rep["unfiltered"], rep["filtered"]
    print(f"\nunfiltered scan : {u['n_tests']:,} tests -> {u['n_survivors']} survivors")
    print(f"  of which target a non-US underlying: "
          f"{rep['n_non_us_targets_in_unfiltered']} "
          f"({rep['artifact_share_non_us_targets']})")
    print(f"region-filtered : {f['n_tests']:,} tests -> {f['n_survivors']} survivors "
          f"({f['n_targets']} eligible targets)")

    print("\ntop unfiltered survivors (the original claim, thirds-validated)")
    print(f"  {'leader':>8s} -> {'target':<8s} {'lag':>3s} {'disc':>8s} {'valid':>8s} "
          f"{'hold':>8s} {'shrink':>7s}  non-US")
    for s in rep["unfiltered_top"]:
        print(f"  {s['leader']:>8s} -> {s['target']:<8s} {s['lag']:>3d} "
              f"{s['corr_discovery']:>8.4f} {s['corr_validation']:>8.4f} "
              f"{s['corr_holdout']:>8.4f} "
              f"{(s['shrink_disc_to_holdout'] if s['shrink_disc_to_holdout'] is not None else float('nan')):>7.3f}"
              f"  {'YES' if s['target_non_us_underlying'] else '-'}")

    if rep["graded"]:
        print("\nregion-filtered survivors, bootstrapped and gated")
        print(f"  {'leader':>8s} -> {'target':<8s} {'lag':>3s} {'hold':>8s} "
              f"{'boot_CI':>18s} {'PF_net':>7s} {'DSRfix':>8s} {'PBO':>6s} gate")
        for g in rep["graded"]:
            b, gt = g.get("bootstrap") or {}, g["gate"]
            ci = (f"[{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}]" if b else "n/a")
            pbo = gt.get("pbo")
            print(f"  {g['leader']:>8s} -> {g['target']:<8s} {g['lag']:>3d} "
                  f"{g['corr_holdout']:>8.4f} {ci:>18s} "
                  f"{(gt.get('profit_factor_net') or float('nan')):>7.3f} "
                  f"{(gt.get('dsr_fixed') if gt.get('dsr_fixed') is not None else float('nan')):>8.3f} "
                  f"{(pbo if pbo is not None else float('nan')):>6.3f} "
                  f"{'PASS' if gt.get('passes_gate_fixed_dsr') else 'fail'}")
    else:
        print("\nno survivors passed the region filter -- nothing left to bootstrap.")

    print(f"\nsurvive correlation + bootstrap + gate: {rep['n_survive_everything']}")
    print(f"VERDICT: {rep['verdict']}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
