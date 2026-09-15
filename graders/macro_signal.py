"""Q-MACRO-02: is the signal macro_gpu_lab exports every day actually right?

macro_gpu_lab gated four models once, on 2026-07-04, and has published a daily
`tv_signal.json` ever since via `shift_export` -- which writes a signal but never
re-gates. Nobody has ever checked the published call against the tape.

The daily export logs are the only point-in-time record of what was claimed, and
they are unambiguous:

    5d: calm=False p_eruption=0.5898 pred=True  -> {'short_spy_overlay': 'LONG_SPY', ...}

Parsing them recovers an honest forward track: what the model said, on the day it
said it, with no chance of hindsight. The catch is size -- exports start
2026-07-04, so there are ~20 observations. That is NO-DATA, and this grader says so
in those words rather than reporting a number it cannot support. It accumulates one
row per trading day from here.

    .venv\\Scripts\\python.exe -m graders.macro_signal [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from graders.lean_ic import newey_west_t  # noqa: E402
from graders.qlib_leans import load_prices  # noqa: E402

QUESTION_ID = "Q-MACRO-02"
MIN_OBS = 60                    # frozen in questions.yaml; below this -> NO-DATA
SIGNAL_LINE = re.compile(
    r"^\s*(?P<h>\d+d):\s*calm=(?P<calm>True|False)\s+"
    r"p_eruption=(?P<p>[0-9.]+)\s+pred=(?P<pred>True|False)\s*->\s*(?P<actions>\{.*\})")
LOG_DATE = re.compile(r"shift_export_(\d{4})(\d{2})(\d{2})\.log$")

# what each published action means as a directional call on SPY
ACTION_TO_SPY_SIGN = {
    "LONG_SPY": 1,
    "SHORT_SPY": -1,
    "FLAT": 0,
    "LONG_VOL": -1,      # long vol is a short-risk expression
    "LONG_STRADDLE": 0,  # direction-neutral: graded on |move|, not sign
}


def parse_exports() -> pd.DataFrame:
    folder = ledger_paths.MACRO_RUNS
    rows = []
    if not folder.exists():
        return pd.DataFrame(columns=["date", "horizon", "p_eruption", "pred",
                                     "calm", "strategy", "action"])
    for log in sorted(folder.glob("shift_export_*.log")):
        m = LOG_DATE.search(log.name)
        if not m:
            continue
        stamp = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        try:
            text = log.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            sm = SIGNAL_LINE.match(line)
            if not sm:
                continue
            try:
                actions = json.loads(sm.group("actions").replace("'", '"'))
            except json.JSONDecodeError:
                actions = {}
            for strategy, action in actions.items():
                rows.append({"date": pd.Timestamp(stamp),
                             "horizon": int(sm.group("h").rstrip("d")),
                             "p_eruption": float(sm.group("p")),
                             "pred": sm.group("pred") == "True",
                             "calm": sm.group("calm") == "True",
                             "strategy": strategy, "action": action})
    return pd.DataFrame(rows)


def grade(signals: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    if signals.empty or "SPY" not in prices.columns:
        return []
    spy = prices["SPY"].dropna()
    out = []
    for (h, strategy), g in signals.groupby(["horizon", "strategy"], sort=True):
        fwd = spy.shift(-h) / spy - 1.0
        g = g.copy()
        g["sign"] = g["action"].map(ACTION_TO_SPY_SIGN)
        g["fwd"] = g["date"].map(fwd)
        g = g.dropna(subset=["fwd", "sign"])
        directional = g[g["sign"] != 0]
        pnl = (directional["sign"] * directional["fwd"]).to_numpy(float)
        n = int(len(directional))
        row = {
            "horizon": int(h), "strategy": strategy,
            "n_signal_days": int(len(g)),
            "n_directional": n,
            "n_distinct_actions": int(g["action"].nunique()),
            "actions_seen": sorted(g["action"].unique().tolist()),
            "mean_pnl": round(float(pnl.mean()), 6) if n else None,
            "hit_rate": round(float((pnl > 0).mean()), 4) if n else None,
            "base_rate": round(float((g["fwd"] > 0).mean()), 4) if len(g) else None,
            "t_nw": None,
            "verdict": "NO-DATA",
            "why": "",
        }
        if n >= 8:
            t = newey_west_t(pnl, lags=max(0, h - 1))
            row["t_nw"] = round(t, 3) if t is not None else None
        if n < MIN_OBS:
            row["why"] = (f"n={n} directional observations, need {MIN_OBS}. "
                          f"Exports begin 2026-07-04; this accumulates ~1/trading day.")
        else:
            positive = (row["mean_pnl"] or 0) > 0 and (row["t_nw"] or 0) > 2
            row["verdict"] = "YES" if positive else "NO"
            row["why"] = (f"mean_pnl={row['mean_pnl']} t_NW={row['t_nw']} "
                          f"hit={row['hit_rate']} vs base={row['base_rate']}")
        if row["n_distinct_actions"] == 1:
            row["why"] += (f" NOTE: only ever emitted {row['actions_seen'][0]} -- "
                           "a constant call cannot be distinguished from buy-and-hold.")
        out.append(row)
    return out


def run() -> dict:
    signals = parse_exports()
    prices = load_prices({"SPY"})
    rows = grade(signals, prices)
    dates = sorted(signals["date"].dt.date.unique().tolist()) if len(signals) else []
    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "evidence": str(ledger_paths.MACRO_RUNS),
        "export_days": len(dates),
        "date_range": [str(dates[0]), str(dates[-1])] if dates else None,
        "min_obs_required": MIN_OBS,
        "rows": rows,
        "verdict": ("NO-DATA" if not rows or all(r["verdict"] == "NO-DATA" for r in rows)
                    else "MIXED"),
        "caveats": [
            "the gate scorecards in macro_gpu_lab/journal/scorecards are all dated "
            "2026-07-04 -- the daily task runs shift_export only and never re-gates, "
            "so the stored PASS/FAIL verdicts do not refresh.",
            "this grades the PUBLISHED action, not the model's internal probability; "
            "a constant action is flagged because it is indistinguishable from "
            "buy-and-hold on this sample.",
            "a full historical regrade of surprise_shift is possible on the "
            "multi-year panel but requires re-running macro_gpu_lab's own model, "
            "which is outside this read-only repo -- see proposals/.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"macro_signal_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 90)
    print(f"{QUESTION_ID}: is the daily macro_gpu_lab exported signal right?")
    print("=" * 90)
    print(f"evidence: {rep['export_days']} export days"
          + (f" ({rep['date_range'][0]} .. {rep['date_range'][1]})"
             if rep["date_range"] else ""))
    print(f"\n  {'h':>3s} {'strategy':<22s} {'n_dir':>6s} {'mean_pnl':>10s} "
          f"{'hit':>6s} {'base':>6s} {'t_NW':>7s}  verdict")
    for r in rep["rows"]:
        def f(v, spec, w):
            return f"{'-':>{w}s}" if v is None else f"{v:>{w}{spec}}"
        print(f"  {r['horizon']:>3d} {r['strategy']:<22s} {r['n_directional']:>6d} "
              f"{f(r['mean_pnl'], '.6f', 10)} {f(r['hit_rate'], '.4f', 6)} "
              f"{f(r['base_rate'], '.4f', 6)} {f(r['t_nw'], '.2f', 7)}  {r['verdict']}")
    for r in rep["rows"]:
        if r["why"]:
            print(f"    {r['horizon']}d {r['strategy']}: {r['why']}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
