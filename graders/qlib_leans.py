"""Q-QLIB-01: does the qlib Alpha158 daily lean predict forward returns?

The evidence has been sitting on disk unmeasured. `qlib_lab/journal/exports/
qlib_score_history.parquet` holds 133,986 point-in-time rows -- 76 assets,
2019-07-18 to 2026-07-23 -- produced by the SAME walk-forward the pipeline gates
on, with the no-lookahead guarantees documented in `qlib_lab/qlib_lab/score_history.py`
(refit only every REFIT_EVERY sessions, on labels fully realised before the refit
date; a model is only ever applied forward of its fit window). Prices come from
`qlib_lab/csv/*.csv`, the same source the leans were computed from.

So this is not a new backtest and costs nothing: it is the information-coefficient
test that the daily FAIL/PASS gate never ran. The gate asked "does the long-short
portfolio clear a deflated Sharpe"; this asks the prior question, "is there any
predictive signal in the cross-section at all", which is what makes a FAIL
interpretable.

One honest caveat printed with every run: REFIT_EVERY=126 and MIN_TRAIN_DAYS=750
mean only ~14 fits exist per horizon over the whole span, so calendar years are
NOT independent samples -- consecutive years can share a fitted model. A per-year
table is still far more informative than a pooled number, but it is not 7 draws.

    .venv\\Scripts\\python.exe -m graders.qlib_leans [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from graders.lean_ic import format_table, ic_table, summarise  # noqa: E402

QUESTION_ID = "Q-QLIB-01"
HORIZONS = (5, 21)


def load_prices(symbols: set[str] | None = None) -> pd.DataFrame:
    """Wide close panel from qlib_lab/csv. `factor` is already applied upstream."""
    folder = ledger_paths.QLIB_PRICES
    if not folder.exists():
        raise FileNotFoundError(f"price folder missing: {folder}")
    series = {}
    for csv in sorted(folder.glob("*.csv")):
        sym = csv.stem.upper()
        if symbols is not None and sym not in symbols:
            continue
        try:
            df = pd.read_csv(csv, usecols=["date", "close"], parse_dates=["date"])
        except Exception:
            continue
        if df.empty:
            continue
        series[sym] = df.set_index("date")["close"].sort_index()
    if not series:
        raise FileNotFoundError(f"no readable price CSVs in {folder}")
    return pd.DataFrame(series).sort_index()


def load_leans() -> pd.DataFrame:
    p = ledger_paths.QLIB_SCORE_HISTORY
    if not p.exists():
        raise FileNotFoundError(f"score history missing: {p}")
    df = pq.read_table(p).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].str.upper()
    return df


def run() -> dict:
    leans = load_leans()
    prices = load_prices(set(leans["asset"].unique()))

    covered = sorted(set(leans["asset"]) & set(prices.columns))
    missing = sorted(set(leans["asset"]) - set(prices.columns))

    results, tables = {}, []
    for h in HORIZONS:
        score_col = f"score_{h}d"
        lean_col = f"lean_{h}d"
        # Two things are worth separating: the CONTINUOUS score (all the
        # information the model emits) and the DISCRETISED lean (+1/0/-1 top-k,
        # which is what downstream consumers actually read). A model can carry
        # signal that the top-k discretisation throws away, or vice versa.
        for kind, col in (("score", score_col), ("lean", lean_col)):
            if col not in leans.columns:
                continue
            sub = leans[["date", "asset", col]].rename(columns={col: "score"})
            rows = ic_table(sub, prices, horizons=(h,), score_col="score")
            key = f"{h}d_{kind}"
            results[key] = {"rows": rows, "summary": summarise(rows, h)}
            tables.append(format_table(
                rows, f"qlib {kind} (score_col={col}) -- horizon {h}d"))

    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "evidence": str(ledger_paths.QLIB_SCORE_HISTORY),
        "rows_in_evidence": int(len(leans)),
        "date_range": [str(leans["date"].min().date()), str(leans["date"].max().date())],
        "assets_covered": len(covered),
        "assets_missing_prices": missing,
        "results": results,
        "tables": tables,
        "caveats": [
            "refit_every=126 sessions and min_train_days=750 -> only ~14 fits per "
            "horizon over 7 years; calendar years are NOT independent draws.",
            "lean is a cross-sectional top-k rank: with topk=8 over 76 assets only "
            "8 assets can be +1 on any date, so most rows are 0 by construction.",
            "prices are qlib_lab/csv close (adjusted upstream); no survivorship "
            "correction is possible for a fixed present-day universe list.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = run()
    out = ledger_paths.JOURNAL / f"qlib_leans_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 90)
    print(f"{QUESTION_ID}: does the qlib daily lean predict forward returns?")
    print("=" * 90)
    print(f"evidence: {rep['rows_in_evidence']:,} point-in-time rows, "
          f"{rep['date_range'][0]} .. {rep['date_range'][1]}, "
          f"{rep['assets_covered']} assets with prices")
    if rep["assets_missing_prices"]:
        print(f"  no price CSV for: {', '.join(rep['assets_missing_prices'])}")
    for t in rep["tables"]:
        print()
        print(t)
    print("\nsummaries")
    for key, r in rep["results"].items():
        s = r["summary"]
        print(f"  {key:14s} IC={s['overall_rank_ic']} t_NW={s['overall_t_nw']} "
              f"hit={s['overall_hit_rate']} (base {s['overall_base_rate']}) "
              f"LS_net={s['overall_ls_net']} | years IC>0: "
              f"{s['years_ic_positive']}/{s['years_graded']}, "
              f"with t>2: {s['years_ic_pos_and_t_gt_2']}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
