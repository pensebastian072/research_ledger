"""Q-NEWS-02: when the news LLM and the macro brain disagree, who is right?

`hq-trading-system/analytics/model_vs_news.py` has logged one row per asset per
snapshot for 29 days, and its own docstring states the goal -- "so we can study,
over time, where the model and the headlines pull apart (and which one the tape
later proved right)". The second half was never implemented. No code anywhere grades
the outcome.

This does, from outside HQ so that repo stays read-only after its 2026-07-22
retirement. For every logged row it joins `model_sign` and `news_sign` to the
forward return of the mapped macro asset and reports each layer's hit rate,
overall and on the subset where they diverge -- which is the only subset where
listening to one over the other changes a decision.

It also measures a degeneracy found while surveying the logs: on 2026-07-29 the
day's summary was `agree 0, diverge 0, partial 8`, because the macro side emits
sign 0 for every asset in a chop regime. A comparison that almost never fires
cannot answer the question no matter how long it runs, so the "fires at all" rate
is reported as a first-class number.

    .venv\\Scripts\\python.exe -m graders.news_vs_model [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from graders.qlib_leans import load_prices  # noqa: E402

QUESTION_ID = "Q-NEWS-02"
HORIZONS = (1, 5)
MIN_DIVERGENCES = 30            # frozen in questions.yaml


def load_rows() -> pd.DataFrame:
    folder = ledger_paths.HQ_MODEL_VS_NEWS
    if not folder.exists():
        return pd.DataFrame()
    recs = []
    for p in sorted(folder.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("ts")
            regime = obj.get("regime")
            for row in obj.get("rows") or []:
                recs.append({
                    "ts": ts, "date": pd.Timestamp(str(ts)[:10]),
                    "regime": regime,
                    "asset": row.get("asset"),
                    "macro_asset": row.get("macro_asset"),
                    "model_sign": row.get("model_sign"),
                    "news_sign": row.get("news_sign"),
                    "verdict": row.get("verdict"),
                })
    return pd.DataFrame(recs)


def _hit(signs: np.ndarray, fwd: np.ndarray) -> float | None:
    mask = (signs != 0) & np.isfinite(fwd)
    if mask.sum() == 0:
        return None
    return round(float((np.sign(fwd[mask]) == signs[mask]).mean()), 4)


def grade(rows: pd.DataFrame, prices: pd.DataFrame) -> dict:
    if rows.empty:
        return {"verdict": "NO-DATA", "why": "no model_vs_news rows on disk"}

    # collapse intraday snapshots to one row per (date, asset): the modal sign
    def modal(s):
        c = Counter(s.dropna().tolist())
        return c.most_common(1)[0][0] if c else np.nan

    daily = (rows.groupby(["date", "asset", "macro_asset"], as_index=False)
             .agg(model_sign=("model_sign", modal),
                  news_sign=("news_sign", modal),
                  regime=("regime", modal),
                  snapshots=("verdict", "size")))

    out = {"n_daily_rows": int(len(daily)),
           "n_days": int(daily["date"].nunique()),
           "assets": sorted(daily["asset"].dropna().unique().tolist()),
           "horizons": {}}

    # how often does the comparison fire at all?
    both = (daily["model_sign"] != 0) & (daily["news_sign"] != 0)
    out["fires_rate"] = round(float(both.mean()), 4)
    out["model_zero_rate"] = round(float((daily["model_sign"] == 0).mean()), 4)
    out["news_zero_rate"] = round(float((daily["news_sign"] == 0).mean()), 4)
    out["n_fires"] = int(both.sum())

    for h in HORIZONS:
        recs = []
        for macro_asset, g in daily.groupby("macro_asset"):
            if macro_asset not in prices.columns:
                continue
            px = prices[macro_asset].dropna()
            fwd = px.shift(-h) / px - 1.0
            g = g.copy()
            g["fwd"] = g["date"].map(fwd)
            recs.append(g)
        if not recs:
            out["horizons"][h] = {"verdict": "NO-DATA",
                                  "why": "no mapped asset has prices"}
            continue
        j = pd.concat(recs, ignore_index=True).dropna(subset=["fwd"])
        ms = j["model_sign"].to_numpy(float)
        ns = j["news_sign"].to_numpy(float)
        fw = j["fwd"].to_numpy(float)
        div = (ms != 0) & (ns != 0) & (ms != ns)

        h_out = {
            "n_graded": int(len(j)),
            "model_hit_rate": _hit(ms, fw),
            "news_hit_rate": _hit(ns, fw),
            "base_rate": round(float((fw > 0).mean()), 4) if len(fw) else None,
            "n_divergences": int(div.sum()),
            "model_hit_on_divergence": _hit(ms[div], fw[div]) if div.sum() else None,
            "news_hit_on_divergence": _hit(ns[div], fw[div]) if div.sum() else None,
        }
        if h_out["n_divergences"] < MIN_DIVERGENCES:
            h_out["verdict"] = "NO-DATA"
            h_out["why"] = (f"only {h_out['n_divergences']} divergences "
                            f"(need {MIN_DIVERGENCES}). The comparison fires on "
                            f"{out['fires_rate']:.1%} of rows because the macro side "
                            f"is sign 0 on {out['model_zero_rate']:.1%} of them.")
        else:
            m, nw = h_out["model_hit_on_divergence"], h_out["news_hit_on_divergence"]
            h_out["verdict"] = "NEWS" if (nw or 0) > (m or 0) else "MODEL"
            h_out["why"] = f"on divergence: news {nw} vs model {m}"
        out["horizons"][h] = h_out
    return out


def run() -> dict:
    rows = load_rows()
    prices = load_prices() if not rows.empty else pd.DataFrame()
    graded = grade(rows, prices)
    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "evidence": str(ledger_paths.HQ_MODEL_VS_NEWS),
        "raw_rows": int(len(rows)),
        "result": graded,
        "caveats": [
            "model_vs_news never graded outcomes itself -- its docstring promised "
            "'which one the tape later proved right' and no code implements it. "
            "This grader supplies that, from outside HQ.",
            "intraday snapshots are collapsed to one modal sign per (date, asset) so "
            "a 5-minute loop does not inflate n by ~288x.",
            "the LLM behind these rows is hq's llm_macro_analyst prompt over "
            "veto_flag.json -- a DIFFERENT model from the finbert scores graded in "
            "graders.news_ic. Do not pool the two verdicts.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"news_vs_model_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    r = rep["result"]
    print("=" * 90)
    print(f"{QUESTION_ID}: when the news LLM and the macro brain disagree, who wins?")
    print("=" * 90)
    if r.get("verdict") == "NO-DATA":
        print(f"NO-DATA: {r.get('why')}")
        print(f"\nwrote {out}")
        return 0
    print(f"evidence: {rep['raw_rows']:,} logged rows -> {r['n_daily_rows']} daily "
          f"(date, asset) rows over {r['n_days']} days")
    print(f"comparison fires (both sides non-zero): {r['fires_rate']:.1%} "
          f"({r['n_fires']} rows) | macro sign==0 on {r['model_zero_rate']:.1%}, "
          f"news sign==0 on {r['news_zero_rate']:.1%}")
    for h, hh in r["horizons"].items():
        print(f"\n  horizon {h}d  (n_graded={hh.get('n_graded')})")
        print(f"    model hit {hh.get('model_hit_rate')} | news hit "
              f"{hh.get('news_hit_rate')} | base {hh.get('base_rate')}")
        print(f"    divergences: {hh.get('n_divergences')} -> "
              f"model {hh.get('model_hit_on_divergence')} vs news "
              f"{hh.get('news_hit_on_divergence')}")
        print(f"    verdict: {hh.get('verdict')} -- {hh.get('why')}")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
