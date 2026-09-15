"""Q-NEWS-01: does the news LLM's sentiment score carry ANY forward information?

This is the question the box has never asked. Two attempts existed and neither
could answer it:

  * `hq-trading-system/analytics/veto_reconcile.py` grades the LLM's SIZE MULTIPLIER
    against realised trade PnL. It last ran 2026-07-05 with n=28 against min 30 and
    returned `"verdict": null`. `live_trades` froze when the webhook retired on
    2026-07-22, so it can never reach quorum -- that path is structurally dead.
  * `alpaca_gpu_lab` battery `B04_xgb_sentiment` reports n_trades = 0 on every
    single row. It asked "do sentiment-GATED trades profit" at threshold 0.6 and
    the filter fired nothing, so it tested the filter, not the signal.

Neither asked the prior question: is there information in the score at all? That
needs no trades, no broker, no threshold -- only the score and the forward return.
`alpaca_gpu_lab/market_data/scores/news_scores.parquet` holds 88,533 finbert-scored
headlines back to 2017, already computed, offline, free.

Point-in-time discipline
------------------------
A headline stamped 23:21 UTC cannot be traded into that day's close, which has
already printed. Every headline is therefore assigned to the first session whose
CLOSE is strictly after the headline timestamp, and the forward return is measured
from that close. This can only ever be conservative: it never uses a bar that
closed before the news existed.

    .venv\\Scripts\\python.exe -m graders.news_ic [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
from graders.lean_ic import format_table, ic_table, summarise  # noqa: E402

QUESTION_ID = "Q-NEWS-01"
HORIZONS = (1, 5, 21)
CONVICTION_BUCKETS = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.01)]


def load_bars() -> pd.DataFrame:
    """Wide close panel from the hive-partitioned 1Day bars."""
    root = ledger_paths.ALPACA_BARS_1DAY
    if not root.exists():
        raise FileNotFoundError(f"bars missing: {root}")
    series = {}
    for sym_dir in sorted(root.glob("symbol=*")):
        sym = sym_dir.name.split("=", 1)[1].upper()
        files = sorted(sym_dir.rglob("*.parquet"))
        if not files:
            continue
        frames = []
        for f in files:
            try:
                frames.append(pq.read_table(f, columns=["timestamp", "close"])
                              .to_pandas())
            except Exception:
                continue
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        # bar timestamp is the session start (04:00 UTC); the CLOSE is knowable
        # only at the end of that session. Keep the session date as the index and
        # remember the moment the close became available.
        df["session"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.dropna(subset=["close"]).drop_duplicates("session", keep="last")
        series[sym] = df.set_index("session")["close"].sort_index()
    if not series:
        raise FileNotFoundError(f"no readable bars under {root}")
    return pd.DataFrame(series).sort_index()


def load_news() -> pd.DataFrame:
    p = ledger_paths.ALPACA_NEWS_SCORES
    if not p.exists():
        raise FileNotFoundError(f"news scores missing: {p}")
    df = pq.read_table(p).to_pandas()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


CLOSE_HOUR_ET = 16          # a session's close is knowable at 16:00 America/New_York


def assign_sessions(news: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each headline to the first session whose close post-dates it.

    Implemented as a searchsorted against session close instants, so the mapping
    is exact and vectorised. A headline at 11:12 ET on a trading day maps to that
    day's close; one at 19:21 ET maps to the NEXT session's close.
    """
    et = news["created_at"].dt.tz_convert("America/New_York")
    # the instant each session's close becomes known, in ET, tz-naive for compare
    closes = pd.Series(sessions) + pd.Timedelta(hours=CLOSE_HOUR_ET)
    idx = np.searchsorted(closes.to_numpy(),
                         et.dt.tz_localize(None).to_numpy(), side="left")
    out = news.copy()
    valid = idx < len(sessions)
    out = out.loc[valid].copy()
    out["date"] = sessions.to_numpy()[idx[valid]]
    return out


def explode_symbols(news: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    n = news.copy()
    n["symbols"] = n["symbols"].fillna("")
    n = n.assign(asset=n["symbols"].str.split(","))
    n = n.explode("asset")
    n["asset"] = n["asset"].str.strip().str.upper()
    n = n[n["asset"].isin(universe) & (n["asset"] != "")]
    return n


def aggregate(news: pd.DataFrame) -> pd.DataFrame:
    """One score per (session, asset): mean finbert score + headline count."""
    g = news.groupby(["date", "asset"], as_index=False).agg(
        score=("score", "mean"), n_headlines=("score", "size"),
        high_impact=("high_impact", "max"))
    g["conviction"] = g["score"].abs()
    return g


def run() -> dict:
    bars = load_bars()
    news_raw = load_news()
    universe = set(bars.columns)

    news = explode_symbols(news_raw, universe)
    news = assign_sessions(news, bars.index)
    agg = aggregate(news)

    results, tables = {}, []

    # ── headline test: the raw mean score, every row ────────────────
    rows = ic_table(agg[["date", "asset", "score"]], bars, horizons=HORIZONS,
                    score_col="score", quantile=0.33)
    results["all"] = {"rows": rows,
                      "summary": {h: summarise(rows, h) for h in HORIZONS}}
    tables.append(format_table(rows, "news sentiment -- all headlines"))

    # ── by conviction: does a louder score predict better? ──────────
    for lo, hi in CONVICTION_BUCKETS:
        sub = agg[(agg["conviction"] >= lo) & (agg["conviction"] < hi)]
        if len(sub) < 200:
            continue
        r = ic_table(sub[["date", "asset", "score"]], bars, horizons=HORIZONS,
                     score_col="score", quantile=0.33, by_year=False)
        key = f"conviction_{lo:.1f}_{hi:.1f}"
        results[key] = {"rows": r, "summary": {h: summarise(r, h) for h in HORIZONS}}
        tables.append(format_table(r, f"news sentiment -- |score| in [{lo}, {hi})"))

    # ── high-impact subset, as flagged by the scorer ────────────────
    hi_imp = agg[agg["high_impact"].astype(bool)]
    if len(hi_imp) >= 200:
        r = ic_table(hi_imp[["date", "asset", "score"]], bars, horizons=HORIZONS,
                     score_col="score", quantile=0.33, by_year=False)
        results["high_impact"] = {"rows": r,
                                  "summary": {h: summarise(r, h) for h in HORIZONS}}
        tables.append(format_table(r, "news sentiment -- high_impact only"))

    return {
        "question": QUESTION_ID,
        "as_of": date.today().isoformat(),
        "evidence": str(ledger_paths.ALPACA_NEWS_SCORES),
        "headlines_scored": int(len(news_raw)),
        "headline_asset_pairs_in_universe": int(len(news)),
        "session_asset_rows": int(len(agg)),
        "date_range": [str(agg["date"].min())[:10], str(agg["date"].max())[:10]],
        "scorers": sorted(news_raw["scorer"].dropna().unique().tolist()),
        "assets": sorted(agg["asset"].unique().tolist()),
        "results": results,
        "tables": tables,
        "caveats": [
            "headlines map to the first session close AFTER the timestamp, so no "
            "bar that printed before the news existed is ever used.",
            "the score is finbert on the headline only -- not the body, and not the "
            "hq llm_macro_analyst prompt, which is a DIFFERENT model. This grades "
            "the alpaca_gpu_lab news pipeline; see graders.news_vs_model for hq's.",
            "Alpaca news history starts ~2015 but the scored parquet begins 2017 and "
            "is thin before 2020; the per-year table shows where it thickens.",
            "one asset can appear in many headlines per session -- rows are averaged "
            "per (session, asset), so a busy news day is one observation, not many.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"news_ic_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 90)
    print(f"{QUESTION_ID}: does the news sentiment score carry forward information?")
    print("=" * 90)
    print(f"evidence: {rep['headlines_scored']:,} scored headlines -> "
          f"{rep['headline_asset_pairs_in_universe']:,} headline-asset pairs -> "
          f"{rep['session_asset_rows']:,} (session, asset) rows")
    print(f"range {rep['date_range'][0]} .. {rep['date_range'][1]} | "
          f"scorers: {', '.join(rep['scorers'])} | {len(rep['assets'])} assets")
    for t in rep["tables"]:
        print()
        print(t)
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
