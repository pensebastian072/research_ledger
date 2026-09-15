"""One cost vector for the whole box: half_spread(symbol) in proportional terms.

Replaces the flat `COST_PER_SIDE = 0.0005` that every backtest here charges to every
asset. That number was never measured, and measuring it changes conclusions:

    measured (Alpaca IEX, 12 sampled sessions x 3 intraday windows)
      SPY  0.23 bps   QQQ 0.34 bps   IWM 0.39 bps   TLT 0.57 bps
      EEM  0.87 bps   GLD 2.56 bps   VIXY 3.53 bps  CPER 5.75 bps
      median across 21 symbols: 0.93 bps  =  0.19x the flat assumption
      20 of 21 symbols are CHEAPER than the flat 5 bps

Two sources, in priority order:

1. MEASURED -- `alpaca_gpu_lab/journal/quote_spreads_*.json`, produced by
   `alpaca_gpu_lab/src/data/quotes_sample.py` from real quotes. IEX is one venue with
   a small share of consolidated volume, so these read WIDER than true NBBO: every
   measured cost here is an OVERSTATEMENT, and an edge that survives it survives
   reality.
2. PREDICTED -- qlib's universe is 76 ETFs and only 21 are measured, so the rest are
   predicted from a log-log fit of measured half-spread on median dollar volume.
   Liquidity explains spread well across two orders of magnitude, but a prediction is
   not a measurement: `source` is always returned alongside the number, and the fit
   quality is printed with every report.

The free alternative was tried and rejected on evidence, not taste: see
`cost/spread_ohlc.py`, which proves the Corwin-Schultz and Abdi-Ranaldo high-low
estimators return exactly 0.0 for any spread below ~40 bps at ETF volatility -- one
to two orders of magnitude above what these ETFs quote.

What this is NOT: market impact. It is quoted half-spread only, so it is a FLOOR on
true cost. A strategy whose edge is a small multiple of the quoted spread is not
proven tradable at size by anything in this module.

    .venv\\Scripts\\python.exe -m cost.costs --report
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402

FLAT_ASSUMPTION = 0.0005          # what every repo currently charges, per side
FALLBACK_HALF = 0.0010            # 10 bps: used only when even ADV is unknown
MIN_HALF = 0.00002                # 0.2 bps floor -- tighter than SPY is not credible
MAX_HALF = 0.0030                 # 30 bps ceiling on a PREDICTED value
# The liquidity fit is estimated on US equity ETFs; crypto is out of domain (its
# dollar volume is not comparable and it quotes far wider), so it takes a flat,
# deliberately pessimistic fallback rather than an extrapolated prediction.
CRYPTO_SYMBOLS = frozenset({"BTC", "BTCUSD", "BTC/USD", "ETH", "ETHUSD", "ETH/USD"})
CRYPTO_FALLBACK_HALF = 0.0005     # 5 bps per side


@dataclass(frozen=True)
class Cost:
    symbol: str
    half_spread: float            # proportional, per side
    source: str                   # "measured" | "predicted" | "fallback"
    stress_half: float | None = None   # p90 within-session, when measured
    n_sessions: int | None = None

    def round_trip(self, legs: int = 1) -> float:
        """In-and-out cost for `legs` legs: 2 crossings per leg."""
        return 2.0 * legs * self.half_spread


def _newest_quote_report() -> dict | None:
    folder = ledger_paths.ALPACA_LAB / "journal"
    if not folder.exists():
        return None
    files = sorted(folder.glob("quote_spreads_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _median_dollar_volume() -> dict[str, float]:
    """Median daily dollar volume per symbol, from whatever bars are on disk."""
    out: dict[str, float] = {}
    folder = ledger_paths.QLIB_PRICES
    if folder.exists():
        for csv in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(csv, usecols=["close", "volume"])
            except Exception:
                continue
            dv = (df["close"] * df["volume"]).replace(0, np.nan).dropna()
            if len(dv) >= 100:
                out[csv.stem.upper()] = float(dv.median())
    root = ledger_paths.ALPACA_BARS_1DAY
    if root.exists():
        for sym_dir in sorted(root.glob("symbol=*")):
            sym = sym_dir.name.split("=", 1)[1].upper()
            if sym in out:
                continue
            vals = []
            for f in sorted(sym_dir.rglob("*.parquet")):
                try:
                    t = pq.read_table(f, columns=["close", "volume"]).to_pandas()
                    vals.append((t["close"] * t["volume"]).dropna())
                except Exception:
                    continue
            if vals:
                s = pd.concat(vals)
                if len(s) >= 100:
                    out[sym] = float(s.median())
    return out


@lru_cache(maxsize=1)
def _model() -> dict:
    """Build the measured table and fit log(half) ~ a + b*log(dollar ADV)."""
    rep = _newest_quote_report()
    measured: dict[str, dict] = {}
    if rep:
        for r in rep.get("per_symbol") or []:
            sym = str(r.get("symbol", "")).upper()
            hm = r.get("half_median")
            if sym and hm and hm > 0:
                measured[sym] = {"half": float(hm),
                                 "stress": float(r.get("half_p90") or hm),
                                 "n_sessions": int(r.get("n_sessions") or 0)}
    adv = _median_dollar_volume()

    xs, ys, used = [], [], []
    for sym, m in measured.items():
        if sym in adv and adv[sym] > 0 and m["n_sessions"] >= 3:
            xs.append(np.log(adv[sym]))
            ys.append(np.log(m["half"]))
            used.append(sym)
    fit = None
    if len(xs) >= 6:
        x = np.asarray(xs)
        y = np.asarray(ys)
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        fit = {"slope": float(b), "intercept": float(a),
               "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
               "n": len(xs), "symbols": used,
               "resid_sd_log": float(np.std(y - pred, ddof=2))}
    return {"measured": measured, "adv": adv, "fit": fit,
            "as_of": (rep or {}).get("as_of"), "feed": (rep or {}).get("feed")}


def half_spread(symbol: str, date_: date | None = None, *,
                stress: bool = False) -> Cost:
    """Per-side proportional half-spread for `symbol`.

    `date_` is accepted for a future time-varying table and is deliberately IGNORED
    today -- the quote sample gives a per-symbol LEVEL, not a daily series. Callers
    must not read a date-dependent cost into this.
    `stress=True` returns the within-session p90 where measured, which is the number
    to use before believing an edge that only clears at median cost.
    """
    sym = symbol.upper()
    m = _model()
    if sym in m["measured"]:
        d = m["measured"][sym]
        return Cost(sym, d["stress"] if stress else d["half"], "measured",
                    stress_half=d["stress"], n_sessions=d["n_sessions"])
    fit, adv = m["fit"], m["adv"]
    if sym in CRYPTO_SYMBOLS:
        # The liquidity fit is estimated on US equity ETFs only. Extrapolating it to
        # crypto returns ~0.2 bps for BTC, which is nonsense -- crypto quotes wide and
        # its "dollar volume" is not comparable to an ETF's. Out of domain: fall back.
        return Cost(sym, CRYPTO_FALLBACK_HALF, "fallback",
                    stress_half=CRYPTO_FALLBACK_HALF * 3)
    if fit and sym in adv and adv[sym] > 0:
        pred = float(np.exp(fit["intercept"] + fit["slope"] * np.log(adv[sym])))
        pred = float(min(MAX_HALF, max(MIN_HALF, pred)))
        # a predicted value carries the fit's own scatter; stress widens by 1 sd
        stress_v = float(min(MAX_HALF, pred * np.exp(fit["resid_sd_log"])))
        return Cost(sym, stress_v if stress else pred, "predicted",
                    stress_half=stress_v)
    return Cost(sym, FALLBACK_HALF, "fallback", stress_half=FALLBACK_HALF)


def cost_table(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        c = half_spread(s)
        rows.append({"symbol": c.symbol, "half_spread": c.half_spread,
                     "bps": c.half_spread * 1e4, "source": c.source,
                     "stress_bps": (c.stress_half or np.nan) * 1e4,
                     "n_sessions": c.n_sessions,
                     "vs_flat_5bps": c.half_spread / FLAT_ASSUMPTION})
    return pd.DataFrame(rows).sort_values("half_spread")


def universe() -> list[str]:
    m = _model()
    return sorted(set(m["measured"]) | set(m["adv"]))


def run() -> dict:
    m = _model()
    syms = universe()
    tbl = cost_table(syms)
    by_src = tbl.groupby("source")["bps"].agg(["count", "median"]).to_dict("index")
    return {
        "as_of": date.today().isoformat(),
        "quote_report_as_of": m["as_of"],
        "feed": m["feed"],
        "flat_assumption_per_side": FLAT_ASSUMPTION,
        "fit": m["fit"],
        "n_symbols": len(syms),
        "by_source": by_src,
        "median_bps": float(tbl["bps"].median()),
        "median_vs_flat": float(tbl["vs_flat_5bps"].median()),
        "n_cheaper_than_flat": int((tbl["vs_flat_5bps"] < 1).sum()),
        "table": tbl.to_dict(orient="records"),
        "caveats": [
            "QUOTED half-spread only -- market impact is not measured. This is a FLOOR "
            "on true cost, and an edge that is a small multiple of it is not proven "
            "tradable at size.",
            "measured values come from the IEX feed (one venue, small volume share) so "
            "they OVERSTATE true NBBO spreads -- a conservative direction for a cost "
            "test.",
            "predicted values are a log-log fit on dollar volume, not measurements; "
            "the `source` column says which is which and must be carried into any "
            "verdict that uses them.",
            "this is a per-symbol LEVEL from 12 sampled sessions in 2025-2026. It is "
            "not a daily series and it does not price 2020-style stress -- use "
            "stress=True (p90) before trusting a marginal edge.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = run()
    out = ledger_paths.JOURNAL / f"costs_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 84)
    print("THE COST VECTOR -- what a side actually costs, vs the flat 5 bps assumed")
    print("=" * 84)
    f = rep["fit"]
    if f:
        print(f"liquidity fit: log(half) = {f['intercept']:.3f} "
              f"+ {f['slope']:.4f} * log($ADV)   R2={f['r2']:.3f}  n={f['n']}  "
              f"resid_sd={f['resid_sd_log']:.3f} (log)")
    else:
        print("no liquidity fit -- too few measured symbols; predictions unavailable")
    print(f"quote report: {rep['quote_report_as_of']} ({rep['feed']} feed)")
    print(f"\n  {'symbol':<8s} {'bps':>7s} {'stress':>8s} {'x flat':>7s}  source")
    for r in rep["table"]:
        print(f"  {r['symbol']:<8s} {r['bps']:>7.2f} {r['stress_bps']:>8.2f} "
              f"{r['vs_flat_5bps']:>7.2f}  {r['source']}")
    print(f"\n  {rep['n_symbols']} symbols | median {rep['median_bps']:.2f} bps = "
          f"{rep['median_vs_flat']:.2f}x the flat assumption | "
          f"{rep['n_cheaper_than_flat']} cheaper than flat")
    for src, agg in rep["by_source"].items():
        print(f"    {src:<10s} n={int(agg['count']):>3d}  median {agg['median']:.2f} bps")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
