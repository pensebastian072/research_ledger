"""OHLC spread estimators -- MEASURED, AND FOUND UNUSABLE AT ETF SCALE. Read this.

The plan was to price the full 1996-2026 history for free from daily high/low, then
calibrate the level against a small real-quote sample. Testing the estimators against
a KNOWN simulated spread first -- before trusting either on real data -- killed that
idea outright, and the negative result is worth more than the module:

**Both estimators collapse to exactly zero below a spread/volatility resolution
floor, and every ETF on this box is below it.**

    daily vol   spread they can resolve   spread they get right (>0.9x)
    0.6%        >= ~0.2%  (20 bps)        >= ~1.0%
    1.2%        >= ~0.4%  (40 bps)        >= ~1.0%
    2.5%        >= ~1.0% (100 bps)        >= ~2.0%

Real ETF half-spreads run 0.5-20 bps (0.00005-0.002). That is one to two orders of
magnitude BELOW the floor, so on SPY these estimators do not return a small number,
they return 0.0 -- and a cost layer that reports zero cost is worse than the flat
5 bps guess it was meant to replace. `run()` therefore refuses to emit a cost vector
and reports the floor instead.

The mechanism is not a bug: both estimators infer the spread from the part of the
high-low range that volatility cannot explain. When the bounce is 1/30th of the daily
range, that residual is buried, `alpha` goes negative, and the paper's own truncation
sends it to zero. Corwin-Schultz (2012) and Abdi-Ranaldo (2017) were built for
individual equities with 50-200 bps spreads, which is exactly where the table above
shows them working.

**Consequence: the cost layer must come from real quotes.** See
`alpaca_gpu_lab/src/data/quotes_sample.py` and `cost/costs.py`. This module stays so
that nobody on this box retries the free shortcut -- `calibrate_floor()` regenerates
the table above on demand.

Every backtest here charges a flat `COST_PER_SIDE = 0.0005` (5 bps per side) to every
asset and nobody has ever measured the real number, which is the difference between
qlib's +9.0 bps gross 5-day spread being an edge or a loss, and decides the
international-ETF reversal outright. That question is still open -- just not
answerable this way.

Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High
and Low Prices". Two-day high-low ranges separate the spread from volatility:

    beta  = E[ (ln(H_t/L_t))^2 + (ln(H_t+1/L_t+1))^2 ]
    gamma = (ln(H_[t,t+1]/L_[t,t+1]))^2
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
            - sqrt(gamma / (3 - 2*sqrt(2)))
    S     = 2*(e^alpha - 1) / (1 + e^alpha)

Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily Close,
High, and Low Prices". Uses the gap between the close and the mid-range:

    eta_t = (ln H_t + ln L_t) / 2
    S^2   = 4 * E[ (c_t - eta_t) * (c_t - eta_t+1) ]

Both return a PROPORTIONAL round-trip spread; half-spread is S/2, which is what a
backtest charges per side. Both are noisy and can go negative on small samples --
negatives are clipped to zero and both are always reported side by side, because
where they disagree is exactly where neither should be trusted.

Neither captures market impact. This is quoted spread only, so it is a floor on
true cost, and every consumer must say so.

    .venv\\Scripts\\python.exe -m cost.spread_ohlc --report
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

FLAT_ASSUMPTION = 0.0005          # the per-side cost every repo currently charges
MIN_DAYS_PER_WINDOW = 12          # below this a monthly estimate is not reported
_K = 3.0 - 2.0 * np.sqrt(2.0)


def _log_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Corwin-Schultz overnight adjustment.

    When the whole range gaps overnight (today's low above yesterday's high, or the
    reverse) the two-day range is dominated by the jump, not by the spread. The
    paper shifts the earlier day's range onto the later one so gamma measures the
    same price level on both days.
    """
    h, l, c = df["high"].copy(), df["low"].copy(), df["close"]
    prev_c = c.shift(1)
    gap_up = l - prev_c
    gap_dn = prev_c - h
    up = gap_up > 0
    dn = gap_dn > 0
    h[up] = h[up] - gap_up[up]
    l[up] = l[up] - gap_up[up]
    h[dn] = h[dn] + gap_dn[dn]
    l[dn] = l[dn] + gap_dn[dn]
    return pd.DataFrame({"high": h, "low": l, "close": c}, index=df.index)


def corwin_schultz(df: pd.DataFrame, *, adjust_overnight: bool = True) -> float | None:
    """Proportional round-trip spread over the whole frame. None if unusable."""
    if len(df) < 3:
        return None
    d = _log_adjusted(df) if adjust_overnight else df
    h, l = d["high"].to_numpy(float), d["low"].to_numpy(float)
    ok = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0)
    h, l = h[ok], l[ok]
    if h.size < 3:
        return None
    hl = np.log(h / l) ** 2
    beta = hl[:-1] + hl[1:]                       # two consecutive single-day ranges
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    gamma = np.log(h2 / l2) ** 2
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    alpha = alpha[np.isfinite(alpha)]
    if alpha.size == 0:
        return None
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # The paper averages the per-pair spread and truncates negatives at the mean,
    # not per observation -- clipping first would bias the estimate upward.
    return float(max(0.0, np.nanmean(s)))


def abdi_ranaldo(df: pd.DataFrame) -> float | None:
    """Proportional round-trip spread from close vs mid-range. None if unusable."""
    if len(df) < 3:
        return None
    h, l, c = (df["high"].to_numpy(float), df["low"].to_numpy(float),
               df["close"].to_numpy(float))
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (l > 0)
    h, l, c = h[ok], l[ok], c[ok]
    if h.size < 3:
        return None
    eta = (np.log(h) + np.log(l)) / 2.0
    lc = np.log(c)
    prod = 4.0 * (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    prod = prod[np.isfinite(prod)]
    if prod.size == 0:
        return None
    return float(np.sqrt(max(0.0, float(np.nanmean(prod)))))


# ── loaders ─────────────────────────────────────────────────────────
def load_qlib_ohlc() -> dict[str, pd.DataFrame]:
    out = {}
    folder = ledger_paths.QLIB_PRICES
    if not folder.exists():
        return out
    for csv in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(csv, usecols=["date", "high", "low", "close"],
                             parse_dates=["date"])
        except Exception:
            continue
        if df.empty:
            continue
        out[csv.stem.upper()] = df.set_index("date").sort_index()
    return out


def load_alpaca_ohlc() -> dict[str, pd.DataFrame]:
    out = {}
    root = ledger_paths.ALPACA_BARS_1DAY
    if not root.exists():
        return out
    for sym_dir in sorted(root.glob("symbol=*")):
        sym = sym_dir.name.split("=", 1)[1].upper()
        frames = []
        for f in sorted(sym_dir.rglob("*.parquet")):
            try:
                frames.append(pq.read_table(
                    f, columns=["timestamp", "high", "low", "close"]).to_pandas())
            except Exception:
                continue
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["date"] = (pd.to_datetime(df["timestamp"], utc=True)
                      .dt.tz_convert("America/New_York").dt.normalize()
                      .dt.tz_localize(None))
        df = (df.dropna(subset=["close"]).drop_duplicates("date", keep="last")
              .set_index("date").sort_index())
        out[sym] = df[["high", "low", "close"]]
    return out


# ── monthly panel ───────────────────────────────────────────────────
def monthly_spreads(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in ohlc.items():
        for period, g in df.groupby(df.index.to_period("M")):
            if len(g) < MIN_DAYS_PER_WINDOW:
                continue
            cs = corwin_schultz(g)
            ar = abdi_ranaldo(g)
            rows.append({"symbol": sym, "month": str(period),
                         "n_days": int(len(g)),
                         "cs_spread": cs, "ar_spread": ar,
                         "cs_half": None if cs is None else cs / 2.0,
                         "ar_half": None if ar is None else ar / 2.0})
    return pd.DataFrame(rows)


def per_symbol_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    g = monthly.groupby("symbol")
    out = pd.DataFrame({
        "n_months": g.size(),
        "cs_half_median": g["cs_half"].median(),
        "ar_half_median": g["ar_half"].median(),
        "cs_half_p75": g["cs_half"].quantile(0.75),
        "ar_half_p75": g["ar_half"].quantile(0.75),
    })
    # The two estimators bracket the truth; the blend is their mean, and their
    # disagreement is the honest error bar on it.
    out["blend_half"] = out[["cs_half_median", "ar_half_median"]].mean(axis=1)
    out["disagreement"] = (out["cs_half_median"] - out["ar_half_median"]).abs()
    out["vs_flat_5bps"] = out["blend_half"] / FLAT_ASSUMPTION
    return out.sort_values("blend_half")


def calibrate_floor(vols=(0.006, 0.012, 0.025),
                    spreads=(0.0002, 0.0005, 0.001, 0.002, 0.004, 0.01, 0.02),
                    n_days: int = 2000) -> list[dict]:
    """Regenerate the resolution-floor table from simulation. Self-contained.

    Duplicated deliberately from tests/test_spread_ohlc.py::simulate so the module
    can prove its own limitation without importing a test.
    """
    out = []
    for vol in vols:
        for s in spreads:
            rng = np.random.default_rng(4)
            ticks = 90
            step = vol / np.sqrt(ticks)
            n = n_days * ticks
            mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * step))
            side = rng.choice([-1.0, 1.0], size=n)
            obs = (mid * (1.0 + side * s / 2.0)).reshape(n_days, ticks)
            df = pd.DataFrame({"high": obs.max(axis=1), "low": obs.min(axis=1),
                               "close": obs[:, -1]})
            cs, ar = corwin_schultz(df), abdi_ranaldo(df)
            out.append({"daily_vol": vol, "true_spread": s,
                        "cs": None if cs is None else round(cs, 8),
                        "ar": None if ar is None else round(ar, 8),
                        "cs_ratio": None if not cs else round(cs / s, 3),
                        "ar_ratio": None if not ar else round(ar / s, 3),
                        "resolved": bool((cs or 0) > 0 or (ar or 0) > 0)})
    return out


ETF_VOL_TYPICAL = 0.012          # the regime this box actually trades


def run(include_floor: bool = True) -> dict:
    ohlc = {**load_qlib_ohlc(), **load_alpaca_ohlc()}   # alpaca wins on overlap
    monthly = monthly_spreads(ohlc)
    summary = per_symbol_summary(monthly)

    floor = calibrate_floor() if include_floor else []
    at_typical = [r for r in floor if r["daily_vol"] == ETF_VOL_TYPICAL]
    resolvable = [r["true_spread"] for r in at_typical if r["resolved"]]
    floor_spread = min(resolvable) if resolvable else None

    # How many real symbols come back as a hard zero? That is the whole verdict.
    zeros = sum(1 for r in summary.reset_index().to_dict(orient="records")
                if not r["blend_half"] or r["blend_half"] <= 0)
    return {
        "as_of": date.today().isoformat(),
        "verdict": "UNUSABLE-AT-ETF-SCALE",
        "usable_as_cost_source": False,
        "flat_assumption_per_side": FLAT_ASSUMPTION,
        "resolution_floor_at_1p2pct_vol": floor_spread,
        "floor_table": floor,
        "n_symbols": int(len(summary)),
        "n_symbols_estimated_zero": zeros,
        "n_monthly_estimates": int(len(monthly)),
        "summary": summary.reset_index().to_dict(orient="records"),
        "caveats": [
            "THIS IS NOT A COST SOURCE. Both estimators return exactly 0.0 for any "
            "spread below their volatility-dependent resolution floor, and every ETF "
            "here is below it. Use cost/costs.py, which is quote-based.",
            "the floor is ~0.4% (40 bps) round-trip at 1.2% daily vol, versus real ETF "
            "half-spreads of 0.5-20 bps -- one to two orders of magnitude apart.",
            "the estimators are correct where they were designed to work: single "
            "equities with 50-200 bps spreads recover at 0.9-1.0x.",
            "quoted spread only in any case -- market impact is never included.",
            "the qlib CSVs are split/dividend adjusted, which compresses historical "
            "high-low ranges and would bias these estimates low even if they resolved.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = run()
    out = ledger_paths.JOURNAL / f"spread_ohlc_{rep['as_of']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print("=" * 88)
    print(f"OHLC spread estimators -- VERDICT: {rep['verdict']}")
    print("=" * 88)
    print("Resolution floor, from simulation with a KNOWN spread")
    print(f"  {'daily_vol':>9s} {'true':>8s} {'CS':>10s} {'CS/true':>8s} "
          f"{'AR':>10s} {'AR/true':>8s}  resolved")
    for r in rep["floor_table"]:
        cs_r = f"{r['cs_ratio']:>8.2f}" if r["cs_ratio"] else f"{'-':>8s}"
        ar_r = f"{r['ar_ratio']:>8.2f}" if r["ar_ratio"] else f"{'-':>8s}"
        print(f"  {r['daily_vol']:>9.3f} {r['true_spread']:>8.4f} "
              f"{(r['cs'] or 0):>10.6f} {cs_r} {(r['ar'] or 0):>10.6f} {ar_r}"
              f"  {'yes' if r['resolved'] else 'NO'}")
    print(f"\n  At the 1.2% daily vol these ETFs actually run, the smallest spread "
          f"either estimator can resolve is {rep['resolution_floor_at_1p2pct_vol']}.")
    print(f"  Real ETF half-spreads are 0.00005-0.002. The floor is 1-2 orders of "
          f"magnitude too high.")

    print(f"\nOn real data: {rep['n_symbols']} symbols, "
          f"{rep['n_monthly_estimates']:,} monthly estimates, "
          f"{rep['n_symbols_estimated_zero']} of {rep['n_symbols']} come back as "
          f"exactly ZERO cost.")
    print(f"\n  {'symbol':<8s} {'months':>6s} {'CS_half':>9s} {'AR_half':>9s} "
          f"{'blend':>9s} {'bps':>7s}")
    for r in rep["summary"][:12]:
        b = r["blend_half"] or 0.0
        print(f"  {r['symbol']:<8s} {r['n_months']:>6d} "
              f"{(r['cs_half_median'] or 0):>9.6f} {(r['ar_half_median'] or 0):>9.6f} "
              f"{b:>9.6f} {b*1e4:>7.2f}")
    print(f"  ... ({max(0, rep['n_symbols'] - 12)} more)")
    print("\n  A cost layer that reports 0 bps is worse than the flat 5 bps guess it "
          "was meant to replace.")
    print("  Cost must come from real quotes -- see cost/costs.py.")
    print("\ncaveats")
    for c in rep["caveats"]:
        print(f"  - {c}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
