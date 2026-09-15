"""The core measurement: does a published lean predict the forward return?

Every question in this repo funnels into `ic_table`. Feed it a long frame of
(date, asset, score) plus a wide price panel and it answers, per horizon and per
calendar year:

  rank_ic      cross-sectional Spearman IC, averaged over dates
  t_nw         Newey-West t-statistic on the daily IC series. Overlapping h-day
               labels make consecutive ICs autocorrelated, so an OLS t is
               inflated; the NW correction with lag h-1 is the minimum honest fix.
  hit_rate     sign(score) == sign(forward return), against the base rate on the
               SAME dates -- a 55% hit rate is worthless if the tape rose 55% of
               those days.
  ls_spread    top-quantile minus bottom-quantile mean forward return, gross and
               net of round-trip cost. Cost default 0.0005/side matches qlib_lab.
  n, n_clusters
               n is observations; n_clusters is DISTINCT DATES. Same-date entries
               are one draw, not many (the pooled-369 -> 41-cluster lesson). Every
               table prints both, and no verdict may cite n when n_clusters is small.

Deliberately NOT here: any promotion decision. This module measures; `questions.yaml`
holds the frozen bar and `collect.py` compares against it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_COST_PER_SIDE = 0.0005          # matches qlib_lab experiments
MIN_ASSETS_PER_DATE = 5                 # below this a cross-sectional IC is noise
MIN_DATES_FOR_IC = 20


@dataclass
class ICResult:
    horizon: int
    scope: str                          # "all" or a calendar year
    n: int
    n_clusters: int
    rank_ic: float | None = None
    ic_std: float | None = None
    t_nw: float | None = None
    t_ols: float | None = None
    hit_rate: float | None = None
    base_rate: float | None = None
    ls_spread_gross: float | None = None
    ls_spread_net: float | None = None
    n_long: int = 0
    n_short: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple h-step-ahead return per column. Index must be sorted dates."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return prices.shift(-horizon) / prices - 1.0


def newey_west_t(x: np.ndarray, lags: int) -> float | None:
    """t-stat of mean(x) with a Bartlett-kernel HAC variance. lags>=0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 8:
        return None
    mu = x.mean()
    dev = x - mu
    gamma0 = float(dev @ dev) / T
    var = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        cov = float(dev[L:] @ dev[:-L]) / T
        var += 2.0 * (1.0 - L / (lags + 1.0)) * cov
    if var <= 0:
        return None
    return float(mu / math.sqrt(var / T))


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation with average ranks for ties; None if degenerate."""
    if a.size < 3:
        return None
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _panel_for_horizon(scores: pd.DataFrame, prices: pd.DataFrame,
                       horizon: int, score_col: str) -> pd.DataFrame:
    fwd = forward_returns(prices, horizon)
    long_fwd = (fwd.stack(future_stack=True)
                .rename("fwd").reset_index())
    long_fwd.columns = ["date", "asset", "fwd"]
    merged = scores.merge(long_fwd, on=["date", "asset"], how="inner")
    merged = merged.dropna(subset=[score_col, "fwd"])
    return merged


def _one_scope(df: pd.DataFrame, horizon: int, scope: str, score_col: str,
               cost_per_side: float, quantile: float) -> ICResult:
    res = ICResult(horizon=horizon, scope=scope, n=int(len(df)),
                   n_clusters=int(df["date"].nunique()))
    if res.n == 0:
        res.notes.append("no overlapping (date, asset) rows")
        return res

    # ── cross-sectional IC per date ────────────────────────────────
    ics, spreads_g, spreads_n = [], [], []
    n_long = n_short = 0
    for _, g in df.groupby("date", sort=True):
        if len(g) < MIN_ASSETS_PER_DATE:
            continue
        s = g[score_col].to_numpy(float)
        f = g["fwd"].to_numpy(float)
        ic = _spearman(s, f)
        if ic is not None:
            ics.append(ic)
        # long/short on the score's own cross-sectional quantiles
        lo, hi = np.quantile(s, quantile), np.quantile(s, 1 - quantile)
        long_leg, short_leg = f[s >= hi], f[s <= lo]
        if long_leg.size and short_leg.size:
            gross = float(long_leg.mean() - short_leg.mean())
            spreads_g.append(gross)
            spreads_n.append(gross - 4.0 * cost_per_side)   # 2 legs, in+out
            n_long += int(long_leg.size)
            n_short += int(short_leg.size)

    if len(ics) >= MIN_DATES_FOR_IC:
        arr = np.asarray(ics)
        res.rank_ic = round(float(arr.mean()), 5)
        res.ic_std = round(float(arr.std(ddof=1)), 5)
        res.t_nw = newey_west_t(arr, lags=max(0, horizon - 1))
        res.t_ols = newey_west_t(arr, lags=0)
        if res.t_nw is not None:
            res.t_nw = round(res.t_nw, 3)
        if res.t_ols is not None:
            res.t_ols = round(res.t_ols, 3)
    else:
        res.notes.append(f"only {len(ics)} usable dates (need {MIN_DATES_FOR_IC})")

    # ── directional hit rate vs the base rate on the same rows ─────
    nz = df[df[score_col] != 0]
    if len(nz):
        hit = np.sign(nz[score_col]) == np.sign(nz["fwd"])
        res.hit_rate = round(float(hit.mean()), 4)
        res.base_rate = round(float((nz["fwd"] > 0).mean()), 4)
    else:
        res.notes.append("every score is zero -- no directional call to grade")

    if spreads_g:
        res.ls_spread_gross = round(float(np.mean(spreads_g)), 6)
        res.ls_spread_net = round(float(np.mean(spreads_n)), 6)
        res.n_long, res.n_short = n_long, n_short

    if res.n_clusters < MIN_DATES_FOR_IC:
        res.notes.append(
            f"n={res.n} but only {res.n_clusters} distinct dates -- treat as "
            f"{res.n_clusters} draws, not {res.n}")
    return res


def ic_table(scores: pd.DataFrame, prices: pd.DataFrame, horizons=(5, 21), *,
             score_col: str = "score", cost_per_side: float = DEFAULT_COST_PER_SIDE,
             quantile: float = 0.2, by_year: bool = True) -> list[dict]:
    """Full per-horizon, per-year IC table.

    scores : long frame with columns [date, asset, <score_col>]
    prices : wide frame, DatetimeIndex x asset columns (close, split-adjusted)
    """
    for col in ("date", "asset", score_col):
        if col not in scores.columns:
            raise ValueError(f"scores is missing column {col!r}")
    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    prices = prices.sort_index()

    rows: list[dict] = []
    for h in horizons:
        panel = _panel_for_horizon(scores, prices, h, score_col)
        rows.append(_one_scope(panel, h, "all", score_col, cost_per_side,
                               quantile).as_dict())
        if by_year and len(panel):
            for year, g in panel.groupby(panel["date"].dt.year, sort=True):
                rows.append(_one_scope(g, h, str(int(year)), score_col,
                                       cost_per_side, quantile).as_dict())
    return rows


def summarise(rows: list[dict], horizon: int) -> dict:
    """Collapse a horizon's per-year rows into the shape a yes_bar is checked on."""
    years = [r for r in rows if r["horizon"] == horizon and r["scope"] != "all"]
    overall = next((r for r in rows if r["horizon"] == horizon
                    and r["scope"] == "all"), None)
    graded = [r for r in years if r.get("rank_ic") is not None]
    pos_t2 = [r for r in graded if r["rank_ic"] > 0 and (r.get("t_nw") or 0) > 2]
    return {
        "horizon": horizon,
        "overall_rank_ic": (overall or {}).get("rank_ic"),
        "overall_t_nw": (overall or {}).get("t_nw"),
        "overall_hit_rate": (overall or {}).get("hit_rate"),
        "overall_base_rate": (overall or {}).get("base_rate"),
        "overall_ls_net": (overall or {}).get("ls_spread_net"),
        "n": (overall or {}).get("n"),
        "n_clusters": (overall or {}).get("n_clusters"),
        "years_graded": len(graded),
        "years_ic_positive": sum(1 for r in graded if r["rank_ic"] > 0),
        "years_ic_pos_and_t_gt_2": len(pos_t2),
        "years": [r["scope"] for r in graded],
    }


def format_table(rows: list[dict], title: str) -> str:
    """Fixed-width text table -- what lands in journal/ and ANSWERS.md."""
    out = [f"  {title}",
           f"  {'h':>3s} {'scope':>7s} {'rank_IC':>9s} {'t_NW':>7s} {'hit':>6s} "
           f"{'base':>6s} {'LS_net':>9s} {'n':>7s} {'clus':>5s}  notes"]
    for r in rows:
        def f(key, spec, width):
            v = r.get(key)
            return f"{'-':>{width}s}" if v is None else f"{v:>{width}{spec}}"
        out.append(
            f"  {r['horizon']:>3d} {r['scope']:>7s} {f('rank_ic','.5f',9)} "
            f"{f('t_nw','.2f',7)} {f('hit_rate','.4f',6)} {f('base_rate','.4f',6)} "
            f"{f('ls_spread_net','.6f',9)} {r['n']:>7d} {r['n_clusters']:>5d}  "
            f"{'; '.join(r.get('notes') or [])}")
    return "\n".join(out)
