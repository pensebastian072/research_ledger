"""Normalise every sibling repo's scorecard shape into one flat gate row.

Each lab grew its own JSON layout. This module is the only place that knows them,
so graders and the collector never re-parse. Every reader is fail-safe: a missing
directory yields no rows, a corrupt file is skipped with a note, nothing raises
(CLAUDE.md hard rule 3 -- a missing input is NO-DATA, not a crash).

Known shapes
------------
macro_gpu_lab/journal/scorecards/*.json
    horizons.<h>.{passes,deflated_sharpe{sr,ratio,n,n_trials},pbo,profit_factor,
                  n_trades,reasons}
    ... and for the surprise model, horizons.<h>.strategies.<name>.{same}
qlib_lab/journal/scorecards/qlib_lgbm_*.json      same horizons shape, + n_trials
qlib_lab/journal/scorecards/experiments*.json     battery.<name>.{same}, + lead_lag
qlib_lab/journal/scorecards/deep_history_*.json   results.<name>.{pf,verdict,gate{}}
alpaca_gpu_lab/journal/scorecards/*.json          combos[].{params,horizons.<h>{}}
alpaca_gpu_lab/journal/RESULTS.md                 markdown ledger (already flat)
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import ledger_paths

PARSE_NOTES: list[str] = []


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                     # corrupt / partial write / locked
        PARSE_NOTES.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None


def _date_from_name(path: Path) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _gate_row(node: dict, *, repo: str, label: str, horizon: str,
              as_of: str | None, n_trials_default=None) -> dict | None:
    if not isinstance(node, dict):
        return None
    ds = node.get("deflated_sharpe") or {}
    return {
        "repo": repo,
        "label": label,
        "horizon": horizon,
        "as_of": as_of,
        "passes": bool(node.get("passes")),
        "sr": ds.get("sr"),
        "dsr_ratio": ds.get("ratio"),
        "dsr_prob": ds.get("prob"),
        "n_trials": ds.get("n_trials", n_trials_default),
        "pbo": node.get("pbo"),
        "profit_factor": node.get("profit_factor"),
        "sharpe": node.get("sharpe"),
        # `or` would turn a genuine 0 into None, and "0 trades" is exactly the
        # finding for B04_xgb_sentiment -- the battery that graded a filter which
        # never fired. Keep the zero.
        "n_trades": (node["n_trades"] if node.get("n_trades") is not None
                     else ds.get("n")),
        "reasons": node.get("reasons") or [],
        "note": node.get("note"),
    }


def _walk_horizons(horizons: dict, *, repo, label, as_of, n_trials_default=None):
    """horizons.<h> is either a gate node or a container of `strategies`."""
    for h, node in (horizons or {}).items():
        if not isinstance(node, dict):
            continue
        strategies = node.get("strategies")
        if isinstance(strategies, dict):
            for sname, snode in strategies.items():
                row = _gate_row(snode, repo=repo, label=f"{label} :: {sname}",
                                horizon=h, as_of=as_of,
                                n_trials_default=n_trials_default)
                if row:
                    yield row
            # the container may ALSO carry detection stats worth keeping
            det = node.get("detection")
            if isinstance(det, dict):
                yield {"repo": repo, "label": f"{label} :: detection", "horizon": h,
                       "as_of": as_of, "passes": None, "sr": None, "dsr_ratio": None,
                       "dsr_prob": None, "n_trials": None, "pbo": None,
                       "profit_factor": None, "sharpe": None, "n_trades": None,
                       "reasons": [], "note": "detection-only",
                       "detection": det,
                       "lift_vs_base": node.get("lift_vs_calm_base")}
            continue
        row = _gate_row(node, repo=repo, label=label, horizon=h, as_of=as_of,
                        n_trials_default=n_trials_default)
        if row:
            if isinstance(node.get("detection"), dict):
                row["detection"] = node["detection"]
            for k in ("first_oos", "last_oos", "window_start"):
                if node.get(k):
                    row[k] = node[k]
            yield row


def macro_rows():
    for p in sorted(ledger_paths.MACRO_SCORECARDS.glob("*.json")):
        d = _load(p)
        if not d:
            continue
        as_of = _date_from_name(p) or d.get("data_through")
        yield from _walk_horizons(d.get("horizons"), repo="macro_gpu_lab",
                                  label=f"macro {d.get('model', p.stem)}", as_of=as_of)


def qlib_rows():
    sc = ledger_paths.QLIB_SCORECARDS
    for p in sorted(sc.glob("qlib_lgbm_*.json")):
        d = _load(p)
        if not d:
            continue
        yield from _walk_horizons(
            d.get("horizons"), repo="qlib_lab",
            label=f"qlib {d.get('model', 'lgbm')}",
            as_of=_date_from_name(p) or d.get("data_through"),
            n_trials_default=d.get("n_trials"))

    for p in sorted(sc.glob("experiments*.json")):
        d = _load(p)
        if not d:
            continue
        as_of = _date_from_name(p)
        n_tr = d.get("n_trials")
        for name, node in (d.get("battery") or {}).items():
            row = _gate_row(node, repo="qlib_lab", label=f"qlib {name}",
                            horizon="battery", as_of=as_of, n_trials_default=n_tr)
            if row:
                row["mean_pnl"] = node.get("mean_pnl")
                row["window_start"] = node.get("window_start")
                yield row

    for p in sorted(sc.glob("deep_history_*.json")):
        d = _load(p)
        if not d:
            continue
        as_of = _date_from_name(p)
        window = d.get("window")
        for name, node in (d.get("results") or {}).items():
            if not isinstance(node, dict) or "pf" not in node:
                continue
            g = node.get("gate") or {}
            yield {"repo": "qlib_lab", "label": f"qlib deep-history {name}",
                   "horizon": "replication", "as_of": as_of,
                   "passes": bool(g.get("passes")), "sr": None, "dsr_ratio": None,
                   "dsr_prob": None, "n_trials": None, "pbo": g.get("pbo"),
                   "profit_factor": node.get("pf"), "sharpe": None,
                   "n_trades": node.get("n"), "reasons": [],
                   "note": f"replication verdict={node.get('verdict')}",
                   "replication_verdict": node.get("verdict"),
                   "pf_recent": node.get("pf_2016_2026"),
                   "window_start": node.get("window_start"),
                   "deep_window": window}


def alpaca_rows():
    for p in sorted(ledger_paths.ALPACA_SCORECARDS.glob("*.json")):
        d = _load(p)
        if not d:
            continue
        as_of = _date_from_name(p) or (d.get("run_at") or "")[:10]
        bid = d.get("battery_id", p.stem)
        tf = d.get("timeframe")
        for i, combo in enumerate(d.get("combos") or []):
            params = combo.get("params") or {}
            tag = params.get("model", f"combo{i}")
            label = f"alpaca {bid}" + (f" [{tf}]" if tf else "") + f" :: {tag}"
            yield from _walk_horizons(combo.get("horizons"), repo="alpaca_gpu_lab",
                                      label=label, as_of=as_of)


RESULTS_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|(.+)\|\s*$")


def alpaca_results_md_rows():
    """The one already-flat ledger on the box -- parse it as-is."""
    p = ledger_paths.ALPACA_RESULTS_MD
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = RESULTS_ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 11:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None

        yield {"repo": "alpaca_gpu_lab", "label": f"alpaca {cells[1]}",
               "horizon": cells[2], "as_of": cells[0],
               "passes": cells[9].upper() == "PASS",
               "sr": num(cells[5]), "dsr_ratio": num(cells[6]),
               "dsr_prob": None, "n_trials": num(cells[8]), "pbo": num(cells[7]),
               "profit_factor": num(cells[4]), "sharpe": num(cells[5]),
               "n_trades": num(cells[3]), "reasons": [],
               "note": cells[10] if len(cells) > 10 else None,
               "source": "RESULTS.md"}


def iter_gate_rows(include_results_md: bool = False):
    """Every gate row on the box, normalised. Duplicates across sources are kept."""
    for gen in (macro_rows, qlib_rows, alpaca_rows):
        try:
            yield from gen()
        except Exception as exc:
            PARSE_NOTES.append(f"{gen.__name__}: {type(exc).__name__}: {exc}")
    if include_results_md:
        try:
            yield from alpaca_results_md_rows()
        except Exception as exc:
            PARSE_NOTES.append(f"alpaca_results_md_rows: {exc}")


def lead_lag_blocks():
    """Top-level `lead_lag` blocks from the qlib experiment scorecards."""
    for p in sorted(ledger_paths.QLIB_SCORECARDS.glob("experiments*.json")):
        d = _load(p)
        if not d or "lead_lag" not in d:
            continue
        ll = d["lead_lag"] or {}
        yield {"as_of": _date_from_name(p), "source": p.name,
               "n_tests": ll.get("n_tests"),
               "bonferroni_alpha": ll.get("bonferroni_alpha"),
               "survivors": ll.get("survivors") or []}


def flag_staleness(today: date | None = None) -> list[dict]:
    """Every published flag file with its age in days -- staleness is a verdict."""
    today = today or datetime.now(timezone.utc).date()
    out = []
    for name, path in ledger_paths.FLAG_FILES.items():
        if not path.exists():
            out.append({"flag": name, "path": str(path), "exists": False,
                        "as_of": None, "age_days": None, "status": "MISSING",
                        "promoted": None})
            continue
        d = _load(path) or {}
        raw = d.get("as_of") or d.get("updated") or d.get("generated_at")
        age = None
        if raw:
            try:
                stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                age = (today - stamp.date()).days
            except ValueError:
                pass
        out.append({"flag": name, "path": str(path), "exists": True,
                    "as_of": raw, "age_days": age,
                    "status": d.get("status"), "promoted": d.get("promoted"),
                    "enforce": d.get("enforce")})
    return out


def scorecard_freshness(today: date | None = None) -> list[dict]:
    """Newest scorecard per repo -- catches 'gated once, never re-gated'."""
    today = today or datetime.now(timezone.utc).date()
    out = []
    for repo, folder in (("macro_gpu_lab", ledger_paths.MACRO_SCORECARDS),
                         ("qlib_lab", ledger_paths.QLIB_SCORECARDS),
                         ("alpaca_gpu_lab", ledger_paths.ALPACA_SCORECARDS)):
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        dates = sorted(d for d in (_date_from_name(f) for f in files) if d)
        newest = dates[-1] if dates else None
        age = None
        if newest:
            age = (today - date.fromisoformat(newest)).days
        out.append({"repo": repo, "n_scorecards": len(files),
                    "newest": newest, "age_days": age,
                    "distinct_dates": len(set(dates))})
    return out
