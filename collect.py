"""Walk every sibling repo, join to questions.yaml, emit ANSWERS.md + answers.json.

This is the file that answers "what did we find out". It does two things:

  1. STATUS -- reads `questions.yaml` and renders one row per question: verdict, the
     number behind it, the frozen bar, and the follow-up. Verdicts come from the
     graders, which write dated JSON into `journal/`; nothing is computed here, so
     the table can never disagree with the evidence that produced it.
  2. FRESHNESS -- walks every sibling scorecard directory and flag file and reports
     staleness in days. A 25-day-old FAIL must not read like today's FAIL
     (CLAUDE.md hard rule 4), and "gated once, never re-gated" is invisible unless
     something looks for it.

    .venv\\Scripts\\python.exe collect.py [--run-graders] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

import ledger_paths
import scorecards

GRADERS = [
    ("audit.dsr_audit", "Q-GATE-01"),
    ("graders.qlib_leans", "Q-QLIB-01"),
    ("graders.news_ic", "Q-NEWS-01"),
    ("graders.news_vs_model", "Q-NEWS-02"),
    ("graders.macro_signal", "Q-MACRO-02"),
    ("graders.leadlag_remine", "Q-REL-01"),
]

_RESULT_KEY = re.compile(r"result_\d{4}_\d{2}_\d{2}")

VERDICT_ORDER = {"YES": 0, "YES-PROVISIONAL": 1, "NEWS-PROVISIONAL": 1,
                 "MIXED": 2, "NO-DATA": 3, "OPEN": 4, "NO": 5}


def latest_result(q: dict) -> tuple[str, str | None]:
    """Newest `result_YYYY_MM_DD` on a question, plus the date it came from.

    A re-run that OVERTURNS an earlier answer is written as a NEW dated key
    rather than by editing the old one, so the record keeps both. Selecting the
    key by a hardcoded date silently pins the table to whichever re-run happened
    to be current when the line was written -- which is exactly what went wrong:
    `result_2026_07_29` was hardcoded here, so 10 questions kept rendering their
    superseded text (Q-QLIB-02 kept publishing H07_vrp_spy at +2.660 for weeks
    after the 2026-07-30 re-run put it at -0.004).
    """
    dated = sorted(k for k in q if _RESULT_KEY.fullmatch(k))
    for k in q:
        if k.startswith("result") and k not in dated:
            # a prefix match would happily pick `result_2nd_pass` as "newest", and an
            # unpadded `result_2026_7_29` sorts AFTER `result_2026_11_05`. Neither is
            # silently tolerable in the file whose job is to be the honest record.
            print(f"  WARNING: ignoring malformed result key {k!r} on {q.get('id')}",
                  file=sys.stderr)
    if not dated:
        return "", None
    key = dated[-1]
    # only call it a re-run when something was actually superseded
    stamp = key[len("result_"):].replace("_", "-") if len(dated) > 1 else None
    return " ".join(str(q.get(key, "")).split()), stamp


def _unbool(v):
    """YAML 1.1 turns bare NO/YES into False/True -- the statuses here are strings.

    questions.yaml quotes them, but a hand-edit that drops the quotes must not
    silently invert a verdict, so coerce on the way in as well.
    """
    if v is True:
        return "YES"
    if v is False:
        return "NO"
    return v


def load_questions() -> dict:
    if not ledger_paths.QUESTIONS.exists():
        return {"meta": {}, "questions": []}
    doc = yaml.safe_load(ledger_paths.QUESTIONS.read_text(encoding="utf-8")) or {}
    for q in doc.get("questions") or []:
        q["status"] = _unbool(q.get("status"))
    return doc


def latest_journal(prefix: str) -> tuple[Path | None, dict | None]:
    files = sorted(ledger_paths.JOURNAL.glob(f"{prefix}_*.json"))
    if not files:
        return None, None
    p = files[-1]
    try:
        return p, json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return p, None


def run_graders() -> list[dict]:
    """Run each grader in-process-isolated so one failure cannot sink the rest."""
    out = []
    py = sys.executable
    for mod, qid in GRADERS:
        proc = subprocess.run([py, "-m", mod], cwd=str(ledger_paths.HERE),
                              capture_output=True, text=True)
        out.append({"module": mod, "question": qid, "returncode": proc.returncode,
                    "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:]})
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"  {mod:28s} {status}")
    return out


def build(questions: dict) -> dict:
    today = datetime.now(timezone.utc).date()
    qs = questions.get("questions") or []
    rows = []
    for q in qs:
        prefix = None
        ab = str(q.get("answered_by") or "")
        for mod, qid in GRADERS:
            if qid == q.get("id"):
                prefix = mod.split(".")[-1]
        jpath, jdata = latest_journal(prefix) if prefix else (None, None)
        result_text, result_date = latest_result(q)
        rows.append({
            "id": q.get("id"),
            "question": " ".join(str(q.get("question", "")).split()),
            "repo": q.get("repo"),
            "status": q.get("status", "OPEN"),
            "registration": q.get("registration"),
            "yes_bar": " ".join(str(q.get("yes_bar", "")).split()),
            "result": result_text,
            "result_as_of": result_date,
            "bar_review": " ".join(str(q.get("bar_review", "")).split()) or None,
            "note": " ".join(str(q.get("note", "")).split()) or None,
            "answered_by": ab,
            "evidence": q.get("evidence"),
            "evidence_journal": str(jpath.name) if jpath else None,
            "evidence_as_of": (jdata or {}).get("as_of"),
        })
    rows.sort(key=lambda r: (VERDICT_ORDER.get(r["status"], 9), r["id"]))

    now = datetime.now(timezone.utc)
    return {
        "generated_at": now.isoformat(),
        # `as_of` is a tz-aware ISO timestamp on purpose: HQ's _read_qlib_flag()
        # feeds it to datetime.fromisoformat and subtracts a tz-aware now(), so a
        # bare date here would raise, be swallowed, and mark the file permanently
        # stale. as_of_date is the human/render field.
        "as_of": now.isoformat(),
        "as_of_date": today.isoformat(),
        "stale": False,
        "status": "READ-ONLY",
        "enforce": "no",
        "promoted": False,
        "meta": questions.get("meta", {}),
        "answers": rows,
        "counts": {k: sum(1 for r in rows if r["status"] == k)
                   for k in sorted({r["status"] for r in rows})},
        "flags": scorecards.flag_staleness(today),
        "scorecard_freshness": scorecards.scorecard_freshness(today),
        "parse_notes": scorecards.PARSE_NOTES,
    }


BADGE = {"YES": "**YES**", "YES-PROVISIONAL": "YES·prov",
         "NEWS-PROVISIONAL": "NEWS·prov", "NO": "NO",
         "NO-DATA": "NO-DATA", "OPEN": "open", "MIXED": "mixed"}


def render_md(doc: dict) -> str:
    L = []
    L.append("# ANSWERS")
    L.append("")
    L.append(f"Generated by `collect.py` on {doc['as_of_date']}. Do not hand-edit -- edit "
             "`questions.yaml` and re-run.")
    L.append("")
    counts = ", ".join(f"{k} {v}" for k, v in doc["counts"].items())
    L.append(f"**{len(doc['answers'])} questions**: {counts}")
    L.append("")
    gc = (doc.get("meta") or {}).get("gate_caveat")
    if gc:
        L.append("> " + " ".join(str(gc).split()))
        L.append("")

    L.append("## Verdicts")
    L.append("")
    L.append("| id | question | verdict | the number |")
    L.append("|---|---|---|---|")
    for r in doc["answers"]:
        num = r["result"] or r["note"] or "not yet graded"
        if len(num) > 300:
            num = num[:297] + "..."
        L.append(f"| `{r['id']}` | {r['question']} | {BADGE.get(r['status'], r['status'])} "
                 f"| {num} |")
    L.append("")

    L.append("## Detail")
    L.append("")
    for r in doc["answers"]:
        L.append(f"### {r['id']} — {BADGE.get(r['status'], r['status'])}")
        L.append("")
        L.append(f"**{r['question']}**")
        L.append("")
        L.append(f"- repo: `{r['repo']}`")
        L.append(f"- frozen bar: {r['yes_bar']}")
        L.append(f"- registration: {r['registration']}")
        L.append(f"- answered by: `{r['answered_by']}`")
        if r["evidence"]:
            L.append(f"- evidence: `{r['evidence']}`")
        if r["evidence_journal"]:
            L.append(f"- graded output: `journal/{r['evidence_journal']}` "
                     f"(as of {r['evidence_as_of']})")
        if r["result"]:
            stamp = f" _(as re-run {r['result_as_of']})_" if r.get("result_as_of") else ""
            L.append(f"- **result**{stamp}: {r['result']}")
        if r["bar_review"]:
            L.append(f"- **bar review**: {r['bar_review']}")
        if r["note"]:
            L.append(f"- note: {r['note']}")
        L.append("")

    L.append("## Staleness")
    L.append("")
    L.append("Published flag files:")
    L.append("")
    L.append("| flag | as-of | age (days) | status | promoted |")
    L.append("|---|---|---|---|---|")
    for f in doc["flags"]:
        age = f["age_days"]
        age_s = "missing" if not f["exists"] else ("?" if age is None else str(age))
        L.append(f"| `{f['flag']}` | {f['as_of'] or '-'} | {age_s} | "
                 f"{f['status'] or '-'} | {f['promoted']} |")
    L.append("")
    L.append("Newest gate scorecard per repo — a single distinct date means "
             "*gated once, never re-gated*:")
    L.append("")
    L.append("| repo | scorecards | newest | age (days) | distinct dates |")
    L.append("|---|---|---|---|---|")
    for s in doc["scorecard_freshness"]:
        L.append(f"| `{s['repo']}` | {s['n_scorecards']} | {s['newest'] or '-'} | "
                 f"{s['age_days'] if s['age_days'] is not None else '-'} | "
                 f"{s['distinct_dates']} |")
    L.append("")
    if doc["parse_notes"]:
        L.append("## Parse notes")
        L.append("")
        for n in doc["parse_notes"]:
            L.append(f"- {n}")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-graders", action="store_true",
                    help="re-run every grader before collecting")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.run_graders:
        print("running graders")
        run_graders()
        print()

    doc = build(load_questions())
    ledger_paths.ANSWERS_JSON.write_text(json.dumps(doc, indent=2, default=str),
                                         encoding="utf-8")
    ledger_paths.ANSWERS_MD.write_text(render_md(doc), encoding="utf-8")

    if args.json:
        print(json.dumps(doc, indent=2, default=str))
        return 0

    print("=" * 78)
    print(f"ANSWERS as of {doc['as_of_date']}")
    print("=" * 78)
    for r in doc["answers"]:
        print(f"  {r['status']:<18s} {r['id']:<12s} {r['question'][:52]}")
    print()
    print("  counts: " + ", ".join(f"{k}={v}" for k, v in doc["counts"].items()))
    print("\n  staleness")
    for s in doc["scorecard_freshness"]:
        once = "  <-- gated once, never re-gated" if s["distinct_dates"] == 1 else ""
        print(f"    {s['repo']:<16s} newest {s['newest']} "
              f"({s['age_days']}d old, {s['distinct_dates']} distinct dates){once}")
    for f in doc["flags"]:
        if f["exists"] and f["age_days"] is not None and f["age_days"] > 3:
            print(f"    STALE FLAG {f['flag']} is {f['age_days']}d old")
    if doc["parse_notes"]:
        print("\n  parse notes:")
        for n in doc["parse_notes"]:
            print(f"    {n}")
    print(f"\nwrote {ledger_paths.ANSWERS_MD.name} and {ledger_paths.ANSWERS_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
