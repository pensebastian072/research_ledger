# research_ledger

A pre-registered answer layer over a set of quantitative research repos.

Four separate research efforts — a macro GPU bench, a Qlib walk-forward bench, a
news-LLM signal, and a cross-asset lead–lag map — all answered the same question:
*does this signal predict returns?* Each wrote its verdict once into a JSON file
and never asked again. Stale verdicts accumulate, bars drift to fit results, and
nobody can say what the current state of the evidence is.

This repo fixes that. It **registers the questions before the graders run**,
grades them on a schedule, and publishes a single `ANSWERS.md` where every row is
**YES / NO / NO-DATA**, with the number and the cause attached.

## Design rules

These are the constraints that make the answers trustworthy:

1. **Read-only over every sibling repo.** This repo opens other repos' files for
   reading and writes only inside itself. It never edits another repo, never
   writes a flag another system consumes, and has no execution path at all.
2. **The pass bar is frozen at registration.** A question's pass condition is
   written into `questions.yaml` *before* its grader runs. A result never edits
   the bar it was tested against — moving a bar to fit a number is the exact
   failure mode this repo exists to catch.
3. **Every verdict carries a number and an as-of date.** A verdict without a
   number is a bug. `NO-DATA` is a legitimate, first-class answer; it is said
   out loud rather than guessed around.
4. **Staleness is reported, never hidden.** A 25-day-old FAIL must not read like
   today's FAIL.
5. **Never one pooled number.** Results are reported per-horizon and
   per-calendar-year, with both `n` and `n_clusters` beside every statistic.
   Same-date entries are one cluster, not *n* independent observations.
6. **Gate math is imported, never re-ported.** The overfit gate (Probability of
   Backtest Overfitting, Deflated Sharpe Ratio) comes from one canonical
   implementation rather than being copied per repo.
7. **Proposals are propose-only.** Anything under `proposals/` is human-gated.
   The repo never applies its own suggestion.

## What it found

The first full run answered three registered questions, and all three were
negative:

| Question | Verdict | Number |
|---|---|---|
| Does the Qlib Alpha158 lean predict returns? | **NO** | IC 0.008; hit rate below base rate |
| Does the news-LLM sentiment signal predict returns? | **NO** | 1-day IC −0.028 (mildly contrarian) |
| Is the cross-asset lead–lag map real? | **NO** | 100% attributable to a stale-NAV artifact in one ETF |

Publishing the NOs is the point. A research layer that only reports its wins is
not measuring anything.

## Layout

- `questions.yaml` — the registered questions and their frozen pass bars
- `graders/` — one grader per question
- `collect.py` — runs the graders, gathers evidence
- `gate.py` — re-exports the canonical PBO / Deflated-Sharpe gate
- `ANSWERS.md` — the published output; read this first
- `answers.json` — the same state, machine-readable
- `audit/`, `cost/`, `journal/` — run history and cost accounting
## Viewer

```bash
pip install flask
python -m ui.app        # http://127.0.0.1:8104
```

All 22 registered questions as cards, grouped into settled, provisional and
still-open, each showing the frozen pass bar, the verdict, the number behind it
and its as-of date.

**It works on a fresh clone.** The repo ships a committed snapshot of the ledger
at `ui/snapshot/answers.json`, so the real verdicts render without the sibling
repos the graders read. If you have run the graders, the viewer prefers the live
`answers.json`. The header says which, and how old it is.

Staleness is recomputed from `as_of` against the clock rather than read from the
`stale` field — that field records what was true when the file was written. The
shipped snapshot is about three weeks old and the page says so, because a
three-week-old NO must not read like today's NO.

Binds `127.0.0.1` only, exposes no POST route, and is wired to nothing that acts.

## Status

Research code, run against a private set of sibling repos. The graders expect
those repos to be present; without them, questions resolve to `NO-DATA` — which
is the intended, honest behaviour rather than an error.

See [`DISCLAIMER.md`](DISCLAIMER.md). Not investment advice. MIT licensed.

