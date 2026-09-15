# PROPOSAL: what to change about the math and the data

**Status: PROPOSED — propose-only, human-gated. Ranked by evidence-per-hour.**
Companion to `2026-07-29-dsr-unit-bug.md`, which is the one that matters most.

Everything here is drawn from the first full run of the graders, not from taste.

---

## 1. Stop re-testing the daily horizon. It is exhausted. (free)

Four repos, ~250 configurations, one conclusion. `Q-QLIB-01` now supplies the missing
*why*: the qlib cross-sectional lean has rank IC **+0.008** at 5d (t 1.07) and **+0.002**
at 21d (t 0.12), and its hit rate (51.6%) is **below the base rate** (55.8%). There is no
signal there to gate, so no gate change and no model change will rescue it.

`alpaca_gpu_lab` rule 7 already says the near-misses may only re-enter as intraday
regime-context features. This makes that promise checkable: any new daily-horizon
battery must first register a question here stating what it expects to find that the
existing IC table rules out.

## 2. Charge `n_trials` per family, and write the policy down (free, but a decision)

`n_trials` is currently a cumulative count that differs per repo — 4 for qlib_lgbm, 8
for the macro surprise strategies, 20–42 across qlib batteries, up to 132 in the alpaca
ledger. Under a *correct* DSR the penalty is `sigma_SR * bracket(N)`, and `bracket`
grows slowly (1.46 at N=8, 2.21 at N=42, 2.63 at N=132) — so the choice of family
boundary is now a real, visible modelling decision rather than a rounding detail.

Two defensible policies:

- **per-family** (recommended): N = the number of configurations searched *for that
  hypothesis*. Matches the paper's intent — you deflate by what you tried on this idea.
- **cumulative box-wide**: N = every evaluation ever run. Maximally conservative, and
  arguably right if hypotheses are cherry-picked from one big pool.

Pick one, record it in `questions.yaml:meta`, and apply it consistently. Right now the
box does neither on purpose, which makes cross-repo DSR numbers non-comparable.

## 3. Grade the news LLM on *magnitude*, not direction (cheap, data already local)

`Q-NEWS-01` is a clean NO on direction: 1d IC **−0.028** (t −2.38, i.e. mildly
*contrarian*), 5d +0.0003, 21d +0.007, and no conviction bucket or `high_impact` subset
improves it. But the same finding in `Kronos` is on record — *"magnitude has a small
stable effect, direction is dead"* — and the news pipeline has never been tested that way.

Proposed: regress **|forward return|** and realised-vol changes on `|sentiment|` and
headline count, using the 88,533 scored headlines already on disk. Zero data cost. If
news predicts *how much* rather than *which way*, the natural consumer is the
options_desk vol lane, not a directional signal. Register as `Q-NEWS-04` before running.

## 4. Give `model_vs_news` something to compare (cheap, unblocks a dead question)

`Q-NEWS-02` fired on only **53.9%** of rows, because the news side is sign 0 on 40.1%
and the macro side on 11.7% — in a chop regime the macro brain emits 0 for everything, so
2026-07-29 logged `agree 0, diverge 0, partial 8`. Twenty-nine days produced 43 usable
divergences, and `Q-NEWS-03` needs ~120 days at that rate.

Two options, both outside the trade path:
- log the macro brain's **continuous** score alongside its sign, so a weak-but-nonzero
  lean still contributes; or
- lower `NEWS_CONVICTION_MIN` (0.35) *for logging only*, keeping the enforcement
  threshold untouched.

Either roughly doubles the fire rate and halves the wait. Requires an HQ edit, which
needs its own decision given HQ's read-only charter.

## 5. Re-gate on a schedule, or admit the verdict is a snapshot (cheap)

`macro_gpu_lab` has **one** distinct scorecard date, 2026-07-04, 26 days stale, while its
daily task publishes a signal that has never been graded (`Q-MACRO-02`: n=11, NO-DATA).
`qlib_lab` re-gates daily and is 1 day old. The asymmetry is invisible unless something
looks — which `collect.py` now does, and prints as *"gated once, never re-gated"*.

Proposed: extend the existing `MacroGpuDailyExport` task to run `train --once` weekly, or
state in `macro_gpu_lab/CLAUDE.md` that its verdicts are a 2026-07-04 snapshot.

## 6. The relationship map needs a mechanism test, not more mining (free)

`Q-REL-01` is settled: of 25 reported lead-lag survivors, thirds-validation leaves 5, and
**all 5 target INDA** — a US-listed India ETF whose underlying market is shut during US
hours. Excluding non-US-underlying targets leaves **0 of 20,832 tests**. The map found
microstructure and labelled it macro.

Mining harder will find more of the same. What is worth testing is the one relationship
with a mechanism and cross-era replication: copper/gold → cyclicals (`Q-REL-02`, PF 2.30
→ 1.36, **n=48**). n=48 is the entire problem and only history fixes it.

**The concrete blocker:** the qlib CSVs start 2016-08. `H10`'s deep-history window began
2011-11 from a different pull. Extending copper/gold/IWM to 1995 via the existing free
Stooq mirrors in `macro_gpu_lab/config.py` would take n from 48 toward ~200 and make the
question answerable. That is the single highest-value data acquisition on this list, and
it costs nothing but a fetch.

## 7. Report cluster counts everywhere, and set bars in clusters (free)

`Q-NEWS-02` cleared a 30-divergence bar on its first run and the bar was **wrong** — 43
divergences drawn from 29 calendar days, with 5d windows overlapping so heavily that
roughly 6 independent blocks exist. The result is logged as PROVISIONAL with the bar
review attached, and `Q-NEWS-03` carries the honest test, per hard rule 2.

Lesson worth generalising: **state every future bar in independent clusters, not rows.**
`lean_ic.py` already reports `n_clusters` beside `n` and annotates when they diverge;
`questions.yaml` should be held to the same standard.

## 8. Things deliberately NOT proposed

- **Softening any threshold.** The DSR proposal argues the opposite — that the honest bar
  is *higher* (1.645), not lower.
- **Re-running the qlib batteries before the DSR decision.** Re-running them under a
  broken ruler would just reproduce 40 uninformative FAILs.
- **Any paid data.** Every grader here is offline and free. The CBOE LiveVol archive is
  metered and untouched by this repo.
- **Wiring any of this into a trade path.** `research_ledger` writes one JSON that a
  read-only console renders. That is its whole surface area, and it should stay that way.
