# research_ledger — agent guide

The box's **answer layer**. Four efforts (`macro_gpu_lab`, `qlib_lab`, the news LLM in
`hq-trading-system` + `alpaca_gpu_lab`, and the cross-asset relationship map) all answer the
same question — *does this signal predict returns?* — but each wrote its verdict once into a
JSON and never re-asked. This repo pre-registers the questions, grades them, and publishes a
single `ANSWERS.md` where every row is **YES / NO / NO-DATA plus the number and the cause**.

Read `ANSWERS.md` for the current state. Read `questions.yaml` for what is being asked.

Skills: `quant-research-gate` before touching any grader or the gate, `win-quant-env` before
installs/git/PS/Task Scheduler, `paper-trading-guardrails` before anything that could be read
as a signal.

## Hard rules

1. **Read-only over every sibling repo.** This repo opens other repos' files for reading and
   writes only inside itself. It never edits, never writes a flag another repo consumes, never
   touches a broker, never enforces. There is no execution path here to protect — keep it
   that way.
2. **`yes_bar` is frozen at registration.** A question's pass condition is written into
   `questions.yaml` *before* its grader runs, and a result never edits the bar it was tested
   against. Moving a bar to fit a number is the whole failure mode this repo exists to catch.
3. **Every verdict carries a number and an as-of date.** A verdict with no number is a bug in
   this repo. `NO-DATA` is a legitimate, first-class answer — say it instead of guessing.
4. **Staleness is reported, never hidden.** A 25-day-old FAIL must not read like today's FAIL.
5. **Never one pooled number.** Per-horizon and per-calendar-year tables, with `n` *and*
   `n_clusters` beside every statistic. Same-date entries are one cluster, not n independent
   observations (the pooled-n=369 → 41-cluster lesson).
6. **Gate math is imported, never re-ported** — `macro_gpu_lab/macro_gpu_lab/validate.py` via
   `MACRO_GPU_LAB_DIR`, the same contract as `alpaca_gpu_lab/src/gate.py`. `gate.py` here
   re-exports it. The **one** exception is `deflated_sharpe_fixed` (see below), which exists
   *because* the canonical one is provably broken and is kept side-by-side rather than
   substituted.
7. **Proposals are propose-only.** Files in `proposals/` are human-gated, like `skill_forge`.
   This repo never applies its own suggestion.

## The DSR defect (read before trusting any historical FAIL)

`macro_gpu_lab.validate.deflated_sharpe` is **unit-inconsistent** and rejects everything by
construction. `e_max` is expressed in units of sigma_SR (the dispersion of the SR estimator)
but is subtracted from `sr` in raw per-observation SR units, and the difference is then scaled
by `sqrt(T-1)/denom` — which inflates the penalty by a factor of about `sqrt(T-1)` (14.5x at
T=210, 20.5x at T=421).

Proof, reproducible via `.venv\Scripts\python.exe -m audit.dsr_audit`:

- On **pure noise**, best-of-N trials, the shipped formula returns `prob = 0.0000` for
  N = 2, 8 and 42. A correct deflated Sharpe returns ~0.5 there by construction — deflation
  removes exactly the selection advantage, no more.
- Power check: the shipped formula passes **0.000** of cases even at a true per-observation
  SR of 0.30 (≈ Sharpe 4.8 annualised on daily data). It is a constant "FAIL", not a test.

`gate.py::deflated_sharpe_fixed` implements Bailey & López de Prado (2014) correctly, in two
variants: `sigma_sr="se"` (estimator standard error — the conservative default) and
`sigma_sr="cross"` (the faithful cross-trial SR dispersion, when the trial SRs are on hand).

**STATUS UPDATE 2026-08-07: the canonical `validate.py` HAS been patched.** The paragraph
above describes the defect as it stood before 2026-07-30 and is kept for the history, but
the proposal was accepted: `macro_gpu_lab/macro_gpu_lab/validate.py::deflated_sharpe` now
computes `sr_star = sigma_sr * bracket` and `ratio = psr_z - bracket`, documents the fix in
its own docstring, and pins the calibration with `test_deflated_sharpe_null_calibration`.

Verified rather than assumed: `graders/intl_reversal_portfolio.py` reports `dsr_fixed` and
`dsr_shipped` side by side and they now agree **exactly** (both 0.5747 on Q-REL-05). If they
ever diverge again, the canonical one has regressed.

Reporting both numbers side by side is still the right habit and costs nothing, so
`ANSWERS.md` keeps doing it — but a `dsr_shipped` value is no longer presumed wrong.

Corollary worth its own line: **`DSR ratio > 0` is a median test, not a significance test.**
It asks "better than the *expected* max of N noise trials", which a coin flip clears ~46% of
the time. If the threshold is ever revisited, `ratio > 1.645` (prob > 0.95) is the defensible
bar. That is a threshold-policy question — do not change it unilaterally either.

## Layout

```
questions.yaml   pre-registered questions + frozen yes_bar + status
collect.py       walks sibling scorecards -> answers.json + ANSWERS.md
gate.py          re-exports the canonical gate + deflated_sharpe_fixed
ledger_paths.py  every sibling path in one place (override via env)
graders/         lean_ic.py (core) + one module per question
audit/           dsr_audit.py — the proof above, re-runnable
proposals/       propose-only math/data upgrades, human-gated
journal/         this repo's dated outputs (committed)
```

## Commands

Always `.venv\Scripts\python.exe` — bare `python` is the Store stub on this box.

- Everything: `.venv\Scripts\python.exe collect.py --run-graders`
- Collect only (no grading): `.venv\Scripts\python.exe collect.py`
- One grader: `.venv\Scripts\python.exe -m graders.qlib_leans`
- DSR audit: `.venv\Scripts\python.exe -m audit.dsr_audit`
- Tests: `.venv\Scripts\python.exe -m pytest`

Graders must stay **offline and free**. Every input is already on disk; nothing here spends a
CBOE LiveVol point or hits a metered API. If a question needs paid data, it stays `NO-DATA`
and the requirement goes in `proposals/`.

## Environment

- uv-managed `.venv` (py3.12, numpy 2). Reinstall:
  `uv pip install --python .venv\Scripts\python.exe --native-tls -r requirements.txt`
- Sibling parquet written by `qlib_lab` (pandas 2.0 / numpy<2) is read through **pyarrow**;
  never import another repo's modules into this venv, only its data. The one deliberate
  exception is `macro_gpu_lab.validate`, which is pure math on `numpy` + stdlib.
- git: per-commit identity, commit to the default branch, retry `add`/`commit` (Norton locks
  `.git/objects`).
