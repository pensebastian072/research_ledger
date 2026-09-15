# PROPOSAL: fix the unit bug in the canonical Deflated Sharpe

**Status: PROPOSED — needs a human decision. Nothing has been patched.**
**Question:** Q-GATE-01 · **Evidence:** `journal/dsr_audit_2026-07-29.json`
**Reproduce:** `.venv\Scripts\python.exe -m audit.dsr_audit`

## Why this is not a request to soften the gate

The box's standing rule is that everything ships SHADOW until PBO < 0.5 **and**
deflated Sharpe > 0, that nothing has ever cleared it, and that the honesty is the
point. This proposal does not touch a threshold. It says the **ruler is miscalibrated**,
and asks that it be fixed so that the rule finally means what it claims.

The test that decides it is not an opinion. A deflated Sharpe is *defined* to remove
the selection advantage of having searched N configurations — no more and no less. So
if you generate N trials of pure zero-mean noise and submit the best one, a correct
implementation must return a probability near 0.5. Ours returns **0.0000**, for
N = 2, 8 and 42 alike.

| best-of-N pure noise, T=210 | shipped | fixed (SE) | fixed (cross-trial) |
|---|---|---|---|
| N=2  | 0.0000 | 0.518 | 0.558 |
| N=8  | 0.0000 | 0.484 | 0.505 |
| N=42 | 0.0000 | 0.488 | 0.492 |

And it has no power at all — it does not pass a *planted* edge either:

| planted true per-obs SR | ≈ annualised | shipped passes | fixed passes |
|---|---|---|---|
| 0.00 | 0.0 | 0.000 | 0.448 |
| 0.05 | 0.8 | 0.000 | 0.535 |
| 0.15 | 2.4 | 0.000 | 0.859 |
| 0.30 | 4.8 | 0.000 | 0.999 |

A test that rejects a Sharpe-4.8 strategy with probability 1 is not a strict test. It
is a constant. **Every `deflated_sharpe_ratio <= 0` FAIL on this box carries no
information about the strategy it rejected.**

## The defect

`macro_gpu_lab/macro_gpu_lab/validate.py:107` (`deflated_sharpe`), the `n_trials > 1`
branch:

```python
e_max = ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n)
         + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n * math.e)))
ratio = (sr - e_max) * math.sqrt(T - 1) / denom
```

Bailey & López de Prado (2014) define

```
DSR = PSR(SR*) = Phi[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) ]
SR* = E[max SR] = sigma_SR * [ (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N e)) ]
```

The bracketed term is **dimensionless** — a count of `sigma_SR`s. It must be
multiplied by `sigma_SR` before being subtracted from `SR`. The code subtracts the
bracket itself, in raw per-observation SR units, and then scales the whole difference
by `sqrt(T-1)/denom`. Net effect: the penalty is inflated by roughly `sqrt(T-1)`.

Concretely, at N=8 the bracket is 1.459. Subtracting 1.459 from a **per-observation**
Sharpe demands a per-trade SR of 1.46 — an annualised Sharpe near 23 — before the term
alone is cleared. Observed inflation factors across the stored scorecards run from
2.9x to **111x**.

The `n_trials == 1` branch is correct (it is just the PSR z-score), which is why this
went unnoticed: the single-trial path behaves sensibly.

## The fix

`gate.py::deflated_sharpe_fixed` in this repo, already written and tested, in two
variants:

- `sigma_sr="se"` (default, conservative) — approximate `sigma_SR` by the SR
  estimator's own standard error. Algebraically this collapses to
  `ratio = SR*sqrt(T-1)/denom - bracket`, i.e. the PSR z-score minus a dimensionless
  penalty. Needs nothing but the return series.
- `sigma_sr="cross"` (faithful) — use the observed dispersion of SR across the N
  trials, which the alpaca ledger and the qlib battery scorecards actually record.

Both calibrate to ~0.5 on noise. `tests/test_gate.py` pins the calibration, the power,
the exact agreement with the canonical function at `n_trials=1`, and the round-trip
accuracy of the scorecard inversion.

## What changes if you accept it

163 stored gate rows can be corrected **by algebra on committed numbers** — the shipped
formula's inputs are recoverable from `(sr, ratio, n, n_trials)`, so `denom` is pinned
exactly and no backtest is re-run. Of those, **15 flip from FAIL to PASS** under the
frozen `ratio > 0` bar:

| what | sr | shipped | corrected | PBO | also clears >1.645 |
|---|---|---|---|---|---|
| qlib `H07_vrp_spy` | 0.397 | −17.29 | **+2.660** | 0.000 | **yes** |
| qlib `V2_backwardation_riskoff` | 0.255 | −28.71 | +1.533 | 0.000 | no |
| qlib `V1_ts_contango_timer` | 0.264 | −27.55 | +1.523 | 0.000 | no |
| qlib `C1_cot_extreme_fade` | 0.139 | −42.03 | +0.825 | 0.008 | no |
| qlib `V4_vvix_stress` | 0.092 | −65.56 | +0.644 | 0.000 | no |
| macro `surprise :: long_vol_overlay` 5d | 0.140 | −17.88 | +0.431 | 0.008 | no |
| macro `surprise :: short_spy_overlay` 21d | 0.275 | −8.00 | +0.400 | 0.000 | no |
| qlib `W03_vol_managed_spy` | 0.288 | −15.54 | +0.365 | 0.000 | no |
| macro `surprise :: short_spy_overlay` 5d | 0.128 | −18.37 | +0.309 | 0.000 | no |
| qlib `L1_netliq_trend` | 0.226 | −20.79 | +0.165 | 0.000 | no |
| + 5 more, all under +0.36 | | | | | no |

Two independent corroborations, which is the reason to take this seriously rather than
treat it as arithmetic that happens to be convenient:

1. `H07_vrp_spy` and `W03_vol_managed_spy` were **already** flagged `"verdict":
   "replicates"` by the 1995–2015 deep-history test, on profit factor, by a completely
   different method (2.92 → 5.02 and 1.39 → 2.18 across eras).
2. `surprise :: short_spy_overlay` already had **permutation p = 0.0315 with PBO 0.0**
   in its own scorecard. The permutation test and the corrected DSR now agree; only the
   broken DSR dissented.

Equally worth stating: the correction **rescues nothing** that had no edge. `rf` goes to
−0.995, `torch_mlp` to −1.641, `qlib_lgbm` has no positive IC to begin with (Q-QLIB-01),
and 148 of the 163 rows stay FAIL. A correct ruler still fails most things — it just
fails them for a reason.

## The threshold question this exposes (decide separately)

Once the DSR is computed correctly, `ratio > 0` is a **median** test: "better than the
*expected* max of N noise trials". Pure noise clears it **44.8%** of the time
(Q-GATE-02). That is not a significance test and never was.

The defensible bar is `ratio > 1.645` (prob > 0.95). Under it, exactly **one** of the 15
survives: `H07_vrp_spy`. Raising the bar and fixing the bug together would leave the box
with one candidate and a correct gate — arguably the honest end state.

These are two separate decisions and I have deliberately not bundled them.

## Options

1. **Accept the fix, keep `ratio > 0`.** Patch `validate.py`, re-gate every repo, 15
   rows become PASS candidates. Note that promotion still requires the every-horizon
   rule, so `straddle_all` (+0.253 at 21d, −3.06 at 5d) stays SHADOW regardless.
2. **Accept the fix AND raise the bar to 1.645.** One candidate (`H07_vrp_spy`),
   a calibrated gate, and the SHADOW discipline intact. My recommendation.
3. **Accept the fix, promote nothing yet.** Correct the ruler, re-gate, and require the
   flipped candidates to be re-run out-of-sample on fresh data before promotion. Slowest
   and safest; the batteries have not been re-run since 2026-07-13.
4. **Reject.** Keep the current formula. If so, `CLAUDE.md` should say plainly that the
   gate is a constant FAIL and that "nothing has cleared it" is a property of the
   arithmetic, not of the models — otherwise the record misleads whoever reads it next.

## If accepted, the work

- Patch `deflated_sharpe` in `macro_gpu_lab/macro_gpu_lab/validate.py`. It is imported
  by `qlib_lab`, `alpaca_gpu_lab/src/gate.py` and `copper_brain`, so one edit reaches
  everything — and `hq-trading-system/analytics/research_scorecard.py` plus
  `copper_brain/copper_brain/validate.py` hold the two older copies of the same math and
  need the same fix.
- Add the null-calibration test to that repo, not just this one, so it cannot regress.
- Re-run: macro_gpu_lab's 4 models, qlib_lab's ~40 batteries, alpaca's B01–B10.
- Keep every historical scorecard. Corrected numbers land as new dated rows; the old
  rows stay on the record with a pointer to this proposal.
- Delete `tests/test_gate.py::test_shipped_is_not_calibrated_on_noise` here, which
  exists purely to pin the defect while it is live.
