# Two box-wide lessons from the 2026-07-29/30 pass

**Status: RECORDED, not proposed.** Both were measured, and both change how every future
result on this box should be read. The DSR proposal
(`2026-07-29-dsr-unit-bug.md`) was accepted and applied; these are what the work found
on the way.

---

## 1. No transaction cost had ever been measured, and the assumption was 5-25x too high

Every backtest here charged a flat `COST_PER_SIDE = 0.0005` — 5 bps per side, to every
asset, forever. Measured from 10.3M real IEX quotes (`alpaca_gpu_lab/src/data/quotes_sample.py`,
12 sessions x 3 stratified intraday windows):

| symbol | assumed | measured | ratio |
|---|---|---|---|
| SPY | 5.00 bps | **0.23** | 0.05x |
| QQQ | 5.00 | 0.34 | 0.07x |
| IWM | 5.00 | 0.39 | 0.08x |
| TLT | 5.00 | 0.57 | 0.11x |
| EEM | 5.00 | 0.87 | 0.17x |
| GLD | 5.00 | 2.56 | 0.51x |
| CPER | 5.00 | 5.75 | 1.15x |

Median across 21 measured symbols **0.93 bps = 0.19x** the assumption; 20 of 21 cheaper
than flat. Extended to 87 symbols via a log-log fit on dollar volume (R² 0.581).

**What it changed.** qlib's 5d long-short went from **−12.31 bps per rebalance at the
flat cost to +4.67 at the measured cost** (+8.14 for the cheapest construction) — a sign
flip on the cost assumption alone. It still fails (t ≤ 1.02, corrected DSR ≤ −0.285), so
the honest reading is that *cost explained the loss and absence of signal explains the
non-win* — two separate facts that only measuring the cost could separate.

The international-ETF reversal (`Q-REL-04`) is decided outright by it: gross +9-10 bps/day
against a measured 0.65-1.30 bps half-spread, where the flat 5 bps would have killed it.

**Rule going forward.** No backtest may cite a flat cost. `cost/costs.py::half_spread`
returns a per-symbol number with a `source` label (measured / predicted / fallback), and
that label belongs in the verdict. This is quoted spread only — a **floor** — with no
market impact, so an edge that is a small multiple of it is not proven tradable at size.

**The free shortcut does not work, and it was tested first.** Corwin-Schultz (2012) and
Abdi-Ranaldo (2017) both return **exactly 0.0** for any spread below ~40 bps at ETF
volatility; 15 of 99 real symbols came back as literally zero cost. `cost/spread_ohlc.py`
is kept as a documented negative result with a reproducible resolution-floor table, so
nobody tries it again. Testing an estimator against a *known* simulated spread before
trusting it on real data is what caught this in minutes rather than shipping a cost layer
that reported zero.

---

## 2. The qlib batteries are not reproducible, so no single-run verdict on them is a fact

Re-running the wave-1 batteries 17 days later, with **n and n_trials identical**:

| battery | sr then → now | PF then → now | n |
|---|---|---|---|
| `H07_vrp_spy` | 0.3967 → **0.2106** | 5.02 → **2.16** | 107 both |
| `H08_absorption_shift` | 0.1199 → **0.2946** | 1.40 → **2.11** | 104 both |
| `H13_turn_of_month` | 0.1124 → 0.1146 | 1.37 → 1.38 | 120 → 119 |

H13 barely moved. H07 and H08 moved enormously, in **opposite** directions.

### The cause, diagnosed — and it is not what I first guessed

My first suspect was dividend re-adjustment rewriting the price history. **That is wrong**,
and the arithmetic says so plainly: adding a dividend multiplies every prior price by the
same constant, which cancels in returns.

The real cause is **phase sensitivity of the non-overlapping trade grid**. Batteries
sample trades at `iloc[253::horizon]`, and `config.DAILY_YEARS = 10` is a **rolling**
window — so every re-run begins the series on a later date and re-phases the entire grid
onto different entry days. Measured directly on `H07_vrp_spy` by sweeping the offset:

| phase | Sharpe | PF | | phase | Sharpe | PF |
|---|---|---|---|---|---|---|
| 0 | +0.319 | 2.235 | | 12 | +0.220 | 1.725 |
| 6 | +0.478 | 3.419 | | 18 | **+0.188** | **1.646** |
| 9 | **+0.489** | **3.257** | | 21 | +0.319 | 2.235 (= phase 0) |

A **2.6x swing in Sharpe and 2.1x in profit factor from nothing but the start offset**,
with phase 0 and 21 identical exactly as the mechanism predicts. Both observed scorecard
values (0.211 and 0.397) fall inside that range.

So every battery verdict is one arbitrary 1/horizon slice of the available data, and
re-running silently redraws the slice.

**Why this matters more than any single verdict.** It cost me a wrong headline in this
very session: inverting H07's stored scorecard made it the strongest candidate on the box
at +2.660 and the only thing clearing the raised bar. Re-running it gives **−0.004**.
The inversion arithmetic was exact; the series it described no longer existed.

It also undermines the deep-history "replicates" verdicts that `Q-REL-02` and `Q-REL-03`
rest on, and it means "H07_vrp_spy replicates with PF 5.02" was never a stable fact.

**Proposed fixes, in order of value:**

1. **Average the PnL over all `horizon` phase offsets** (or report the full phase
   distribution instead of one draw). This is the real fix: the current construction
   discards 20/21 of the available observations and then reports the arbitrary slice it
   kept as the result. Phase-averaging uses every observation, makes the verdict
   deterministic, and shrinks its variance — and it would have prevented this entire
   episode. Note it also invalidates the *comparison* of any two existing battery
   scorecards, so it is a re-baseline, not a patch.
2. **Store the PnL series with the scorecard.** Then a verdict can be *re-derived* rather
   than *re-computed*, and a gate change never requires re-running a model. This alone
   would have made the whole DSR correction a pure recomputation with no ambiguity.
3. **Report a reproducibility delta.** On every re-run, print sr/PF/n against the prior
   scorecard. A silent 2.4x PF change should be impossible to miss.
4. *Not recommended:* anchoring the grid to a fixed calendar date. It makes results
   reproducible but keeps them arbitrary — one phase out of `horizon`, chosen by fiat.

**Caveat on the affected verdicts.** Phase sensitivity does not mean the effects are
fake; it means their reported magnitudes were never precise. H07's Sharpe is somewhere in
0.19–0.49 depending on the slice, not 0.397 and not 0.211. Any battery with a
`horizon` > 1 inherits this, which is nearly all of them, and the deep-history
"replicates" verdicts behind `Q-REL-02` and `Q-REL-03` inherit it too.

Registered as `Q-GATE-03`, currently **NO**, with the diagnosis above recorded against it.

---

## 3. Smaller things worth keeping

- **A constant PnL series used to pass the gate.** `sigma <= 0` does not catch a constant
  series, whose float std is ~1.4e-17: `[0.1]*20` produced `sr ≈ 7e15`, `ratio 3e22`,
  `prob 1.0`. Fixed in all three copies. Found by a test written to check something else.
- **`B04_xgb_sentiment` never tested its hypothesis.** Absolute acceptance at p ≥ 0.60
  fired zero trades on every row, so it graded a filter that never triggered. The rank
  acceptance B06-B08 already use cannot return zero trades and was sitting in the same
  file the whole time. Re-registered as `B04b_xgb_sentiment_rank`, leaving B04's record
  intact. **A battery reporting n=0 should be a hard error, not a FAIL** — a FAIL implies
  the hypothesis was tested.
- **Raising a bar rejects your favourites.** At 1.645, nothing on this box passes: not the
  surprise overlay (+0.238), not V2/V1 (+1.533/+1.524), not the international-ETF
  reversal (+1.281). That was accepted before it was measured, and it duly landed. A gate
  that discriminates with an empty pass list is the honest end state.
