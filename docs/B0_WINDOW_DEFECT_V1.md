# B0 window defect V1 — the receipts misstated their own sample

**Found 2026-09-19 while building the T5 sensitivity report.** This is the most consequential
correction of the B0 work package, and it substantially revises an earlier headline conclusion.

## The defect

`--first-day` / `--last-day` appeared in exactly two places in `scripts/paper_trade_perp_v1.py`:

- assigned from the frozen protocol,
- written into the receipt's `data.first_day` / `data.last_day` metadata.

They were **never used to filter the loaded bars.** Every run therefore used the **entire
archive** while its receipt claimed the narrower window.

It was caught because the period-split runs for the sensitivity report returned byte-identical
results to the full-period run — a "half" of the sample cannot equal the whole sample.

## Evidence

| Check | Before the fix | After the fix |
|---|---:|---:|
| Quotes loaded (protocol window 2024-01-01 … 2026-08-31) | **4,008** | **2,919** |
| Decisions | **83** | **61** |
| Binance net PnL | +21,173.77 | **−34,080.29** |
| Archive days actually available (BTCUSDT) | 1,334 (2023-01-01 … 2026-08-31) | — |

4,008 quotes ÷ 3 symbols ≈ 1,336 days = the whole archive; 2,919 ÷ 3 = 973 days = the window the
receipt claimed. The narrower-window runs now genuinely narrow (2024 only: 1,098 quotes / 22
decisions; 2025 only: 1,095 quotes / 15 decisions).

## The fix

`day_window_ms(first_day, last_day)` parses the window as inclusive UTC days and returns a
half-open `[start_ms, end_ms)` interval, which is applied to the bar open times immediately
after the three series are intersected. A reversed window raises. Four regression tests cover
the interval semantics, the selection predicate, the reversed case, and the presence of the
filter in the driver.

## What this corrects in the published record

**Every previously reported paper-trading number was computed over 2023-01-01 … 2026-08-31**,
not over the 2024-01-01 … 2026-08-31 its receipts claim — including the "four contradictory
results" table (runs A–D) and the first set of B0 baselines.

### Corrected protocol baselines (window enforced, capacity `fixed`, 0 divergences each)

| Venue | Decisions | Net PnL | Max drawdown |
|---|---:|---:|---:|
| Binance | 61 | **−34,080.29** | — |
| Bybit | 61 | **−37,597.40** | — |
| Aster | 61 | **−38,044.85** | — |

**Cross-venue spread: 3,964.51 — about 4% of initial capital, and all three venues now agree in
sign and magnitude.**

For comparison, the previously published spread was **271,945** (2.7× capital) with Aster the
outlier at **+194,026**.

### The Aster outlier was mostly this defect

Aster with the participation cap still coupled to venue-reported volume, measured under the
corrected window:

| Aster capacity mode | Divergences | Net PnL |
|---|---:|---:|
| `venue_volume` (coupled) | 11 | −37,224.22 |
| `fixed` (decoupled) | 0 | −38,044.85 |

The capacity-coupling effect is **820**, not 279,000.

## Revised conclusion — and a retraction

The earlier conclusion was: *"the strategy is chaotic with respect to its input; a 1–2 bps price
difference swings the result by 2.7× capital."* **That is now substantially retracted.** Most of
the apparent divergence came from two tooling defects, not from the strategy:

1. a receipt that misstated its own sample window, so the venues were not in fact compared over
   the same period;
2. the participation cap coupled to venue-reported volume, which produced Aster's partial fills.

What survives, and is now the honest statement:

- On the **corrected** window the same strategy returns **−34.1k / −37.6k / −38.0k** across three
  venues — a spread of **≈4% of capital**, same sign, comparable magnitude.
- A 4%-of-capital spread is still **too large to ignore** and still means no single-venue number
  is a result, but it is an **interpretable, bounded dispersion**, not chaos.
- **The reference strategy loses money on all three venues** over this window. That is a
  statement about a mechanical moving-average crossover under declared provisional costs, not
  about any model, and it authorises nothing.
- The remaining dispersion is the legitimate subject of T5 attribution, not a mystery.

**The meta-lesson, which is the point of B0:** both defects produced plausible-looking numbers
that were quoted in documents. They were caught not by inspecting the code but by a reproduction
check that could not possibly have passed — a "half" equal to the "whole". Measurement integrity
is therefore not a formality before the science; here it *was* the finding.

## Still open

- The old receipts under `results/paper_trade_perp_{binance,bybit,aster,aster2}_v1.json` keep
  their false window metadata. They are retained deliberately as evidence of the defect and are
  **superseded**; `results/paper_trade_b0_baseline_*_v1.json` are the corrected artifacts.
- Fundings, fees and drawdowns in the corrected runs are recorded in the receipts but not all
  restated in this note.
- The old non-protocol receipts also used `capacity=venue_volume`, so they differ for two
  reasons at once and must not be diffed against the corrected baselines without saying so.

## Reproduction

```bash
.venv/bin/python scripts/paper_trade_perp_v1.py --source binance \
  --output results/paper_trade_b0_baseline_binance_v1.json
.venv/bin/python -m unittest scripts.test_perp_pipeline_v1.DayWindowTest -v
```
