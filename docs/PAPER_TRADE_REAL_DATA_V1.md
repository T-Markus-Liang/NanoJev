# Real-data paper trading V1 — T5 attributed sensitivity report

Status (2026-09-20): **T5 implementation and verification complete / READY_FOR_REVIEW**.
The harness now refuses a financial headline: `headline_allowed: false`.
This is offline measurement/accounting evidence, not a return, model edge, or trading approval.

Authoritative output: [B0 report](../results/paper_trade_b0_report_v1.json);
[study declaration](../research/paper_trade_t5_sensitivity_v1.json).
Raw final evidence: `results/paper_trade_t5_v1_run2/`.
The first 36-cell run in `results/paper_trade_t5_v1/` is preserved, superseded by
the hardened run2 (additional cash/order checks and explicit missing-day lists).

## Scope and accounting

Frozen base: `research/paper_trade_b0_protocol.json`, unchanged.
BTCUSDT/ETHUSDT/SOLUSDT, 20/60 MA crossover, initial cash 100,000, 3× leverage,
5 bps fee, 1 bps half-spread, fixed-notional sizing, fixed reference capacity.
The mechanical reference strategy exercises the execution path; no NanoJev trading policy
or financial training was used.

For each actual fill, with buy sign +1 and sell sign −1:

- Price/signal on the **executed path** = terminal marked inventory − sum(sign × quantity × multiplier × mid).
- Execution shortfall = sum(sign × quantity × multiplier × (execution price − mid)).
- Shortfall splits into spread, slippage, and tick rounding. The simulator's stored spread/
  slippage amounts exclude the contract multiplier; the report applies it.
- Subtract explicit fees, signed funding cost, and liquidation fees. Funding income contributes positively.
- Independently reconcile to reported net PnL and equity minus initial cash within absolute **1e−6**.
  Fill fees/notional/counts, terminal quantities, order fill totals, funding sums and cash also reconcile.

**This is not a cost-free strategy counterfactual.** Removing costs could alter later margin,
fills and positions. The old field `net_pnl_excluding_all_costs` adds back fees/funding but
still contains execution shortfall; it must not be called gross price/signal PnL.

Unfilled submitted-order quantities are reported by asset, with booked cash effect **0** and
counterfactual/opportunity PnL **null (not measured)**. Rejections are not automatically
attributed to capacity. All 36 actual cells have zero order divergences/remainders; synthetic
tests exercise partial/unfilled cases. Pre-submission rejections are not inferred from order remainders.

Reconciliation verifies internal arithmetic, not authenticity of quoted mids or market fills.
Source/input hashes make the run auditable; a forged, consistently rewritten receipt is not
cryptographically authenticated by an accounting identity.

## Full-window attribution — primary diagnostic sources

2024-01-01..2026-08-31, seed 20260919. Every value is simulated quote-currency PnL,
not a return. The other two seeds give identical values and are not independent evidence.

| Signed component | Binance | Bybit |
|---|---:|---:|
| Price/signal, executed path | −15,878.45 | −19,337.41 |
| Fees | −2,587.89 | −2,587.31 |
| Funding | −15,084.79 | −15,142.84 |
| Spread | −517.58 | −517.46 |
| Tick rounding | −11.57 | −12.39 |
| Slippage / liquidation fees / unfilled booked effect | 0 / 0 / 0 | 0 / 0 / 0 |
| **Net, shown only with dispersion below** | **−34,080.29** | **−37,597.40** |

Primary full-window min/median/max = **−37,597.40 / −35,838.85 / −34,080.29**;
spread **3,517.11**, 3.5171% of initial cash.
Most of the Binance–Bybit difference lies in the executed-path price term
(about 3,458.95), not explicit fee differences. This is an accounting attribution,
not a causal explanation of venue economics.

**Coverage warning:** Binance has **973** usable days per symbol; Bybit has **974**.
On **2026-06-29**, all three Binance symbols have last-price bars but lack mark/index
observations. The requested date interval is correct, but the actual calendars differ.
Therefore these are source-data sensitivity diagnostics, **not a matched-calendar venue effect**.
No missing observation was fabricated, dropped from the other sources, or refetched.

## Mandatory sensitivity matrix

36 cells = 3 sources × seeds 20260917/18/19 × full / calendar 2024 / calendar 2025 /
2026 YTD through August 31. Each window starts with fresh cash and flat positions,
with MA warmup **inside that window**. Full overlaps all subwindows.
These are neither additive slices of one equity curve nor equal-duration returns;
no annualization or pooled “best strategy” estimate is made.

Primary sources only; each row holds window and seed fixed while varying source.
All three seed copies give the same row:

| Window | Min net | Median net | Max net | Spread | % of initial cash |
|---|---:|---:|---:|---:|---:|
| Full | −37,597.40 | −35,838.85 | −34,080.29 | 3,517.11 | 3.5171% |
| 2024 | −53,913.71 | −53,750.64 | −53,587.58 | 326.13 | 0.3261% |
| 2025 | 109,242.67 | 110,052.73 | 110,862.79 | 1,620.13 | 1.6201% |
| 2026 YTD | 10,205.76 | 13,432.95 | 16,660.13 | **6,454.37** | **6.4544%** |

Seed-axis spread is **0 in every source/window** because fill probability is 1 and rejection
probability is 0. This checks deterministic equivalence, not stochastic robustness.

Window-axis min/median/max/spread (descriptive, unequal durations/overlap):
Binance −53,587.58 / −8,710.08 / 110,862.79 / 164,450.37;
Bybit −53,913.71 / −13,695.82 / 109,242.67 / 163,156.38.
Full period and independently restarted windows must not be summed or treated as replications.

### Why a headline is refused

The exploratory threshold was fixed **before the new matrix** at spread > **5% of initial cash**.
Historical corrected full-period baselines were already known: this is **not blind preregistration**
and not an established economic or statistical robustness criterion. Equality does not exceed it.

The report computes refusal reasons, including:

- 2026 YTD source spread exceeds 5%; window dispersion also exceeds it.
- Incomplete Binance calendar coverage confounds source comparisons.
- Fewer than three non-appendix sources; Aster cannot satisfy a primary result gate.
- Costs, contract templates, fill/funding conventions remain provisional pending R1.
- Current history snapshot is not an as-of vintage; Bybit terms remain unverified.
- A mechanical reference strategy is not evidence about a model.

This deliberately satisfies B0's **explicit refusal** branch, not its robust-economic-result branch.

## Data identity, venue identity and rights

Before and after both matrix runs, all manifest-listed files were hash-checked:
Binance **880**, Bybit **30**, Aster **35**, all match. Loader inputs are covered by those
manifests; undeclared Binance zip inputs are rejected. The replay makes **zero network data calls**.

Data source and simulation template are now separate receipt fields. Previously Aster's
receipt was incorrectly labelled Bybit; new receipts correctly say `aster_perp`.
All sources still use a **shared provisional `binance_um` simulation template** for
contracts/quotes/decisions. This is not per-venue execution fidelity; economics are unchanged.

Binance research permission is CC BY-NC-SA; §4.2 separately prohibits live proprietary
execution/commercial order generation. Bybit/Hyperliquid terms remain unverified.
Aster has a concrete unresolved permission conflict, not a clearance.
See [licensing audit](VENUE_DATA_LICENSING_V1.md). No new fetching, licence-risk acceptance,
live trading, training, provider switch, checkpoint promotion, or active context pruning occurred.

## Appendix — Aster only, unresolved licence and provenance metadata conflict

Excluded from every primary table and headline. Aster's original fetch manifest declares
`venue: aster` but incorrectly records `base_url: https://api.hyperliquid.xyz/info`.
File hashes match, but that does **not** repair the source attestation.
The manifest is preserved unchanged; no legal/source clearance is inferred.

Full-window signed attribution:
price −14,579.34; fees −2,599.10; funding −20,334.06; spread −519.82;
rounding −12.54; slippage/liquidation/unfilled booked effect 0;
net **−38,044.85**. Full-window three-source diagnostic min/median/max =
−38,044.85 / −37,597.40 / −34,080.29, spread **3,964.56** (3.9646% of initial cash).
This corrects earlier approximate text of 3,964.51; see machine-readable exact values.

Aster window net diagnostics: 2024 −52,966.26; 2025 110,496.31; 2026 YTD 10,890.74.
Each belongs alongside the all-source dispersion rows in the report's labelled appendix,
not as a standalone performance number. Aster full-window funding cost is about 5,249.27
greater than Binance while its price term is about 1,299.12 less negative; no causal or
trading inference follows.

## Verification and provenance

- **75 new tests**: 22 accounting, 53 report/real-evidence tests.
- Full suite: **651 tests OK, 2 skipped**, 56.801 s. Existing unclosed-file ResourceWarning remains.
- 36 final cells reconcile; maximum absolute residual **3.6307028494775295e−9**.
- Seed-20260919 full-window ledger, counts and `replay_sha256` equal each of the three
  corrected frozen baselines exactly: instrumentation did not change replay economics.
- Repeated Binance/full/20260917 receipt is byte-identical. Seeds are not independent trials.
- Standalone report CLI reconstructs the matrix; source and input hashes remain unchanged.
- Final report SHA-256: `f8f2a6010677a65c6ee68135f5bc415cb85b41927cc857d55c3055f125f3e76a`.
- Study SHA-256: `7a0a4241e271e753f37090c70177b33fef23d99451e36d16d62f10497319f6d3`.

Local NanoJev advisory inference used MPS/FP32, temperature 1, no remote model calls:
development `363ea8f9-9dfa-4412-95ce-855b5d2ce839`;
testing `575b04c5-d838-4a7b-9102-2f5c22c73263`;
new-evidence testing `beba315c-c42e-430d-84f3-72fc92312678`.
All abstained; main-model/deterministic checks decided and fallback feedback was recorded.
No optimization/deployment phase took place.

Official DeepSeek V4.1 Flash only:
audit `1789837095-1ba19b172e3c` (static, no executed test);
accounting tests `1789837679-b0a2f6f71f09`;
report tests `1789837976-0a05d65554a0`.
Both test jobs were isolated, changed only their permitted test file, and returned success.
Main reviewed actual files, removed the staging-only dependency stub, strengthened tests and
reran locally. Accounting tests do not prove market authenticity.
Report mutation tests mock protocol/hash I/O; separate real-matrix tests exercise actual files.

## Reproduction and historical correction

Use **fresh output paths**; the driver/runner/report refuse to overwrite evidence:

```bash
.venv/bin/python scripts/run_paper_trade_t5_v1.py \
  --output-dir results/paper_trade_t5_review_run \
  --report-output results/paper_trade_t5_review_report.json
.venv/bin/python scripts/paper_trade_report_v1.py \
  --receipts results/paper_trade_t5_v1_run2/receipts \
  --output /tmp/paper_trade_t5_reconstructed_report.json
.venv/bin/python -m unittest discover -s scripts -p 'test_paper_trade*py' -v
```

Raw `data/` is ignored and must already exist; do not turn this reproduction into an
automatic fetch. Protocol/receipt paths record the original machine's locations.
The original study runner is rooted at this checkout; a relocated evidence bundle needs
an explicitly reviewed path relocation rather than silently ignoring hashes.

Earlier “four contradictory numbers” (+21k/−78k/+194k) used a broken window and mixed
capacity policies. **The claim of strategy chaos is withdrawn**, not retold as a current
finding. Old receipts remain as evidence; see [window defect record](B0_WINDOW_DEFECT_V1.md).
The former un-attributed tables are replaced by this report.

Historical infrastructure remains: Binance/Bybit/Aster/Hyperliquid fetchers, the 6,554-row
PIT pilot and its unchanged validator. That **9-feature pilot is not the closed 12-feature
R1 dataset**. Three omitted features (OI level/change, liquidation intensity) and the
71 provisional parameters/19 construction values/numeraire require owner decisions.
Next task: **T6 R1 decision package**, not training or live deployment.
