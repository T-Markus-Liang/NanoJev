# Financial R1 decision package V1

Status: **OWNER-DECIDED (2026-09-20) — protocol rebase and real-data PIT acceptance still pending.**
The project owner explicitly authorized the current experimental decisions in the user message on
2026-09-20. This records those choices; it does not grant venue permission, replace an independent
reviewer, or claim that the frozen protocol/real-data PIT receipt already exists.

This is roadmap task T6. It depends on the T5 evidence in
[the attributed sensitivity report](PAPER_TRADE_REAL_DATA_V1.md) and
`results/paper_trade_b0_report_v1.json`, and reads the draft
[data plan](FINANCIAL_DATA_PLAN_V1.md), [simulator specification](FINANCIAL_SIMULATOR_V1.md),
and `research/financial_experiment_protocol_v1.json`.

## 1. Non-negotiable scope and gate

- Instrument class: crypto secondary-market **perpetual contracts only**; spot is excluded.
- V1 venue set after the owner decision: Binance and Bybit only. Aster is dropped; Hyperliquid is
  excluded from V1 because the required historical OI series is unavailable. Any future extension
  requires a new protocol and source review.
- No order submission, broker credential, account creation, paid data purchase, or live trading.
- No Track B training (T11 financial baselines or T14 RLCD-like estimators) before the owner-decided
  choices are applied to a new frozen protocol, its hash is recorded, and the real-data PIT receipt
  is accepted.
- No number in T5 is a return, edge, profitability, calibration, or execution-fidelity claim.

## 2. The three decisions R1 must freeze

| ID | Decision | Current draft/default | Owner state |
|---|---|---|---|
| **D1** | PILOT 9-feature subset vs the closed 12-feature allowlist | Closed 12 target; liquidation feature conditional | **OWNER_DECIDED (D1-a)** |
| **D2** | The **71** simulator/risk parameters plus **19** paper-construction values | Provisional engineering guesses | **OWNER_DECIDED (D2-a)** |
| **D3** | V1 quote numeraire and contract family | Single USDT-margined linear | **OWNER_DECIDED (D3-a)** |

The recommendation in §7 is now adopted as the owner decision. The protocol hash still must be
recomputed after the feature/numeraire rebase, and the unmodified validator must accept a real
frozen cohort before financial training can start; until those two receipts exist the no-training
gate remains closed.

## 3. D1 — feature contract

### 3.1 Existing contract and evidence

The draft protocol declares a **closed 12-feature set**:

`mark_price`, `index_price`, `mark_index_basis_bps`, `last_funding_rate`,
`funding_interval_hours`, `open_interest_level`, `open_interest_log_change_1d`, `quote_volume`,
`trade_count`, `taker_buy_ratio`, `realized_vol_24bar`, `liquidation_intensity_1d`.

The unmodified PIT validator requires every record to carry exactly the same feature-name set.
The real cohort currently delivered is a **6,554-record, 45.03% positive PILOT**. It is not the
R1 cohort. Binance daily `metrics` can supply the two open-interest fields (5-minute rows), but
no venue in the declared set publishes a historical liquidation/ADL series. Aster has no
open-interest endpoint. Missing liquidation must never be replaced with zero: zero means “none
observed”, not “not available”.

### 3.2 Options

| Option | Consequence | Evidence / required work |
|---|---|---|
| **D1-a: closed target, conditional liquidation** | Build OI from Binance metrics; freeze an 11-feature realizable schema by dropping `liquidation_intensity_1d` unless a written source ruling changes the universe. Preserves OI but requires a new ingestion/coverage receipt. | Data plan §§5.3–5.4; no historical liquidation source found. |
| **D1-b: freeze PILOT 9** | Fastest freeze, but permanently loses OI and is not the declared R1 closed contract. Pilot evidence cannot silently become R1 evidence. | Roadmap B1; current cohort. |
| **D1-c: restrict to venues with liquidation history** | Changes the declared venue set or requires a deposit-gated authenticated route; no declared venue currently satisfies it. | Data plan §5.4. |
| **D1-d: defer** | Blocks training and all formal results until OI ingestion and a liquidation ruling exist. | Safest if the owner rejects a conditional feature. |

Before D1 freeze, record the OI unit (base coin/contracts/USD), one-sided vs two-sided
convention, common grid and resampling rule, missing-value exclusions, and a recomputed definition
hash. The draft hash `8796b7f90a0f62f972f0f40e80e07b100f559f47342aece0bf5582fc27392ac9` is not frozen;
the superseded spot-era hash is void.

## 4. D2 — simulator and construction contract

### 4.1 Exact inventory

The code contains **48** `PROVISIONAL_POLICY_PARAMETERS` and **23**
`PROVISIONAL_RISK_PARAMETERS`, exactly **71** total. Their complete names are:

```text
fees.fee_bps, fees.fixed_fee_per_fill, fees.minimum_fee,
spread.half_spread_bps, spread.fixed_half_spread_price,
slippage.fixed_bps, slippage.impact_bps_at_full_capacity,
capacity.max_participation_fraction, capacity.max_order_quantity,
capacity.max_order_notional, capacity.on_order_exceeds_limit,
fills.reject_probability, fills.fill_probability, fills.minimum_fill_quantity,
fills.require_inventory_for_sell, fills.order_time_to_live_ns,
latency.decision_to_order_ns, latency.order_to_ack_ns, latency.ack_to_execution_ns,
guard.forbid_post_decision_prices_in_decisions,
guard.forbid_post_execution_prices_in_executions,
guard.forbid_post_decision_prices_in_executions,
marks.fallback, marks.price_source,
account.require_sufficient_cash, account.on_insufficient_cash,
account.cash_buffer_fraction,
perps.allow_short, perps.default_leverage, perps.max_leverage,
perps.margin_mode, perps.position_mode, perps.maintenance_margin_rate,
perps.funding_interval_ns, perps.funding_anchor_ns, perps.funding_rate_source,
perps.default_funding_rate, perps.funding_rates, perps.funding_rate_schedule,
perps.liquidation_fee_bps, perps.trace_perp_curve,
contracts.*.contract_multiplier, contracts.*.tick_size, contracts.*.lot_size,
contracts.*.min_notional, contracts.*.margin_mode, contracts.*.max_leverage,
contracts.*.maintenance_margin_rate,
limits.max_position_quantity, limits.min_position_quantity,
limits.max_gross_exposure_notional, limits.max_abs_net_exposure_notional,
limits.max_net_loss_notional, limits.max_realized_loss_notional,
limits.max_turnover_notional, limits.turnover_window_ns,
limits.max_orders_per_window, limits.order_rate_window_ns,
limits.max_order_notional, limits.max_data_staleness_ns,
limits.max_clock_drift_ns, limits.require_data_quality_fields,
limits.latch_kill_switch_on_block, limits.max_leverage,
limits.max_position_notional_per_venue, limits.max_funding_cost_notional,
limits.max_margin_ratio, limits.min_available_margin_notional,
limits.min_liquidation_distance_fraction, limits.max_initial_margin_utilization,
limits.require_margin_sufficiency
```

The separate **19 construction values** are:

```text
path.ticks, path.tick_ns, path.feed_delay_ns, path.regime_length_ticks,
path.initial_price, path.drift_bps_per_tick, path.noise_bps_per_tick,
path.volume, path.mark_basis_fraction, path.tick_size, path.lot_size,
path.min_notional, strategy.fast_ticks, strategy.slow_ticks,
strategy.target_quantity, strategy.rebalance_ticks, strategy.threshold_fraction,
strategy.allow_short, backtest.initial_cash
```

These are not venue measurements. Fees, spread, impact, queue/fill probability, funding,
margin tiers, liquidation execution, and latency remain modelling choices; the archive has no
order book. T5 therefore refuses a headline partly because these values are provisional.

### 4.2 Options and additional contract rulings

| Option | Consequence |
|---|---|
| **D2-a: freeze all 71+19** | Strongest reproducibility; each entry needs a value, admissible range, provenance or explicit conservative rationale. Unsupported values remain labelled guesses, not facts. |
| **D2-b: freeze a subset** | Formal comparisons remain non-reproducible and training stays blocked. |
| **D2-c: retain defaults** | Permitted only for plumbing validation; cannot support a strategy/model result. |

R1 must also rule on venue-rule fidelity (real tick/lot/min-notional vs synthetic template),
margin tiers and portfolio margin, liquidation/ADL close-out, funding interval/anchor/rate
provenance and clamps, venue-segregated vs one-way inventory, seeded-cap fill model vs any
order-book model, and mid vs last vs microprice reference. These are contract decisions attached
to D2, not permission to quietly alter one parameter after seeing results.

## 5. D3 — numeraire

| Option | Consequence | Recommendation status |
|---|---|---|
| **D3-a: USDT-margined linear only** | One quote basis; defers coin-margined inverse and Hyperliquid USDC-margined contracts to separately registered extensions. Instrument identity includes venue + contract type + symbol. | Recommended, not approved |
| **D3-b: add coin-margined inverse** | Requires point-in-time conversion into a common quote and a separate inverse-contract label/ledger; otherwise comparisons mix bases. | Not recommended for V1 |
| **D3-c: add USDC-margined on-chain** | Adds a third stablecoin/peg and different contract semantics; cannot be pooled silently with USDT. | Not recommended for V1 |

The gross mark-price event may be described per venue, but cost, funding, margin and execution
numbers are not comparable merely because the underlying symbol matches.

## 6. Cross-cutting evidence and open questions

### 6.1 Rights, vintage and venue coverage

Binance Vision is CC BY-NC-SA 4.0; §4.1 permits personal non-commercial research and §4.2
prohibits live proprietary execution. Bybit and Hyperliquid terms remain unverified. Aster has a
read concrete conflict: §6.1(b) requires prior written consent to download material and §6.2(e)
requires express permission for automated access; its results remain appendix-only. Project-owner
authorization is not venue permission. If live execution is ever intended, the data right must
change and features/labels must be re-derived.

All current histories are snapshots, not as-of vintages; Binance documents in-place archive
rewrites. A future real-data PIT receipt must be produced by the unmodified validator against the
frozen cohort. Synthetic compatibility is not evidence of real-data acceptance.

### 6.2 Holdout, labels, seeds and OI

- Decide whether a venue holdout is mandatory in addition to instrument holdout. Hashing only
  instrument IDs can scatter near-identical contracts across groups; an underlying-clustered
  partition is the candidate refinement.
- The draft event is gross `perp_forward_mark_return_up_25bps_1d_gross`: exact mark-to-mark,
  25 bps, 1-day, no fees/spread/slippage/funding/leverage. Gross event quality and net simulator
  utility must never share a column.
- Training seeds are 1729, 2718, 3141; data seed is 20260919 and `split_seed` is null. T5's
  seeds were intentionally degenerate (`fill_probability=1`, `reject_probability=0`), so equal
  results prove determinism, not robustness.
- OI grids differ: Binance 5-minute metrics, Bybit selectable grids, Hyperliquid hourly, Aster
  none. R1 must freeze a common grid, resampling direction, unit, sidedness, and missingness;
  Binance `create_time` is a UTC string, not an epoch integer.

### 6.3 T5 evidence that informs, but does not decide, R1

| Evidence | Current fact |
|---|---|
| Matrix | 36 offline cells = 3 sources × 3 seeds × 4 independently restarted windows |
| Accounting | Maximum absolute attribution residual `3.6307028494775295e−9` (< `1e−6`) |
| Seed axis | Zero spread in every cell family; deterministic, not stochastic robustness |
| Full primary window | Binance/Bybit min/median/max `−37,597.40 / −35,838.85 / −34,080.29`; spread `3,517.11` (3.5171% initial cash) |
| 2026 YTD | Primary spread `6,454.37` (6.4544%) > declared exploratory 5% threshold |
| Coverage | Binance has 973 usable days/symbol vs Bybit 974; all Binance symbols lack mark/index on `2026-06-29` |
| Report gate | `headline_allowed: false`; Aster is appendix-only and its manifest base URL incorrectly names Hyperliquid |

T5 is accounting/plumbing evidence from a mechanical reference strategy. It neither selects
features nor proves a numeraire, cost, model, or economic edge.

## 7. Adopted owner decision

The owner adopts **D1-a** (build OI, drop unobservable liquidation), **D2-a** (freeze all 71+19
values as explicit simulator/construction assumptions with provenance and ranges), and **D3-a**
(USDT-margined linear only). The PILOT 9 remains pilot evidence and is not relabelled as R1.
This is an internal experiment decision, not a venue licence or a claim that the values are real
market measurements. Any later change creates a new protocol/review record rather than editing a
result in place.

## 8. Owner decision record — resolved choices and bound PIT evidence

| ID | Decision to record | Options | Owner state | Frozen value / rationale |
|---|---|---|---|---|
| D1 | Feature allowlist | D1-a | **OWNER_DECIDED** | 11-feature realizable schema; build OI and remove `liquidation_intensity_1d` rather than impute it |
| D1.1 | Build OI features | metrics ingestion | **OWNER_DECIDED** | Binance metrics plus venue-specific coverage receipt; units/sidedness/grid must be recorded |
| D1.2 | Liquidation feature | drop | **OWNER_DECIDED** | No uniform historical source; no zero substitution and no universe restriction in V1 |
| D1.3 | Definition hash | recompute after freeze | **OWNER_DECIDED** | Draft hash is not reused; new feature/numeraire protocol gets a new SHA-256 |
| D2 | 71 policy/risk parameters | D2-a | **OWNER_DECIDED** | Freeze every named field to the code-default value at the protocol rebase; each remains an explicit model assumption |
| D2.1 | 19 construction values | freeze | **OWNER_DECIDED** | Freeze synthetic construction defaults; they are plumbing controls, never venue measurements |
| D2.2 | Venue rule fidelity | synthetic scope | **OWNER_DECIDED** | Do not claim venue-specific tick/lot/margin/ADL fidelity without a separate source receipt |
| D2.3 | Margin/liquidation/funding semantics | explicit limitation | **OWNER_DECIDED** | Linear quote-settled, single maintenance rate, declared funding schedule; inverse/portfolio-margin/ADL excluded |
| D2.4 | Fill and reference-price model | seeded cap + mark price | **OWNER_DECIDED** | Deterministic seeded fills and declared mark-price valuation; no order-book authenticity claim |
| D3 | Quote numeraire | D3-a | **OWNER_DECIDED** | USDT-margined linear perpetuals only; inverse and USDC/on-chain instruments are future extensions |
| C1 | Data rights and live scope | offline-only | **OWNER_DECIDED** | No order submission, credentials, live trading, or paid data; **Aster dropped as a data source by owner decision 2026-09-20** (no permission requested, no risk acceptance; existing Aster receipts are historical appendix only) |
| C2 | PIT/vintage policy | hash-pinned forward | **OWNER_DECIDED** | Unmodified validator, explicit as-of/availability, no restatement or silent forward fill |
| C3 | Venue holdout | required | **OWNER_DECIDED** | Report venue and instrument holdouts separately; no cross-venue pooling without a declared group key |
| C4 | Mark-price gross label | accept | **OWNER_DECIDED** | Keep the 25 bps/1-day mark-price event separate from net utility and execution costs |
| C5 | Seeds | 1729, 2718, 3141 | **OWNER_DECIDED** | Fixed training seeds; data seed remains 20260919 and split geometry is protocol-bound |
| C6 | OI grid and resampling | common daily UTC grid, causal backward/as-of join | **OWNER_DECIDED** | Missing OI excludes the row; no interpolation or future observation |
| C7 | Threshold/horizon | 25 bps / 1 day | **OWNER_DECIDED** | Exact decimal threshold, half-open `[25.0,+inf)` bps interval |
| C8 | Calendar stress windows | accept declared list | **OWNER_DECIDED** | Keep declared UTC stress windows; any revision creates a new protocol |
| C9 | Real-data PIT receipt owner | project owner execution + unmodified validator | **OWNER_DECIDED** | Owner may run the receipt; this is not an independent reviewer identity or an external approval |

### 8.1 D2 value source and freeze boundary

The 71 simulator/risk fields are frozen to the deterministic defaults in the current simulator and
risk modules at this owner decision. The 19 construction fields are frozen to the deterministic
defaults in `scripts/financial_backtest_v1.py` (`ticks=480`, `tick_ns=180000000000`,
`feed_delay_ns=0`, `regime_length_ticks=120`, `initial_price=30000.0`,
`drift_bps_per_tick=0.6`, `noise_bps_per_tick=6.0`, `volume=250.0`,
`mark_basis_fraction=0.0002`, `tick_size=0.0`, `lot_size=0.0`, `min_notional=0.0`,
`fast_ticks=8`, `slow_ticks=32`, `target_quantity=1.0`, `rebalance_ticks=4`,
`threshold_fraction=0.0`, `allow_short=true`, `initial_cash=100000.0`). These are synthetic
construction controls, not observations of Binance, Bybit, Aster or Hyperliquid.

The code-default provenance fingerprints at the time of this decision are:

- `scripts/financial_simulator_v1.py` — `2396d24091a688cf3cd1a46396be273025cf1a7d234c19a3a5315045c1cc9f8a`;
- `scripts/financial_risk_v1.py` — `bc2b933bbcb73fc143ed0e667996b5dd5194c4dfeca4ce812b1c765167b2c60b`;
- `scripts/financial_backtest_v1.py` — `51f70ffc32aa48eb57a0089f107deae2f1cdb36f7939255314d91858f7fb449e`.

This records the owner choice. W35 completed the protocol-side freeze in
`research/financial_experiment_protocol_v2.json` (SHA-256
`ab1eec401e14ee20d37a43ff4a5cca962dc60f58abc357a2f95fcc8fbcef1340`) and recorded the new
event-definition hash `cffd49217c95e83bedffc05f43f018964758cc58ba153f3963fc221d1f37124c`.
W36's fail-closed binder correctly rejected the older 9-feature pilot; the subsequent Binance
metrics/OI cohort now satisfies the R1 PIT gate. The authoritative bound receipt is
`results/financial_pit_r1_bound_receipt_20260920_v5.json` (`preflight_passed_not_training_authorized`),
with 3,266 records and 11 features. This closes the T6 evidence gate but does not grant training or
measurement authorization.

## 9. Strict no-training-before-freeze gate

Before T11 or T14 may start, all of the following must exist:

1. D1, D2, D3 and the cross-cutting choices are recorded with owner authority.
2. The frozen protocol and its SHA-256 are recorded in the W35 freeze receipt
   `results/financial_r1_protocol_freeze_v1.json`.
3. The unmodified PIT validator accepts the projected core on the **R1 11-feature** real frozen cohort, with all
   four phases nonempty in every fold.
4. The event-definition hash is recomputed from the frozen object and verified.
5. Any change to the feature set, label, universe, split/embargo, holdout, action space, seeds,
   numeraire or simulator semantics creates a new protocol/review; no in-place result repair.

The R1 cohort receipt now exists, but formal training, strategy selection, result headlines, live execution, data purchase,
and licence-risk acceptance remain prohibited unless their own review and authorization gates pass. This package itself makes no
network calls and no code or data changes.
