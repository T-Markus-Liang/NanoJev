# Financial execution simulator, perpetual-contract policy layer and paper-trading backtest V1

Status: **executable synthetic simulator core, perpetual-contract policy layer, deterministic
risk guards and an offline paper-trading backtest harness**.

[financial_simulator_v1.py](../scripts/financial_simulator_v1.py) replays constructed decisions
against constructed quotes with an explicit, parameterized cost/timing/fill model, a full order
lifecycle, a conserving ledger, a hard no-future-information guard, and — since the R1 scope
change — a perpetual-contract layer with signed two-sided positions, margin, leverage, declared
funding and liquidation.
[financial_backtest_v1.py](../scripts/financial_backtest_v1.py) consumes a synthetic market path
plus a decision stream (or a deterministic reference strategy) and produces a complete paper-trading
run: fills, funding, fees, margin usage, liquidations, an equity curve, drawdown, turnover and
per-instrument / per-regime breakdowns.
[financial_risk_v1.py](../scripts/financial_risk_v1.py) evaluates position, exposure, loss, turnover,
order-rate, stale-data, clock-drift, kill-switch, leverage, margin-sufficiency, margin-ratio,
liquidation-distance, funding-cost and per-venue-position guards entirely outside any model.

**Declared scope (R1 decision):** the financial track covers **crypto secondary-market perpetual
contract trading only** — Binance, Bybit, Aster and Hyperliquid are the declared venue universe.
Naming a venue declares which market structure the synthetic contract shape is *modelled on*; no
venue is contacted, no venue rule is replicated or verified, and no venue-specific claim is made.

## Honesty statement (read first)

- **All numbers in this document, in the module defaults, in the unit tests, and in
  [the stress receipt](../results/financial_simulator_v1_stress.json) are synthetic.** No quote,
  price, spread, fee, volume, funding rate, latency, liquidation or fill is measured from a real
  venue, a real feed, or a real broker. Nothing here downloads market data or connects to anyone.
- **"Paper trading" here means offline replay of synthetic data.** It is not shadow trading against
  a live feed, it submits no order anywhere, and it is not a step towards live capital.
- **No real-data authenticity is established.** The simulator can only consume caller-supplied
  observations; it cannot tell whether an observation was genuinely point-in-time. That audit
  belongs to the source/rights/universe manifest work in P1.
- **No profitability, calibration, fill-authenticity, capacity realism, or live-readiness claim is
  made.** The receipt records that guards *fire*, that the ledger *balances*, and that the run is
  *reproducible*; it does not record that a strategy makes money. Every reported PnL figure is an
  accounting residual of a synthetic tape.
- **Every cost, timing, capacity, fill, margin, funding and liquidation rule is a policy object
  whose numeric default is an engineering guess.** The parameters marked *provisional pending R1*
  below must be frozen at review gate R1 before this simulator is used to compare strategies.
- Live trading, broker/exchange credentials, borrowed inventory and real capital remain out of
  scope for this work package and for the V2 roadmap as a whole.

## Artifacts

| Path | Role |
|---|---|
| `scripts/financial_simulator_v1.py` | Deterministic event-time replay simulator, spot + perpetual policy layer, order lifecycle, conserving ledger, leak guard, synthetic stress suite and its CLI |
| `scripts/financial_backtest_v1.py` | Paper-trading backtest harness: synthetic path generator, deterministic reference strategies, metrics, machine-readable receipt and its CLI |
| `scripts/financial_risk_v1.py` | Out-of-model deterministic allow/reduce/block guards (spot and perp), windowed turnover/order-rate state and the kill switch |
| `scripts/test_financial_simulator_v1.py` | 80 pure synthetic unit tests |
| `results/financial_simulator_v1_stress.json` | Machine-readable stress receipt produced by the CLI |
| `docs/FINANCIAL_SIMULATOR_V1.md` | This specification |

It consumes the record and split contract of [Financial PIT V1](FINANCIAL_PIT_V1.md) conceptually —
synthetic decision timestamps follow the same "nonnegative integer UTC Unix nanoseconds" and
"nothing unavailable at the decision may influence the decision" rules — but the simulator does
**not** import or extend the PIT schema. No PIT schema change is made here, so no migration is
required.

## Event timeline

Four distinct timestamps are modelled and never collapsed into one:

| Timestamp | Meaning | Constraint |
|---|---|---|
| `decision_ns` | When the decision was made | Only quotes with `available_ns <= decision_ns` may influence it |
| `submit_ns` | Order handed to the (synthetic) venue | `decision_ns + latency.decision_to_order_ns` |
| `ack_ns` | Venue accepted or rejected | `submit_ns + latency.order_to_ack_ns` |
| `execution_ns` | Fill event | `>= ack_ns + latency.ack_to_execution_ns`, and `== quote.available_ns` of the observation used |
| `expire_ns` | Order time to live elapses | `max(decision_ns + ttl, submit_ns)`; terminal `expired` if not fully filled |

A `Quote` carries both `event_ns` (when the source observed the price) and `available_ns` (when it
became usable). The decision-time reference price is selected by a pluggable `reference_selector`;
the default `causal_reference_selector` returns the newest quote already available at `decision_ns`,
but the selector receives *all* quotes including future ones, so the guard — not the selector — is
the authority on causality.

## Order lifecycle

```
submitted ──► accepted ──► partially_filled ──► filled
    │            │              │
    │            └──────────────┴────────────► expired
    ├──► rejected
    └──► expired            (lapsed before the venue acknowledged)
```

`validate_status_history` enforces the transition table, non-decreasing timestamps, and exactly one
terminal status on every emitted order. `partially_filled` may repeat. Every fill carries
`order_id`, `fill_id`, `asset_id`, `venue`, `position_key`, `side`, `quantity`,
`contract_multiplier`, `position_effect`, `execution_ns`, `quote_event_ns`, `quote_available_ns`,
`quote_source_id`, `quote_version`, `mid_price`, `execution_price`, `spread_cost`, `slippage_cost`,
`explicit_fee`, `realized_pnl`, `closed_quantity`, `position_after`, `is_liquidation`,
`liquidation_id` and `cash_after`, so each fill is traceable to its order, to the exact observation
it used, and to the position it produced.

## Actions: two-sided, with abstain and no-trade as first-class

| Action | Meaning |
|---|---|
| `abstain` | The model does not know; logged with `model_abstain`, no order, no ledger effect |
| `no_trade` | The policy declines; logged with `policy_no_trade`, no order, no ledger effect |
| `buy` / `sell` | Order direction. Under one-way netting a `sell` that exceeds the long position opens a short when the declared contract allows it |
| `long` / `short` | **Target signed position.** `long 3` moves the position to `+3`; `short 3` moves it to `−3`; the engine derives the required buy or sell and whether it opens, reduces or flips |
| `flat` | Closes the instrument to zero (under hedge mode: both legs) |
| `open_long` / `open_short` / `close_long` / `close_short` | Explicit order effects. `close_*` is reduce-only and never requires new margin |

`position_effect` on a decision makes any order explicitly `open` (increase only), `close` (reduce
only) or `auto` (net). Target-position actions require a declared `perp` contract; on a spot
instrument they are rejected with `perp_action_requires_declared_contract` rather than silently
treated as a long-only trade.

A risk `block` is recorded separately as `outcome: "risk_blocked"` with the guard's reason codes. All
of these are distinguishable in the receipt, and every decision lands in exactly one counted bucket
(`abstain`, `no_trade`, `risk_blocked`, `rejected`, `no_order`, `ordered`), so the counts always
close — this answers the P2 review finding that a residual of "no `no_order_decisions` field" made
up to 22 decisions unauditable in some scenarios. (The review's guess that the residual was
`no_order` was itself wrong: it is `rejected_decisions`, sells refused for insufficient inventory.)

## Perpetual-contract layer

A caller declares a `ContractSpec` for an `(asset_id, venue)` pair with `instrument_type == "perp"`.
Only then do the perp semantics activate for that instrument; without a declared contract it behaves
exactly as the long-only spot slice did. Linear, quote-settled contracts only: inverse contracts,
portfolio margin, insurance funds, auto-deleveraging, bad-debt socialisation and cross-asset netting
are **not** modelled, and mixed account-level margin modes are refused at construction rather than
half-modelled.

### Contract specifications

`ContractSpec` declares `contract_multiplier`, `tick_size`, `lot_size`, `min_notional`,
`margin_mode`, `max_leverage`, `maintenance_margin_rate` and `settlement_asset`. Enforced behaviour:

- a quantity is floored to `lot_size` (`quantity_rounded_down_to_contract_lot_size`), and an order
  whose floored notional is below `min_notional` is rejected (`below_contract_minimum_notional`);
- a buy execution price is rounded **up** to the next `tick_size` and a sell **down**
  (`execution_price_rounded_to_contract_tick`), so tick granularity is never a free improvement;
- `contract_multiplier` scales notional, fee base, funding notional, margin and PnL;
- one multiplier is required per `asset_id`; conflicting declarations raise rather than silently
  pick one.

### Positions, side and netting

Positions are **signed** in contracts: long positive, short negative. Quote-currency cost basis
carries the same sign, so `average_cost` is positive in either direction and
`unrealized = market_value − cost_basis` needs no special case.

| `position_mode` | Position key | Behaviour |
|---|---|---|
| `one_way` (default) | `asset_id` (venues aggregated, as in the spot slice) | One signed net position; crossing zero realizes the closed part and opens the remainder |
| `hedge` | `asset_id::long` and `asset_id::short` | Two legs that never net; `buy` acts on the long leg, `sell` on the short leg, `flat` closes both |

Positions reduced to floating-point dust are snapped exactly flat (`FLAT_QUANTITY_TOLERANCE`), so a
residual never triggers a phantom liquidation.

### Margin model

`margin_mode` is declared once for the account (`PerpPolicy.margin_mode`, which every declared
contract must agree with).

- **`margin_balance`** = `equity` = `cash + Σ signed mark-price notional`.
- **`allocated_margin`** (the initial-margin reservation) accrues as
  `opening_notional / leverage` and is released pro rata on reduction.
- **`available_margin`** = `margin_balance − allocated_margin`.
- **`maintenance_margin`** = `Σ |quantity| × multiplier × mark_price × maintenance_margin_rate`.
- **`margin_ratio`** = `maintenance_margin / margin_balance`; a synthetic position is `1.0` when
  maintenance exactly consumes the balance.
- **Margin sufficiency is enforced, never assumed.** Every risk-increasing order must fit its
  opening portion inside `available_margin` at the declared leverage. If it does not, the quantity
  is reduced (`insufficient_initial_margin` plus `initial_margin_quantity_reduce`) and the order is
  rejected outright when nothing is affordable. Closes and reductions need no new margin, so a
  risk-reducing exit is always available. The P2 review's "no cash/buying-power check" gap is closed
  on both sides: the perp path checks margin and the spot path checks available cash
  (`insufficient_cash_for_spot_purchase`), so no order can be filled into negative cash.
- Leverage scales the margin requirement only. For a fixed position size, leverage changes
  `required_initial_margin` and the liquidation distance, **not** the marked PnL — which also means
  the honest way to gain exposure with leverage is to hold a larger position, with a correspondingly
  larger loss. The per-order `required_initial_margin` reported on each order is indicative: it is
  the requested quantity times the decision-time reference price times `1/leverage`, whereas the
  ledger's `allocated_margin` accrues from the actual execution prices (and only on the opening
  portion of a crossing order).

An analytic per-position liquidation price is reported alongside the account view:

- long: `P_liq = (cost_basis − allocated_margin) / (quantity × multiplier × (1 − mmr))`
- short: `P_liq = (cost_basis + allocated_margin) / (|quantity| × multiplier × (1 + mmr))`

so for a single position entered at `E`, the isolated long trigger is `E(1 − 1/L)/(1 − mmr)`, i.e.
the liquidation distance shrinks as leverage rises, as expected.

### Liquidation model

Liquidation is evaluated at **every available synthetic price observation** and at **every funding
boundary** between decisions, and the full observation timeline up to the as-of time is visited
exactly once, so a breach that later recovers is still detected and no observation after the
evaluation time can influence the state at that time.

| Margin mode | Trigger | Scope |
|---|---|---|
| `isolated` | a leg's `allocated_margin + unrealized − funding_cost_to_date ≤ leg maintenance margin` | that leg only; the rest of the wallet is untouched |
| `cross` | `margin_balance ≤ total maintenance margin` | every open perpetual leg |

A breach flattens the position at the mark price, tick-rounded against the trader, and writes
explicit records: a `liquidations` entry (position key, signed quantity, mark and last price,
analytic liquidation price, margin balance, maintenance margin, margin ratio, trigger reason code
`margin_below_maintenance_requirement`, close price, realized PnL, fee, `position_after = 0.0`), a
trade cash entry carrying `is_liquidation` and the `liquidation_id`, and a separate
`liquidation_fee` cash entry. `conservation_report` independently verifies that every liquidation
fill leaves its position key flat, and adds `liquidation_did_not_flatten_position` to `violations`
if it does not.

Not modelled, and therefore not claimed: close-out slippage beyond the declared fee, insurance
funds, auto-deleveraging, bad-debt socialisation, partial liquidation, and "loss capped at the
allocated margin" guarantees. In isolated mode the realized loss can exceed the allocated margin
when the synthetic mark gaps past the trigger; the receipt shows the realized number rather than
assuming a cap.

### Funding model

Funding is charged at declared intervals from a **declared funding-rate source field**
(`PerpPolicy.funding_rate_source`), recorded on every payment so a receipt always states where its
rates came from:

| Source mode | Rate used |
|---|---|
| `policy_constant` | `default_funding_rate` at every boundary |
| `declared_schedule` | an explicit `(asset_id, venue, funding_ns) → rate` declaration wins; the declared per-instrument constant table is the fallback; then `default_funding_rate` |
| `quote_funding_rate_field` | `Quote.funding_rate` when declared, else `default_funding_rate` |

Payment arithmetic, on the **mark-price** notional:

```
notional = position_quantity × contract_multiplier × mark_price
amount   = −notional × funding_rate          # a long pays when the rate is positive
```

so a short receives when the rate is positive, and `funding_cost = −Σ amount` is monotone in the
declared rate for a fixed position. Each payment is its own ledger row with reason code
`declared_funding_payment` (`funding` in the cash entries), its rate, its declared source mode, its
mark price and the cash balance after it. In isolated mode funding is charged against that leg's
allocated margin, so a position can be funded into liquidation. When no usable mark observation
exists at a boundary the payment is **not invented** and the boundary is skipped.

### Mark price versus last price

`Quote.mark_price` and `Quote.last_price` are optional declared inputs; when omitted, the mid is used
for both. `MarkPolicy.price_source` selects which one values an open position (`mark` by default).
Funding and margin use the mark price. Execution always crosses the synthetic bid/ask and is
unaffected by the valuation source. Both `mark_price` and `last_price` are reported per mark, so a
receipt shows both even when only one was used.

## Paper-trading backtest (`financial_backtest_v1.py`)

A paper-trading backtest here is a **deterministic, offline, synthetic replay** with a complete
machine-readable run record. It is explicitly *not* shadow trading, not a live-feed simulation, and
not evidence of profitability.

- **Market path** — `build_synthetic_perp_path(seed, …)` constructs quotes from a seeded SHA-256 draw
  function plus an explicit **synthetic regime schedule** (`synthetic-regime-up`,
  `synthetic-regime-down`, `synthetic-regime-chop`) that drives the constructed drift. Defaults span
  480 three-minute ticks (24 synthetic hours) so a declared 8-hour funding interval produces three
  boundaries. Regime labels are path-construction labels, not measured market regimes.
- **Decision stream** — either caller-supplied `Decision` objects or a deterministic reference
  strategy. `MovingAverageCrossoverStrategy` is causal (only quotes with
  `available_ns <= decision_ns` enter the averages) and emits two-sided target actions. `NoOpStrategy`
  is the zero-trade control.
- **Run outputs** — fills, orders, decisions, the full ledger, funding payments, liquidations, an
  equity/margin curve sampled at synthetic observation times, drawdown (notional and fractional,
  with peak, trough, recovery and longest-drawdown timestamps), turnover, cost totals, margin usage
  (max/mean allocated margin, max margin ratio, max initial-margin utilization, minimum liquidation
  distance, max gross position notional), per-instrument and per-regime breakdowns, and the
  instrument/regime PnL attribution residuals.
- **Determinism and replayability** — the harness replays the run twice and records both receipt
  hashes. An identical path, decision stream, policy and seed produce a byte-identical receipt
  (`backtest_sha256`), and the receipt separately proves the equity curve, fills, funding payments
  and liquidations are identical across the two runs.
- **Self-checks** — `require_backtest_invariants` re-derives the attribution sums, the ledger
  conservation report, the curve endpoints against ledger equity, and the byte-identity claim, and
  raises `BacktestError` on any mismatch.

Reproduce (never downloads anything, never contacts a broker):

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_financial_simulator*.py'
.venv/bin/python scripts/financial_simulator_v1.py --stress --output results/financial_simulator_v1_stress.json
.venv/bin/python scripts/financial_backtest_v1.py --output /tmp/financial_backtest_v1.json --seed 20260919 --with-risk-limits
.venv/bin/python scripts/financial_risk_v1.py --self-check
```

## Policy objects (all defaults are provisional pending R1)

| Policy | Fields | Default semantics in V1 |
|---|---|---|
| `FeePolicy` | `fee_bps`, `fixed_fee_per_fill`, `minimum_fee` | `fee = max(minimum_fee, notional × fee_bps/10000) + fixed_fee_per_fill`, charged per fill on both sides |
| `SpreadPolicy` | `half_spread_bps`, `fixed_half_spread_price` | execution crosses `mid × half_spread_bps/10000 + fixed_half_spread_price`; the quoted spread is twice that |
| `SlippagePolicy` | `fixed_bps`, `impact_bps_at_full_capacity` | `bps = fixed_bps + impact_bps_at_full_capacity × (fill_qty / capacity_qty)` |
| `CapacityPolicy` | `max_participation_fraction`, `max_order_quantity`, `max_order_notional`, `on_order_exceeds_limit` | per-observation capacity is `quote.volume × max_participation_fraction`; order-level caps clamp or reject |
| `FillPolicy` | `reject_probability`, `fill_probability`, `minimum_fill_quantity`, `require_inventory_for_sell`, `order_time_to_live_ns` | seeded independent draws per order and per execution opportunity |
| `LatencyPolicy` | `decision_to_order_ns`, `order_to_ack_ns`, `ack_to_execution_ns` | additive, fixed, no jitter and no queue model |
| `MarkPolicy` | `fallback`, `price_source` | valuation fallback (`cost_basis`, `zero`, `last_fill_price`) and mark/last/mid selection |
| `AccountPolicy` | `require_sufficient_cash`, `on_insufficient_cash`, `cash_buffer_fraction` | closes the P2-flagged spot buying-power gap: reduce or reject an unaffordable purchase |
| `PerpPolicy` | leverage, margin mode, position mode, maintenance rate, funding interval/anchor/source/rates, liquidation fee, curve tracing | the perpetual-contract layer described above |

`ExecutionPolicy` bundles all of them and is serialized verbatim into every replay result and into
each receipt, so a receipt always identifies the exact policy that produced it. `ContractSpec`
declarations are serialized alongside it.

### Determinism

All synthetic variability is `uniform_draw(seed, purpose, *key_parts)`, a SHA-256-derived value in
`[0, 1)` that depends only on its explicit key parts. It never uses `random`, wall-clock time,
dictionary iteration order, or Python's per-process string hash. Identical quotes, decisions, policy
and seed therefore produce a byte-identical ledger (`ledger_sha256` is recorded for both runs in the
receipt), and a different seed changes the ledger whenever any draw is actually consulted.

## Ledger and conservation

The ledger tracks cash, explicit fees, liquidation fees, funding cost, allocated margin, signed
positions, average-cost basis, realized PnL (total and per asset), funding cost (total, per asset and
per position key), turnover per asset, mark-to-market value, unrealized PnL, equity and net PnL, plus
append-only lists of cash entries, fills, funding payments and liquidations.

`conservation_report(ledger)` is an **independent second implementation** of the accounting: it
replays the raw fill list into signed positions, cost basis, realized PnL and fees, replays the
funding list, and replays the liquidation-fee entries, then compares that replay against the stored
aggregates. `require_conservation` always recomputes (the embedded report is a receipt, not proof).

Checked identities, each with a `1e-9 × max(1, initial_cash, gross_traded_notional, |cash|, Σ|cash flows|)`
tolerance:

- `cash == initial_cash + Σ cash-entry amounts`
- cash entries reconcile with the fill, funding and liquidation-fee cash flows (`-= direction ×
  notional` for a fill, the funding amount, the negative liquidation fee)
- `fees_paid == Σ fill fees`; `liquidation_fees_paid == −Σ liquidation-fee entries`;
  `funding_cost == −Σ funding payments`; funding payments reconcile with their cash entries
- signed position quantities and cost bases equal the fill replay
- `realized_pnl` equals the fill replay
- `equity == cash + Σ market value`
- `net_pnl == equity − initial_cash`
- `net_pnl_excluding_explicit_fees == net_pnl + fees_paid`
- `net_pnl_excluding_all_costs == net_pnl + fees_paid + liquidation_fees_paid + funding_cost`
- every liquidation fill leaves its position key flat

Tampering with any fill, funding payment, liquidation fee, cash entry, position valuation or
aggregate makes the recomputation disagree and raises `LedgerConservationError` (covered by unit
tests, including the new perp paths).

## No-future-information guard

`NoFutureInformationGuard` gates both boundaries and raises `FutureInformationError` with a recorded
violation:

1. **Decision boundary (always on).** A decision may not consume a price that became available after
   `decision_ns`.
2. **Execution boundary (always on).** An execution may not consume a price that became available
   after its own `execution_ns`, and may not be timestamped before its order's submission or its
   decision.
3. **Strict switch (`forbid_post_decision_prices_in_executions`, off by default).** Forbids *any*
   execution from using a price timestamped after the decision.

Liquidation fills also pass through the guard, using the availability timestamp of the exact mark
observation that triggered them. Interpretation note, because this matters for R1: the third switch
is the literal reading of "no execution may use a price timestamped after the decision", but with
nonzero latency every realistic fill necessarily uses a post-decision price — that is the causal
order of events, not leakage. The default therefore enforces the leak that can actually corrupt a
decision (switch 1) plus the impossibility checks at execution (switch 2), and exposes switch 3 as an
explicitly tested strict mode for a zero-latency contract.

## Out-of-model risk controls

`RiskEngine.evaluate(context)` is deterministic, model-free, and returns `allow` / `reduce` /
`block` with `allowed_quantity` and machine-readable reason codes. Per-guard entries distinguish a
hard `block`, a quantity-limiting `allow`, and `not_evaluated` (the context omitted the inputs).

| Guard | Reason code | Behaviour |
|---|---|---|
| Kill switch | `kill_switch_engaged` | Blocks everything until explicitly released; optional latching on first block |
| Missing/invalid context | `missing_context_field`, `nonfinite_input`, `unknown_side`, `nonpositive_requested_quantity`, `invalid_reference_price` | Fail closed |
| Stale data | `stale_data`, `data_timestamp_in_future` | Blocks when data age exceeds the limit or the input timestamp is ahead of the clock |
| Clock drift | `clock_drift` | Blocks when the feed's own wall clock differs from the local clock beyond the limit |
| Position | `position_limit` | Reduces to the remaining allowance, blocks at zero, and refuses to cross `min_position_quantity`. **Perp use requires `min_position_quantity < 0`; the default `0.0` forbids shorts, which is fail-closed by design** |
| Gross / net exposure | `gross_exposure_limit`, `net_exposure_limit` | Reduces or blocks. Gross now moves with the change in absolute net exposure, so opening a short increases gross exactly as opening a long does |
| Loss | `net_loss_limit`, `realized_loss_limit` | Blocks when net or realized loss exceeds the limit |
| Turnover | `turnover_limit` | Reduces against the trailing-window turnover budget |
| Order rate | `order_rate_limit` | Blocks once the trailing-window order count reaches the cap |
| Order notional | `order_notional_limit` | Reduces or blocks |
| Leverage | `leverage_limit`, `contract_leverage_limit` | Blocks a declared leverage above the cap or the contract maximum, and reduces projected account leverage |
| Margin sufficiency | `insufficient_margin` | Reduces to the affordable opening notional, blocks at zero, and fails closed when available margin is below the declared minimum |
| Margin ratio | `margin_ratio_limit` | Blocks risk-increasing orders once `margin_ratio ≥ max_margin_ratio`; risk-reducing orders pass |
| Liquidation distance | `liquidation_distance_limit` | Blocks risk-increasing orders once the buffer to the trigger falls below the declared minimum |
| Funding cost | `funding_cost_limit` | Blocks risk-increasing orders once cumulative (plus declared expected) funding cost exceeds the cap |
| Venue position | `venue_position_limit` | Reduces or blocks against the per-venue notional cap |

The simulator accepts any risk gate object with `.evaluate(context)` and optional
`.record_order(context)` / `.record_fill(context, notional)` hooks; `RiskEngine` implements all
three. The context is a plain documented mapping, so the two modules stay decoupled and the guards
can be tested without the simulator. Absent optional perp keys are never evaluated, so passing a
spot-only context reproduces the earlier guard behaviour exactly.

## Measured synthetic results

Receipt: [financial_simulator_v1_stress.json](../results/financial_simulator_v1_stress.json), seed
`20260919`, generated in well under a second. All values below are synthetic.

The receipt file is **byte-identical across separate processes** for the same seed
(`deterministic_receipt: true`); wall-clock timing is deliberately excluded from the file and is
reported on the CLI stream instead, so re-running the command cannot produce a diff.

| Observation | Value |
|---|---|
| Determinism | two runs, identical `ledger_sha256`, `byte_identical: true` |
| Conservation | `ok: true`, every residual within the `0.001` tolerance (largest is float noise `2.3e-10`) |
| Baseline scenario | 60 decisions (10 abstain, 10 no-trade), 35 orders, 35 fills, 3 risk-blocked, 2 rejected |
| Decision-count closure | every scenario closes: `abstain + no_trade + risk_blocked + rejected + no_order + ordered == decisions` |
| Rejection scenario | 18 orders, 18 rejected, 0 fills |
| Partial scenario | 33 partially filled orders, 583 fills, every one expiring with a remainder |
| No-fill scenario | 18 orders, 0 fills, all expired |
| Order statuses observed | all six: `submitted`, `accepted`, `rejected`, `partially_filled`, `filled`, `expired` |
| Cost monotonicity | fee sweep `0 → 50 bps` gives net PnL `−5.64 → −189.05`, zero violations |
| Leak guard | 200/200 deliberately leaked decisions blocked; strict-mode and post-execution leaks blocked |
| Short round trip | short 10 at 100, close at 90 → realized and net PnL exactly `+100`, position flat, conservation ok |
| Funding monotonicity | rates `0 → 1e-3` give funding cost `0 → 6.0` (6 payments each) and net PnL `0 → −6.0`, zero violations |
| Liquidation | one isolated liquidation at `90.452` (analytic trigger), `position_after = 0.0`, `margin_below_maintenance_requirement`, conservation ok |
| Margin modes | same path, same size: isolated `1` liquidation, final equity `9519.60`; cross `0` liquidations, final equity `7273.98` |
| Leverage effect | fixed size across `1x/2x/5x/10x`: net PnL identical (spread `0.0`), required initial margin `1000/500/200/100` |
| Contract multiplier | `1x → 4x` scales the PnL by exactly `4` (ratio error `0.0`) |
| Margin sufficiency | a 500-unit order on a 1000-unit account at 10x is reduced to `100` units, allocated margin `1000`, reason codes `insufficient_initial_margin` + `initial_margin_quantity_reduce` |
| Netting | one-way: 1 position key, net `+3`; hedge: 2 leg keys, `+5` long and `−2` short; both conserve |
| Paper backtest | 39 fills, 6 funding payments, 0 liquidations, 2 instruments, 4 regime labels, 238 curve points, turnover `4,689,151`, max drawdown `0.99%`, attribution residuals `4.2e-11` and `0.0` |
| Backtest determinism | two runs, identical `backtest_sha256`, equity curve / fills / funding identical |
| Risk guards | every spot and perp guard returned its expected `allow`/`reduce`/`block` with reason codes; a fully blocking gate produced zero fills |
| Hand-computed reference | one-fill cases match independently computed price, spread cost, fee, cash, cost basis, unrealized PnL and net PnL to 12 decimal places; the short round trip and the multiplier scaling are closed-form |

## Parameters that must be frozen at review gate R1

Every entry below is a guess in the code today and is marked **provisional pending R1**. R1 must give
each one a sourced or explicitly conservative value together with its admissible range and the data
that justifies it. The receipt lists all of them under `provisional_pending_r1` (71 entries).

**Execution costs** — provisional pending R1
`fees.fee_bps`, `fees.fixed_fee_per_fill`, `fees.minimum_fee`, `spread.half_spread_bps`,
`spread.fixed_half_spread_price`, `slippage.fixed_bps`, `slippage.impact_bps_at_full_capacity`.

**Capacity and fills** — provisional pending R1
`capacity.max_participation_fraction`, `capacity.max_order_quantity`, `capacity.max_order_notional`,
`capacity.on_order_exceeds_limit`, `fills.reject_probability`, `fills.fill_probability`,
`fills.minimum_fill_quantity`, `fills.require_inventory_for_sell`, `fills.order_time_to_live_ns`.

**Latency and valuation** — provisional pending R1
`latency.decision_to_order_ns`, `latency.order_to_ack_ns`, `latency.ack_to_execution_ns`,
`marks.fallback`, `marks.price_source`.

**Account and cash sufficiency** — provisional pending R1
`account.require_sufficient_cash`, `account.on_insufficient_cash`, `account.cash_buffer_fraction`.

**Perpetual contracts** — provisional pending R1
`perps.allow_short`, `perps.default_leverage`, `perps.max_leverage`, `perps.margin_mode`,
`perps.position_mode`, `perps.maintenance_margin_rate`, `perps.funding_interval_ns`,
`perps.funding_anchor_ns`, `perps.funding_rate_source`, `perps.default_funding_rate`,
`perps.funding_rates`, `perps.funding_rate_schedule`, `perps.liquidation_fee_bps`,
`perps.trace_perp_curve`, and per declared contract `contracts.*.contract_multiplier`,
`contracts.*.tick_size`, `contracts.*.lot_size`, `contracts.*.min_notional`,
`contracts.*.margin_mode`, `contracts.*.max_leverage`, `contracts.*.maintenance_margin_rate`.

**Leak-guard switches** — provisional pending R1
`guard.forbid_post_decision_prices_in_decisions`, `guard.forbid_post_execution_prices_in_executions`,
`guard.forbid_post_decision_prices_in_executions`.

**Risk limits** — provisional pending R1
`limits.max_position_quantity`, `limits.min_position_quantity`, `limits.max_gross_exposure_notional`,
`limits.max_abs_net_exposure_notional`, `limits.max_net_loss_notional`,
`limits.max_realized_loss_notional`, `limits.max_turnover_notional`, `limits.turnover_window_ns`,
`limits.max_orders_per_window`, `limits.order_rate_window_ns`, `limits.max_order_notional`,
`limits.max_data_staleness_ns`, `limits.max_clock_drift_ns`, `limits.require_data_quality_fields`,
`limits.latch_kill_switch_on_block`, `limits.max_leverage`,
`limits.max_position_notional_per_venue`, `limits.max_funding_cost_notional`,
`limits.max_margin_ratio`, `limits.min_available_margin_notional`,
`limits.min_liquidation_distance_fraction`, `limits.max_initial_margin_utilization`,
`limits.require_margin_sufficiency`.

**Paper-backtest construction** — provisional pending R1
`path.ticks`, `path.tick_ns`, `path.feed_delay_ns`, `path.regime_length_ticks`, `path.initial_price`,
`path.drift_bps_per_tick`, `path.noise_bps_per_tick`, `path.volume`, `path.mark_basis_fraction`,
`path.tick_size`, `path.lot_size`, `path.min_notional`, `strategy.fast_ticks`, `strategy.slow_ticks`,
`strategy.target_quantity`, `strategy.rebalance_ticks`, `strategy.threshold_fraction`,
`strategy.allow_short`, `backtest.initial_cash`.

### Contract decisions R1 must also confirm (not yet parameters)

- **Venue rule fidelity.** Venue names declare the modelled market structure only. R1 must decide
  whether any venue-specific rule (tick/lot/min-notional tables, funding schedule, margin tiers,
  liquidation fee, price bands, ADL) needs to be sourced and pinned, or whether the simulator keeps
  declaring a synthetic contract shape and restricting its applicability accordingly.
- **Margin tiers and portfolio margin.** The maintenance requirement is a single declared rate.
  Real venues tier it by notional and some offer portfolio margin; neither is modelled.
- **Liquidation execution.** Synthetic forced closure happens at the mark price plus a declared fee.
  Close-out slippage, insurance funds, ADL and bad-debt socialisation are unmodelled, and an
  isolated loss is reported even when it exceeds the allocated margin.
- **Funding conventions.** The funding interval, anchor, rate source and mark-price notional are all
  declared. R1 must confirm the interval/anchoring convention, the rate's provenance, and whether
  the interest-rate component and clamp need modelling.
- **Position scope.** Under one-way netting, positions are aggregated per `asset_id` across venues;
  venue segregation is deferred. Hedge mode keeps two legs per asset. R1 must decide whether the V1
  state contract needs venue-segregated inventory and whether hedge mode is required at all.
- **Fill model shape.** Fills are a deterministic seeded draw times a participation cap, not a
  queue-position or order-book reconstruction. R1 must confirm that perpetual contracts can be
  defended with this conservative model, or restrict the simulator's declared applicability.
- **Reference-price convention.** The reference is the mid of the newest available quote. R1 must
  confirm mid versus last versus microprice and the exact convention for the traded horizon.

## What this cannot prove

The simulator detects ordering and accounting errors, not dishonest inputs. It cannot establish that
a supplied quote was genuinely available at its declared timestamp, that a fill or liquidation would
have occurred, that fees, spreads, funding rates or margin tiers resemble any real venue, that
capacity and impact are realistic, or that any strategy is profitable after costs. Watch the specific
failure modes this design deliberately leaves open:

- seeded draws are a **model**, not evidence of a venue's fill distribution; changing the seed
  changes fills;
- a partially filled synthetic order may be an artefact of the participation cap rather than of queue
  dynamics;
- spreading costs into fill prices means "net PnL excluding explicit fees" still contains spread and
  slippage, and must never be quoted as a gross return;
- the margin, funding and liquidation model is a declared synthetic contract shape; it is not a
  replication of any venue's margin engine, and the equity curve is sampled at synthetic observation
  times, so intra-interval excursions are not represented;
- per-instrument and per-regime attributions prove that the receipt's arithmetic closes, not that a
  regime or instrument explains performance;
- synthetic stress coverage proves guard *behaviour*, never market realism;
- a paper-trading receipt is not shadow trading, not a live-feed trial, and not authorisation for
  live capital.

## Invariants this suite does not test

Stated explicitly so they are not mistaken for verified properties:

- **No assertion about real venue parameters.** Nothing checks a Binance/Bybit/Aster/Hyperliquid
  tick table, leverage tier, funding schedule, liquidation fee or price band against any source.
- **Point-in-time authenticity of a real feed** is out of reach here entirely (P1 source/rights
  audit).
- **No cross-venue netting or transfer test.** Venue cross-margin between venues is not modelled and
  therefore not tested.
- **No inverse-contract, quanto or portfolio-margin test** — those products are declared out of
  scope rather than approximated.
- **No queue-position, order-book reconstruction, latency-jitter, or partial-liquidation test.**
- **No test that the reference strategy has any predictive value**, and no test of a real PnL
  distribution: the reference strategy exists to exercise the harness, not to be a signal.
- **No loss-cap test in isolated mode.** The suite verifies that a breach flattens the leg and that
  the ledger conserves; it does not verify that the realized loss is bounded by the allocated margin,
  because that bound is deliberately not modelled.

The next step after R1/R2 is the deterministic and supervised baseline work in P3, evaluated under
this same simulator, with gross and net results, turnover, drawdown, tail loss, exposure, calibration
and abstention reported separately. No strategy result may be described as profitable from a
synthetic tape.
