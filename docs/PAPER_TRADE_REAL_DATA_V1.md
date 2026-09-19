# Real-data paper trading V1 (Binance USDT-M perpetuals)

Status: **real venue data, simulated execution, complete and conserving — and explicitly NOT a result.**

This page records the first end-to-end paper-trading run over real perpetual-contract data
instead of a constructed path. Read the honesty section before quoting any number from it.

**Licence.** The Binance data used here is the public Binance Vision archive, whose Dataset
Terms v1.0 place it under **CC BY-NC-SA 4.0 — non-commercial**. Research use is permitted
(§4.1); **live proprietary trading execution and automated commercial order generation are
prohibited** (§4.2), and commercial licensing is excluded entirely (§3.4). Everything on this
page is offline, non-commercial research.

For the other three venues, see the [venue data licensing audit](VENUE_DATA_LICENSING_V1.md):
**Aster's terms were read in full** and §6.1(b)/§6.2(e) require prior written consent to
download their material and express permission for automated access, with **no research
carve-out** — so using Aster data here is a recorded, unresolved conflict, not a clearance.
**Bybit and Hyperliquid terms could not be read** and remain **unverified**. Data from those
venues is used only under the project owner's explicit research authorisation, which settles the
project's decision but is not the venue's permission.

**No profitability claim.** Nothing on this page is evidence of profit, edge, skill or
tradability, and no figure here may be presented as a return. See "What this does NOT
establish" — the headline result is that the number is not stable enough to be a result at
all.

## What was actually done

| Step | Artifact | Result |
|---|---|---|
| Bounded archive fetch | `scripts/fetch_binance_vision_v1.py` → `data/binance_vision_v1/` | 880 files, 1,185,364 B, 5 symbols, 2023-01..2026-08 |
| Multi-venue fetch | `scripts/fetch_venue_perp_v1.py` → `data/venue_perp_v1/` | Bybit 30 files / 3,734,650 B; Hyperliquid 15 files / 14,131,314 B; **Aster 35 files / 4,032,704 B** (all via egress proxy) |
| PIT cohort build | `scripts/build_perp_pit_v1.py` → `data/perp_pit_v1/records.jsonl` | 6,554 records, 2,951 positives (45.03%) |
| Point-in-time audit | unmodified `scripts/financial_pit_v1.py` + the FROZEN protocol core | exit 0; all four phases nonempty in all three folds |
| Paper trading (Binance) | `scripts/paper_trade_perp_v1.py --source binance` → `results/paper_trade_perp_binance_v1.json` | 83 fills, 10,251 funding payments, 0 liquidations, conservation OK |
| Paper trading (Bybit) | `scripts/paper_trade_perp_v1.py --source bybit` → `results/paper_trade_perp_bybit_v1.json` | 81 fills, 9,123 funding payments, 0 liquidations |

The validator used in step 4 is **unmodified**, and the protocol core is projected verbatim
from the R1 draft; no frozen artifact was edited.

### A pagination bug found and fixed while connecting Bybit

The first Bybit fetch returned exactly **1,000 days per series**. Bybit's v5 kline endpoints
return the **most recent** `limit` rows inside the requested window, so paging forward by
`start` silently truncates history at one page. The fetcher now pages **backward** by moving
`end` to just before the oldest row received. The refetch grew from 1,165,457 to **3,734,650
bytes (3.2×)**, and the Bybit-sourced run went from 61 to 83 decisions, matching Binance's
trigger count. Any earlier cross-venue comparison would have been made on unequal ranges.

## Venue connectivity (the honest status)

| Venue | Status | Detail |
|---|---|---|
| **Binance** | Connected | Public bulk archive, no key. Licence read in full. |
| **Bybit** | Connected | `api.bybit.com` is **DNS-poisoned in this environment** (resolves to 179.60.193.16, an unrelated address, and never connects). Via the local egress proxy the **official** host works. |
| **Hyperliquid** | Connected | `POST api.hyperliquid.xyz/info` (`metaAndAssetCtxs`, `candleSnapshot`, `fundingHistory`). Reachable directly. |
| **Aster** | **Connected (via proxy)** | `fapi.asterdex.com` refuses **direct** connections from this environment on every probed host. Through the local egress proxy it returns real data: klines, mark/index klines, funding, `fundingInfo`, `premiumIndex`, `exchangeInfo` (602 symbols). |

### The proxy is what unblocked Bybit and Aster

Both blockers were **egress problems in this environment, not venue-side blocks**:

| Host | Direct | Via `http://127.0.0.1:7890` |
|---|---|---|
| `api.bybit.com` | HTTP 000 (DNS-poisoned to an unrelated address) | **HTTP 200** |
| `fapi.asterdex.com` | HTTP 000 (connection refused) | **HTTP 200** |
| `api.hyperliquid.xyz` | HTTP 200 | HTTP 200 |

The fetcher therefore routes through a local egress proxy by default
(`NANOJEV_PROXY`, default `http://127.0.0.1:7890`, overridable with `--proxy`, and
`--proxy ""` forces direct connections). The proxy used is recorded in every fetch manifest
as `egress_proxy`, so a fetch is repeatable.

**Limits of this claim:** the proxy solves *reachability*. It does **not** verify that any
venue's terms permit the use — those terms pages remain unread from here, so Bybit, Aster and
Hyperliquid permissions stay **unverified**, and the project owner's authorisation remains a
project decision rather than a legal determination. Commercial and live-execution use stays
unlicensed and out of scope.

## PIT audit on real data

The frozen protocol core (asof 2027-03-01, 3 folds, one-sided embargo) accepted the real
cohort:

| Fold | train | dev | calibration | test |
|---|---:|---:|---:|---:|
| 1 | 3,744 | 120 | 120 | 905 |
| 2 | 4,649 | 120 | 120 | 905 |
| 3 | 5,554 | 120 | 120 | 465 |

Fold 3's test window extends past the last available bar, which is why it retains fewer
rows. Exclusions are reported per fold (`outside_windows`, `label_unavailable_at_fit_cutoff`,
`purged_overlap_or_embargo`).

Reproduce:

```bash
.venv/bin/python -c "import json,pathlib; json.dump(json.loads(pathlib.Path('research/financial_experiment_protocol_v1.json').read_text())['pit_validator_core'], open('/tmp/frozen_core.json','w'))"
.venv/bin/python scripts/financial_pit_v1.py --input data/perp_pit_v1/records.jsonl --protocol /tmp/frozen_core.json --output /tmp/fold_audit.json
```

## The pilot cohort is not the R1 cohort

The R1 feature allowlist is a **closed 12-feature set**: "no feature may be added, removed,
or conditionally omitted for any record". Three of those features cannot be built from this
archive:

| Omitted feature | Why |
|---|---|
| `open_interest_level` | needs the daily metrics archive, not downloaded in this slice |
| `open_interest_log_change_1d` | same |
| `liquidation_intensity_1d` | **no venue in the reviewed set publishes historical liquidations** |

So this cohort declares a documented **9-feature subset** and is labelled a PILOT. It must
not be presented as the frozen R1 cohort. Resolving which way to go (download metrics and
drop only the liquidation feature, or amend the allowlist at R1) is a gate decision.

## Paper-trading run

Reference strategy: mechanical 20/60 moving-average crossover, two-sided (long and short),
3× declared leverage, cross margin, 5 bps fee, 1 bps half-spread, 2024-01-01..2026-08-31,
BTCUSDT/ETHUSDT/SOLUSDT, 100,000 initial cash.

### The headline is that this is NOT a result

The same strategy over the same period and instruments produced **four mutually
contradictory numbers** depending on data set and venue:

| Run | Data | Fills | Net PnL | Max drawdown |
|---|---|---:|---:|---:|
| A | Binance, before a 3-day index-data fix | 82 | **−49,920.90** | — |
| B | Binance, after the fix | 83 | **+21,173.77** | 78.86% |
| C | Bybit (same period, same instruments) | 81 (2 rejected) | **−77,919.27** | 94.22% |
| D | Aster (same period, same instruments) | 83 (27 partial) | **+194,025.96** | 64.79% |

**The spread of net PnL across venues is 271,945 on 100,000 of initial capital — 2.7× the
capital itself.** Runs A and B differ only by three BTCUSDT days (2023-02-13, 2023-04-07,
2023-04-08) and by exactly one decision that was rejected in A and executed in B: **that
single decision flips the sign.**

### The divergence is NOT a data-quality problem

Before blaming the venues, their mark prices were cross-checked directly:

| Comparison | Common days | Mean absolute deviation |
|---|---:|---:|
| Aster vs Bybit mark close | 1,358 | **0.0199%** (≈2 bps) |
| Aster vs Binance mark close | 1,337 | **0.0122%** (≈1.2 bps) |
| Worst single day (Aster vs Bybit, 2024-10-13) | — | 0.199% (≈20 bps) |

The venues agree to within roughly **one to two basis points**. Yet the same strategy on
those three price series returns +194,026, +21,174 and −77,919. **The strategy is chaotic with
respect to its input** — a one-basis-point difference in the price path compounds through
position sizing and the reject/partial-fill path into a 2.7× swing in final PnL.

So the ~2 bps of genuine cross-venue basis is not the cause of a 271,945 spread; it is the
trigger. **No number on this page may be quoted as a return, a loss, an edge or a result.**

### The mechanism to fix

The strategy tracks an intended target position and issues absolute-quantity orders, so a
**rejected or partially filled** order silently breaks the intended position path and the
strategy then compounds from an assumed position it does not hold. Across the three venues
that produced 0, 2 and 27 such events respectively. A robust harness must treat any
divergence from the intended fill as an explicit state error (or use close-only
`position_effect` semantics) before any inference is possible.

A second amplifier: the quantity is recomputed from current price each time
(`0.25 × equity × leverage / price`), so the position size compounds with the path rather
than being fixed.

A third, venue-specific amplifier explains Aster's 27 partial fills. Aster reports far lower
daily volume than the other two venues:

| Venue | Median daily bar volume (last 30 bars, BTCUSDT) |
|---|---:|
| Binance | 128,881.6 |
| Bybit | 57,314.3 |
| **Aster** | **9,959.5** |

The capacity policy caps a fill at a fraction of the observed bar volume, so on Aster the cap
bites and orders fill partially. That is a modelling interaction, not a data error — but it
means **the same strategy receives different position sizes on each venue**, a third
independent reason the per-venue results are not comparable.

### Full ledger for run B (Binance)

| Measure | Value |
|---|---:|
| Quotes / decisions / fills | 4,008 / 83 / 83 |
| Funding payments | 10,251 |
| Liquidations | 0 |
| Initial cash → final equity | 100,000.00 → 121,173.77 |
| Net PnL | +21,173.77 |
| Fees paid | 4,722.50 |
| Funding cost | 27,167.86 |
| Gross traded notional | 9,445,009.73 |
| Max drawdown | 117,970.85 (78.86% of peak) |
| Longest drawdown | 529,632 s ≈ 6.1 days of curve time |
| Conservation check | **ok** |

Cost decomposition again: funding (27,167.86) is **5.8×** the explicit fees (4,722.50). The
gross move before costs is therefore strongly positive in this run and costs consume a large
share of it — but per the section above, none of that should be believed.

### Per-regime breakdown (run B)

Regimes are **causal**: expanding quantiles over trailing observations only, nearest-rank, no
interpolation, with a 48-bar minimum history. Priority order: basis blowout → funding extreme
→ high vol → low liquidity → normal. The schedule is keyed off one reference instrument
(BTCUSDT) and is descriptive, not an independent experiment.

| Regime | Observations | Equity change | Funding | Fees | Fills | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| basis_blowout | 476 | +67,939.98 | 11,096.48 | 196.66 | 4 | 113,514.16 |
| vol_high | 832 | −58,539.96 | 3,488.59 | 706.43 | 11 | 117,106.13 |
| normal | 2,727 | +33,180.07 | 7,015.57 | 2,779.51 | 47 | 117,970.85 |
| liquidity_low | 955 | +1,920.64 | 1,769.86 | 967.31 | 19 | 114,514.21 |
| funding_extreme | 120 | −32,822.14 | 3,797.35 | 0.00 | 0 | 87,118.72 |
| unlabeled | 2 | +3,979.57 | 0.00 | 0.00 | 0 | 55,102.12 |

**Attribution is coherent**: the regime equity changes sum to the curve's total equity change
with a residual of −2.18e-11 (float noise). Note the attribution is relative to the first
curve point, not to the initial cash.

The two regimes that carry most of the loss (`vol_high`, `funding_extreme`) are exactly the
conditions a cost-blind backtest would ignore — but with only 11 and 0 fills respectively,
these buckets are far too thin to support any inference.

## What this does NOT establish

- **Not a result at all.** The sign flips across a three-day data difference and across two
  venues. Any number on this page is a plumbing artefact.
- **Not a return.** Execution costs are declared provisional guesses pending R1. The archive
  has no order book, so bid/ask equal the mark close and the entire spread cost is one
  declared parameter.
- **Not predictive.** The reference strategy is a mechanical execution-path exerciser. Its
  loss is not evidence about any model, and a profit would not have been evidence either.
- **Not an as-of vintage.** The archive is a current snapshot; Binance documents in-place
  file replacement. Point-in-time authenticity is not proven.
- **Not live-ready.** No fill authenticity, queue position, capacity, insurance fund, ADL,
  partial liquidation or venue-parameter fidelity is modelled.
- **Fill and funding conventions are approximations**: a decision at bar `d`'s close fills at
  bar `d`'s mark price (the newest available observation), and each funding boundary charges
  the most recent *settled* rate rather than the rate settling at that instant.
- **Regime buckets are descriptive, not experiments.** The schedule is keyed off one
  reference instrument and several buckets hold under 20 fills.
- **The rejected-order path is a known harness weakness**, described above; it must be fixed
  before any result could carry meaning.
- **No public claim of any kind** should be derived from this page.

## Reproduction

```bash
.venv/bin/python scripts/fetch_binance_vision_v1.py --first-month 2023-01 --last-month 2026-08
.venv/bin/python scripts/fetch_venue_perp_v1.py --venue bybit --start 2023-01-01 --end 2026-09-19
.venv/bin/python scripts/fetch_venue_perp_v1.py --venue aster --start 2023-01-01 --end 2026-09-19
.venv/bin/python scripts/build_perp_pit_v1.py
.venv/bin/python scripts/paper_trade_perp_v1.py --source binance --output results/paper_trade_perp_binance_v1.json
.venv/bin/python scripts/paper_trade_perp_v1.py --source bybit   --output results/paper_trade_perp_bybit_v1.json
.venv/bin/python scripts/paper_trade_perp_v1.py --source aster   --output results/paper_trade_perp_aster_v1.json
```

All fetches are public read-only data calls. The replay is fully offline; it contacts no
venue, opens no socket, and can never place an order or read an account. Re-running the same
command twice produces a **byte-identical receipt** (verified: `replay_sha256`, `net_pnl` and
the entire receipt compare equal).

Determinism is verified; **robustness is not** — see the three-run table above. A pipeline
that reproduces the same wrong-sign number exactly is still not evidence.
