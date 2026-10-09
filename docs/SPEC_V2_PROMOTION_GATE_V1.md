# Spec v2 Promotion Gate v1 — xs_dfh_carry_v1

T155, declared 2026-09-26. Machine-readable thresholds: `research/spec_v2_promotion_gate_v1.json`.
Subject: `research/financial_signal_spec_xs_dfh_v1.json` (status `spec_candidate_pending_review`,
frozen 2026-09-25). Evidence arm: `results/forward_ledger_v2.jsonl` (xs_v1 10-asset universe, the
spec's own), confirmation arm `results/forward_ledger_v2_xs2.jsonl` (xs_v2 30-asset).

## What promotion means

Promotion = **frozen measurement contract**: the signal definition locks for continued append-only
paper accounting. It is still paper-only, never a trading authorization, never RLCD/training
authorization. Thresholds below are predeclared and must not be revised to fit accumulating
evidence; changes require a new versioned gate file.

## Thresholds and their measured justification

| Gate | Value | Measured anchor |
|---|---|---|
| `min_oos_days` / `min_active_days` | 90 / 30 | Ledger records one row per calendar decision day; at the measured ~53% occupancy (in-sample 708/1313 = 0.539; ledger backfill 0.533) 90 days yields ~48 active days — enough for a sign/Sharpe read but deliberately thin; promotion remains a paper contract, not a significance proof |
| `net_sharpe_floor` | ≥ 0.5 ann. | Half the T106 in-sample net Sharpe 0.986 — allows ~2x OOS decay while still requiring the edge to be net-positive after the 5bps/leg cost model |
| `min_cumulative_net_bps` | > 0 | T106 measured +13.467 net bps/day; a quarter of post-freeze accounting must clear break-even |
| `max_drawdown_bps` | ≤ 1200 | ~2x the measured full-history worst week (−604.4bps) and ~70% of the 1363-day max drawdown (1717.7bps) — a single quarter approaching lifetime DD means the edge is not surviving OOS |
| `gate_occupancy` | [0.30, 0.80] | Measured 0.499–0.539 across both universes. Lower bound guarantees ≥27 active days of evidence; upper bound catches a stuck-on gate — the gate is load-bearing (ungated days measured −13.8bps) |
| `max_worst_week_share_of_net` | ≤ 0.50 | Current context-window worst week −410.6bps was 93% of that window's +439.2bps net — the fragility this cap rejects; full-history it is only 5.5% |
| `max_best_week_share_of_net` | ≤ 0.40 | Full-history best week is 11.5% (v1) / 15.2% (xs2) of cum net; T115 anatomy showed 65.6% PnL concentration = noise, not edge |
| Comparator: ungated | `net_gated > net_ungated` | T106: ungated does not survive. Fresh-replay context window: ungated −84.3bps (xs_v1) / +26.5 (xs2) vs gated +439.2 recorded / +745.3 replay |
| Comparator: EW-long | gated net > 0 in each EW-sign subwindow (≥10d) | Dollar-neutral book must not be beta in disguise; the context window's EW-long Sharpe was 5.06 (+2102.6bps in 25d) — a bull tape makes this check necessary |
| Comparator: spec v1 sleeve | both sleeves net > 0 over window | Shared BTC master gate; simultaneous survival is the regime-gate sanity check (v1: 9 exits, +802.4bps in context window). bps magnitudes are not comparable across the two accounting conventions |
| Probe freshness | ledger age ≤72h, lag ≤5d, probe ≤24h | `forward_ledgers_all_v1_state.json`: launchd 07:10 daily chain, probe default 48h + slack for open-month daily-zip gaps |
| Secondary arm | xs_v2 net > 0, same sign | Same edge measured on both universes; xs2 may not contradict |

## Kill / demote branch

- **Day-60 checkpoint**: cumulative net < 0 → demote to `spec_candidate_under_review`.
- **Hard kill (owner sign-off, retire to §3 graveyard)**: cum net < −800bps (~1.3x measured worst
  week); drawdown > 1200bps; or OOS book-level spearman(dfh20 rank, next-day return) < 0 with
  ≥30 active days — a sign flip means the ranked edge inverted.
- **Demote**: ungated ≥ gated at evaluation (gate dead); occupancy outside [0.15, 0.90] for ≥20
  consecutive days (instrumentation/regime break — pause counting, not a signal verdict); xs2 arm
  net < 0 while primary passes.
- **Infra pause**: staleness > 7d or missing input dates suspends counting until refresh recovers.

## Blocking precondition found while writing this gate

The live ledger gates on **BTC close > SMA20** (`financial_forward_ledger_v2.py`,
sleeve `xs_dfh20_top2_bot2_btc_sma20_gate_v1`) while the spec contract gates on **ret20 > 0**
(close/close[t-20]−1; the T109/T114-measured gate). The two definitions disagree on **211/1363
(15.5%)** of xs_v1 decision days, and on 2 days (2026-09-09, 09-11) inside the current context
window. Until this is resolved — rebuild the ledger under ret20 (preferred, matches measured
evidence) or amend the spec to SMA20 with re-measured gated stats — **no OOS day formally counts**.

## Progress snapshot (as of probe run 2026-09-26)

| Metric | Value |
|---|---|
| Spec freeze | 2026-09-25T00:00:00Z |
| Last ledger decision_date | 2026-09-24 (both arms) |
| Post-freeze OOS days accumulated | **0 / 90** — the gate clock has not started; earliest full evaluation ≈ late Dec 2026 |
| Context segment 2026-08-31→09-24 (25d, recorded, not counted) | v2 net +439.2bps, mean +17.6/d, Sharpe ~2.7, mdd 563.8bps, occupancy 16/25 = 64%; xs2 net +652.5bps, mdd 484.9bps |
| Context-segment comparators (fresh replay) | ungated −84.3bps; EW-long +2102.6bps; v1 sleeve +802.4bps (9 exits) |
| Caveat | recorded ledger vs fresh replay differed ~355bps over the segment — known append-only late-bar repricing; recorded rows are the accounting of record |

Directional read: the refresh-fed segment is net-positive with the gate adding value vs ungated,
but 93% of that net was eroded within a single week and the segment is 25 backfilled days — context,
not evidence. The gate decision waits for post-freeze accumulation.
