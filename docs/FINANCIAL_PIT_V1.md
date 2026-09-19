# Financial point-in-time contract V1

Status: **executable timestamp and split-integrity foundation**, not market-data validation, a trained financial model, an execution simulator, or a trading strategy.

[financial_pit_v1.py](../scripts/financial_pit_v1.py) validates declared data timing and builds chronological train/dev/calibration/test folds. The first contract handles realized binary events for cross-entropy/Brier baselines. Multi-action policies, sizing, venues, fill simulations, risk state, and RLCD training remain subsequent roadmap work.

## Record contract

Records contain exactly `schema_version`, `id`, `asset_id`, `venue`, `decision_ns`, `universe_available_ns`, `features`, and `label`. The schema is `nanojev-financial-pit-v1`.

All timestamps are nonnegative integer UTC Unix nanoseconds, not local dates, floating-point seconds, or booleans. Asset IDs must be stable source identifiers rather than assumed permanent ticker symbols.

Each named feature contains exactly:

- `value`: finite numeric or boolean value; missing-value treatment must be declared and applied upstream
- `event_ns`: timestamp of the observed source event, not the future event being forecast
- `available_ns`: earliest time the feature was genuinely usable by the decision process
- `fit_cutoff_ns`: latest data timestamp used to fit its transform; zero denotes a transform requiring no fitted observations
- `source_id`: source lineage identifier
- `version`: transformation/snapshot version

Required ordering is `event_ns <= available_ns <= decision_ns` and `fit_cutoff_ns <= available_ns`. Universe-membership information must also be available by the decision time. A known future calendar date can be a feature value, but its source observation and publication times cannot be in the future.

Each label contains `event`, `definition_sha256`, `end_ns`, `available_ns`, and Boolean `outcome`. The definition hash identifies a separately frozen event predicate, horizon, and price/observation convention. Labels must satisfy `decision_ns < end_ns <= available_ns`. A probability argmax is not accepted as an observed event label.

One V1 cohort has a fixed feature-name set and one event-definition hash. Duplicate record IDs or duplicate `(asset_id, venue, decision_ns, event)` samples are rejected. Input to the predictor is an allowlisted projection of asset, venue, decision timestamp, and feature values; labels, record IDs, split tags and audit metadata are excluded. Predictive models and encoders must not be given the raw training row.

## Purged walk-forward protocol

A protocol JSON file has `folds`, `embargo_ns`, and `asof_ns`. Every fold contains four nonempty chronological, disjoint half-open windows, each `[start_ns, end_ns)`:

1. `train`: fit parameters using only labels available by this window's end.
2. `dev`: choose the predeclared candidate using labels available by this window's end.
3. `calibration`: fit probability calibration and thresholds using only this period.
4. `test`: final evaluation; labels must be mature by the fixed evaluation as-of.

For each non-test row, `label.end_ns + embargo_ns` must be strictly earlier than the next phase's start. Otherwise it is excluded as `purged_overlap_or_embargo`. A label not yet available at the phase's fit cutoff is also excluded. This is a one-sided embargo for strictly chronological expanding-window evaluation; it is not bidirectional shuffled cross-validation.

The chronological decision cutoff does not mean future outcomes were known at decision time: features obey decision-time availability, while labels become usable later at their own fit/evaluation cutoff. Test outcomes never enter the current fold's predictor input or selection process. Later expanding windows may reuse matured historical observations only under an already frozen walk-forward protocol, not as a way to retune a strategy after looking at its test performance.

Test windows across folds must be chronological and nonoverlapping. Fold audits record every retained ID, excluded ID/reason, phase counts, declared windows, and dataset hash. Empty phases are explicitly reported; the CLI exits unsuccessfully after saving such an audit so it cannot be mistaken for a training-ready fold.

## Measured validation

The [synthetic stress receipt](../results/financial_pit_v1_stress.json) records 12,000 records across three synthetic assets and three expanding folds. Each fold retains 1,200 test events and removes 63 train/dev/calibration boundary events through the two-minute embargo around five-minute labels.

The harness independently verifies retained label intervals and fit-time availability. It rejects 1,000 injected future-feature/fit/universe/label-timestamp violations and confirms 1,000 label mutations cannot alter the predictor input. Unit tests additionally cover duplicate snapshots, schema changes, delayed labels, unmatured test labels, boundary equality, empty splits and overlapping test windows.

These observations establish integrity behavior on constructed cases only. They do not establish forecast accuracy, calibrated probability quality, trading returns, or millisecond inference. The synthetic outcome rule is deliberately explicit and is not a financial signal.

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_financial_pit_v1.py'
.venv/bin/python scripts/benchmark_financial_pit_v1.py --output results/financial_pit_v1_stress.json
.venv/bin/python scripts/financial_pit_v1.py --input market_rows.jsonl --protocol frozen_protocol.json --output fold_audit.json
```

The last command requires real caller-supplied data and a frozen protocol; this milestone does not download either.

## What this cannot prove

The validator can detect inconsistent timestamps, not dishonest or incorrectly reconstructed timestamps. Source licensing, availability history, revision/corporate-action treatment, universe completeness including delistings, event-definition correctness, fitted-transform provenance, exchange calendars, and feed delays need separate raw-source audits. Merely setting an availability timestamp or supplying a SHA-256 string does not prove point-in-time authenticity.

The next financial deliverables are a versioned source/rights/universe manifest, instrument/regime holdouts, independently verified feature transformations, realistic spread/fees/slippage/latency/capacity simulation, simple deterministic and proper-loss baselines, and then matched RLCD-like experiments. No order submission, broker credentials, live capital, or claimed profitability is introduced here.
