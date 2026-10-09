# Winnow-12B post-hoc calibration v1 (T168)

Post-hoc calibration layer for the local Winnow-12B Q8 scorer's `irrelevant`
`noul` probabilities on the context-filtering decision. **Advisory only** —
production service config and gateway thresholds are unchanged, and nothing
here is wired into the serving path.

Receipt: `results/winnow_calibration_v1.json`
Raw rows: `results/winnow_calib_fit_v1.jsonl` (train, 2,682),
`results/winnow_calib_eval_v1.jsonl` (eval, 678)
Script: `scripts/winnow_calibration_v1.py` (`score` posts each record's
`request` verbatim to `http://127.0.0.1:8091/v1/systemone`; `fit` fits and
evaluates).

## Data protocol

- Scored **our own CC0 data only**: `data/valen_nano_v1/train.jsonl` (2,682
  records) for fitting, `eval.jsonl` (678) for measurement only — eval rows
  never entered any fit.
- Train split into **cal-fit / cal-holdout** by group: unique `group_id`s
  sorted by sha256 hex (same convention as the dataset's own
  `sha256(group_id) mod 100` train/eval rule); first 80% of groups → cal-fit
  (913 groups / 2,094 rows), rest → cal-holdout (229 groups / 588 rows).
  Group-isolated; both source datasets are represented in both splits
  (file-order splitting would not be — sources are concatenated in blocks).
- Label: `targets.irrelevant.probabilities` argmax; `noul` = P(true) =
  P(candidate certainly irrelevant == drop).

## Fitted parameters (NLL minimization on cal-fit)

| layer | params | fit NLL |
|---|---|---|
| temperature | T = **1.9142** | 0.4574 |
| Platt / logistic | a = **0.7784**, b = **+1.4958** | 0.3649 |

`p_cal = sigmoid(a · logit(p) + b)` for Platt; `sigmoid(logit(p)/T)` for
temperature. b ≈ +1.5 means the map is strongly asymmetric — it lifts the
low/mid range (Winnow under-predicts `true` there: rows at p ∈ [0.4, 0.9] are
~100% actually-irrelevant) while barely touching the extreme keep side.

## Metrics (ECE = 15-bin top-label-confidence ECE, repo convention)

| split | layer | ECE | Brier | NLL | acc@0.5 |
|---|---|---|---|---|---|
| cal-holdout (588) | raw | 0.1213 | 0.1382 | 0.5103 | 0.8367 |
| | temperature | 0.1353 | 0.1377 | 0.4407 | 0.8367 |
| | **Platt** | **0.0775** | **0.1081** | **0.3412** | 0.8554 |
| | isotonic (ref) | 0.0175 | 0.0882 | 0.2868 | 0.8707 |
| eval (678, measurement-only) | raw | 0.1191 | 0.1278 | 0.4867 | 0.8599 |
| | temperature | 0.1409 | 0.1303 | 0.4238 | 0.8599 |
| | **Platt** | **0.0920** | **0.0993** | **0.3162** | 0.8628 |
| | isotonic (ref) | 0.0263 | 0.0798 | 0.2406 | 0.8820 |

## Findings

- **Platt is a real win**: eval ECE 0.119 → 0.092, holdout 0.121 → 0.077,
  Brier and NLL improve ~25–35%, and accuracy@0.5 is *not* degraded (slightly
  improved: +0.3pp eval, +1.9pp holdout — the b>0 shift correctly flips some
  borderline candidates to drop).
- **Temperature scaling alone does not help ECE here.** NLL-optimal T=1.91
  softens all probabilities, which reduces NLL/Brier but *increases* top-label
  ECE (eval 0.119 → 0.141): Winnow's error is not symmetric overconfidence —
  it is underconfident on `true` in the mid-range (p ∈ [0.4,0.9] → ~100%
  positives) and overconfident on `false` at the extreme (p<0.1 → 17%
  positives). No single scalar on the logit fixes both directions.
- **Platt ≉ temperature**: Platt's fitted (a,b) does not collapse to
  (1/T, 0) — the intercept does the real work (fit NLL 0.365 vs 0.457).
  Materially different.
- **Parametric saturates ~0.08 ECE**; a non-parametric isotonic map on the
  same cal-fit data reaches 0.018 (holdout) / 0.026 (eval). The residual is
  functional-shape, not data volume — the empirical calibration map is
  step-like, not logistic.

## Caveats

- **Domain-specific parameters.** Fitted only on our context-filtering
  distribution (`valen_nano_v1`: context_relevance_v1 + oracle_v1 sources).
  These constants must NOT be applied globally to other question types or
  domains; per J-P spec direction, ship them as a per-domain calibration
  bundle alongside the scorer, keyed by (model, question qid, domain).
- **Accuracy threshold caveat**: with b ≠ 0, Platt slightly changes which
  side of 0.5 borderline items land on (here it helped; verify before
  assuming neutral elsewhere).
- Isotonic numbers are a headroom diagnostic, not a shipped layer — it has
  2,094 free knot values and would need its own bundle format + guardrails.
- Advisory layer only; no serving-path changes were made.

## Isotonic bundle v1 (checkpointed artifact, advisory)

`scripts/winnow_isotonic_v1.py` promotes the isotonic diagnostic into a
shippable bundle: `results/winnow_isotonic_bundle_v1.json`
(`schema_version: nanojev-winnow-isotonic-bundle-v1`). Pure-numpy
pool-adjacent-violators (no sklearn/scipy dependency), fitted **on cal-fit
rows only** with the same group-isolated split re-verified against
`train.jsonl`. The bundle stores piecewise breakpoints (`map.x`/`map.y` —
step-function block boundaries), before/after metrics, fit/eval file
sha256s, and a timestamp.

**Domain key**: rows carry no explicit `domain` field, so domain =
`group_id` source prefix (`context_relevance_v1` /
`context_relevance_oracle_v1_seed20260919`). Per-domain maps are fitted per
source (min 100 fit rows; below that, or for unseen domains, apply falls
back to `maps.global`).

| split | layer | ECE | Brier | NLL | acc@0.5 |
|---|---|---|---|---|---|
| cal-holdout (588) | raw | 0.1213 | 0.1382 | 0.5103 | 0.8367 |
| | isotonic global | 0.0176 | 0.0882 | 0.2867 | 0.8707 |
| | isotonic per-domain | 0.0293 | **0.0535** | **0.1936** | **0.9337** |
| eval (678, measurement-only) | raw | 0.1191 | 0.1278 | 0.4867 | 0.8599 |
| | isotonic global | **0.0263** | 0.0798 | 0.2407 | 0.8820 |
| | isotonic per-domain | 0.0268 | **0.0463** | **0.1527** | **0.9440** |

Per-domain ECE within each source (eval):

| domain | n | raw | global map | own-domain map |
|---|---|---|---|---|
| oracle_v1_seed20260919 | 368 | 0.1766 | 0.0909 | **0.0073** |
| context_relevance_v1 | 310 | 0.0615 | 0.1220 | **0.0595** |

Findings:

- **Did per-domain beat global?** On pooled ECE it's a wash (eval 0.0268 vs
  0.0263 — pooled bins mix domains and opposite-signed residual errors
  cancel, masking domain-level gains). On every other axis per-domain wins
  clearly: eval Brier 0.080→0.046, NLL 0.241→0.153, acc@0.5 0.882→0.944,
  and *within* each domain ECE drops to 0.007 / 0.059. Notably the global
  map actively **miscalibrates** `context_relevance_v1` (eval ECE
  0.061→0.122 — worse than raw) while per-domain preserves it (0.059).
  Per-domain is the correct ship shape; pooled ECE parity is a
  cancellation artifact, not a tie.
- Both domain maps were fitted on ~1,000 cal-fit rows each; no domain fell
  below the 100-row minimum, so nothing fell back to global during eval.
- cal-fit ECE ≈ 0 is expected (isotonic interpolates its own fit data);
  holdout/eval are the honest numbers.
- Sanity coverage: `scripts/test_winnow_isotonic_v1.py` (9 tests) — PAVA
  monotonicity, duplicate-x collapse, unseen-domain global fallback, and
  bundle receipt checks (all maps monotone non-decreasing, eval ECE
  improved, eval rows marked measurement-only).

**Downstream consumption (advisory only).** A consumer loads the bundle,
keys a map by `(model="Winnow-12B", qid="irrelevant", domain)` where domain
is the request's source prefix, and computes
`p_cal = np.interp(p_raw, map.x, map.y)` — flat outside the fitted range,
unknown domain → `maps.global`. The bundle is a versioned, checkpointed
artifact: pin `schema_version` + file sha256 at the consumer, treat output
as an advisory calibrated probability (e.g. shadow-logging, threshold
tuning studies, escalation gating in analysis tooling). Nothing is wired
into `serve_decisions.py` or any production routing; activating this layer
in the serving path is a separate gated decision.
