# Real-context eval V1 — next evaluation tier for the context-filter decision head

**Status:** design proposal only. No implementation, no collection, no provider
calls, no active filtering are authorized or performed by this document.

**Date:** 2026-09-28

## 1. Motivation

`data/valen_nano_v3` (12,122 train / 2,950 eval, 38.5% hard negatives, eval
SHA-256 `c95286ff…`) is saturated as a discriminator. W135 (2026-09-28,
`CURRENT_PROGRESS_AND_HANDOFF.md`, roadmap T166/T167):

| checkpoint | recipe | v3 eval acc (fp32) | brier | nll |
|---|---|---:|---:|---:|
| nano_sft_v3 | head-only SFT 6ep | 0.9156 | 0.129 | 0.228 |
| nano_rlcd_v3 | SFT→RLCD 300 steps | 0.9858 | 0.019 | 0.033 |
| nano_sft_text_v3 | init sft_v2, LoRA 21.6M 3ep | **1.0000** | ~9e-11 | ~1e-7 |

The strongest head answers all 2,950 eval items correctly with extreme
confidence and zero borderline predictions. Root cause (recorded in
`VALEN_NANO_V3_EXPANSION_V1.md` §Known limitations): the source is finite,
self-authored, programmatic — ~1,550 distinct normalized states for 11,712 v3
records — so eval is largely a generalization-over-paraphrase test of a rule
family the unfrozen backbone can now memorize.

We therefore need a next tier that (a) discriminates future model variants
again, and (b) gives a first defensible estimate of production performance on
real agent transcripts. This document proposes **Real-Context Eval V1**: a
scored keep/drop eval set built from owner-selected real local transcripts plus
a harder synthetic cohort, in the same `eval.jsonl` contract as `valen_nano_v3`,
under the same privacy and publication discipline as the existing real-context
holdout machinery.

## 2. Relationship to existing artifacts

| Artifact | Role | Reuse here |
|---|---|---|
| `docs/REAL_CONTEXT_HOLDOUT_PROTOCOL_V1.md` | Intake protocol for a 30-case safety holdout | This proposal reuses its data boundary, forbidden sources, labeling contract, and isolation rules. The holdout is gate-level safety plumbing; this eval is a scored model-comparison set. Different purpose, same discipline. |
| `docs/REAL_CONTEXT_COLLECTION_RUNBOOK_V1.md` | Owner authorization template (`research/real_context_collection_authorization_v1.json`) | Collection under this proposal requires a new scoped authorization artifact with `collection.scope` naming this eval; the existing template's checklist applies verbatim. |
| `scripts/real_context_holdout_manifest_v1.py` | Content-free manifest builder/validator (hash-only, credential scan, duplicate `request_sha256` rejection) | Reused or mirrored for the eval manifest; the manifest must stay content-free. |
| `data/valen_nano_v3/` + `external/valen/scripts/build_nanojev_valen_v3.py` | Record schema, `group_id` split rule, verbatim `noul` question | The eval records are byte-compatible with this schema (§5). |
| `docs/WINNOW_CALIBRATION_V1.md` | Metric conventions (15-bin top-label ECE, NLL, Brier, acc) | Scoring protocol adopts the same conventions (§7). |
| `docs/CONTEXT_FILTER_THRESHOLD_POLICY_V1.md` | Threshold-as-operating-point discipline | All thresholds reported here remain diagnostic; no production threshold is selected. |
| `~/.local/state/nanojev-eval/log.jsonl`, `~/.codex/nanojev/usage.jsonl`, `data/context_shadow_v1_*` | Existing logs | **Important negative finding:** all existing logs are content-free by design (event ids, hashes, latencies, probabilities — no request bodies). They cannot be mined for eval rows directly; they only prove the sampling frame. Real text must come from owner-selected transcript archives (§3.1). |

This proposal does not modify any frozen artifact, the production scorer
(`nano_rlcd_v2` at `:8093`), or the shadow gate.

## 3. Data sources

Three source classes, mixed into one eval set with per-record `meta.source_dataset`
provenance.

### 3.1 Real local transcripts (primary, target ≥ 60% of items)

Because the repo deliberately never logs raw conversation bodies, the source is
the owner's local agent transcript archives, e.g.:

- `~/.claude/projects/**/*.jsonl` — Claude Code session transcripts (~50 MB
  locally at design time). These contain exactly the target distribution: long
  multi-turn conversations with tool calls, tool results, file contents,
  corrections, and topic drift.
- `~/.codex/sessions/` — analogous session logs where present.
- Owner-selected local service request captures (`local_test_input.json`-style
  bodies) and sanitized application/tool traces, per the holdout protocol's
  allowed-sources list.

Sampling rules:

- transcripts are **owner-selected** after the collection authorization names
  the scope (§3.4 boundary);
- whole transcripts are the isolation unit: one transcript → one `group_id`,
  so a transcript never straddles any future train/eval boundary;
- per transcript, enumerate candidate pointers over a declared segment policy
  (assistant text, tool results, older user turns — matching the shadow gate's
  eligibility model); cap candidates per transcript (proposed ≤ 25) to bound
  labeling cost and group size;
- stratify sampling across wire formats (`openai_chat`, `anthropic_messages`),
  conversation length bands, and the family taxonomy in §6.

Redaction pipeline (mirrors `real_context_holdout_manifest_v1.py` checks):

1. deterministic credential/secret scan (API keys, tokens, emails, paths with
   secrets); failure ⇒ drop the transcript, not just the segment;
2. owner manual pass for personal identifiers and unredactable content;
3. raw text stays only under a gitignored data root (proposed
   `data/real_context_eval_v1/`); the manifest carries SHA-256 only;
4. near-duplicate and exact-hash check against `valen_nano_v2/v3` and intra-set
   (`request_sha256` overlap ⇒ reject), per holdout protocol §Isolation.

### 3.2 Synthetic-harder families (secondary, target ~25%)

A new generator namespace (proposed `context_relevance_v4`, disjoint seed and
fresh `group_id` lineage, never reusing v3 group ids) aimed at the failure
modes the v3 template space cannot express:

- **realistic tool segments**: multi-call tool histories with
  resolved/unresolved pairing, mutable and non-repeatable results (per
  `TOOL_HISTORY_SHADOW_V1.md` semantics), where single obsolete results are
  droppable but the pair structure constrains the label;
- **cross-pointer dependencies**: a candidate whose label flips depending on a
  remote pointer (e.g. an early assistant claim that is sole evidence for a
  late user question);
- **long-context dilution**: 30–120 segment conversations where a single
  required fact sits among near-duplicate distractors;
- **topic-shift boundaries**: mid-conversation task changes where pre-shift
  segments are genuinely irrelevant except declared corrections;
- **adversarial restatements**: restatements that differ from the operative
  value by one token (extend `near_duplicate_evidence` / `correction_confirmed`
  with subtler deltas);
- **OOD-language drift**: zh↔en flips *within* a segment, not only across
  records.

All synthetic records keep `meta.hard_negative` semantics and the v2 label
contract (label is a pure function of state content; `correction`/
`overlap_distractor`-style context-dependent kinds emit label and variant from
the same branch).

### 3.3 Public datasets (optional, target ≤ 15%, eval-only)

Precedent: `data/open_corpus_v1` (UltraChat MIT, ToolACE CC-BY-NC-4.0, local-only).
Candidate public sources must pass a license screen *before* any use:

| Requirement | Rule |
|---|---|
| Redistribution in a public manifest | only if the license permits it; otherwise the manifest stores source id + row hash, never content |
| NC-licensed rows (e.g. ToolACE) | local eval-only; excluded from anything published |
| Labels | public sets ship no keep/drop labels → they go through the same human labeling path as §3.1; provider or scorer outputs are never labels |
| Provenance | `meta.source_dataset`, `meta.source_split`, license string, snapshot date recorded per row |

### 3.4 Forbidden sources (inherited)

- JevBench or any benchmark rows; `*_test.jsonl` / `*_ood.jsonl` frozen splits;
- provider outputs (official Jev or other) as labels or as state content —
  reaffirmed: Jev outputs never enter training, calibration, **or** eval labels;
- scorer outputs (Winnow, valen head) as labels;
- any source the owner has not explicitly selected within the authorized scope.

## 4. Labeling strategy

Labels are manual and conservative, per the holdout contract, scaled to
hundreds of items:

1. **Labeler**: the owner, using a written labeling guideline that restates the
   `noul` question verbatim (`true` == candidate certainly irrelevant ==
   drop). Uncertain ⇒ `false` (keep). The judge reads the full conversation;
   no truncation.
2. **Independent second labeler**: a non-author reviewer (or the owner after a
   ≥ 7-day blind re-label with shuffled order) labels a random ≥ 25% subset.
3. **Agreement measurement**: Cohen's κ on the double-labeled subset, reported
   in the manifest; target κ ≥ 0.75. Disagreements are adjudicated by
   discussion; unresolved disagreements keep `false` and are flagged
   `meta.label_confidence: "low"`.
4. **Low-confidence handling**: low-confidence rows are excluded from the
   scored set (mirroring holdout rule "low-confidence labels cannot be used for
   scored cases") but retained in a separate `review` slice for abstention-band
   diagnostics (§7).
5. **Triage aid, not labels**: local scorers *may* be used after labeling to
   stratify *future* sampling (e.g. over-sample items where a head disagrees
   with the label). Any scorer-influenced selection must be recorded in `meta`
   (`selection_method`) because it biases the sample; the v1 baseline uses
   deterministic stratification only.
6. **Contract labels for scored rows**: each scored real-transcript row declares
   the downstream evidence basis in `meta` (which pointer(s) make the candidate
   required, or why it is certainly irrelevant), so label disputes are
   auditable without re-reading raw text.

## 5. Schema compatibility

Records are byte-schema-compatible with `data/valen_nano_v3/eval.jsonl`:

```json
{
  "group_id": "<source_lineage>:<source_group_id>",
  "request": {
    "state": "<JSON string: {\"conversation\": [{\"pointer\",\"role\",\"content\"}…], \"candidate_pointer\": \"…\", \"user_messages_in_order\": […]}>",
    "questions": {"irrelevant": {"type": "noul", "instructions": "<verbatim, unchanged>"}}
  },
  "targets": {"irrelevant": {"probabilities": {"true": 0.0, "false": 1.0}}},
  "meta": {
    "record_id": "…", "domain": "…", "modality": "text", "language_bucket": "…",
    "source_dataset": "real_transcript|context_relevance_v4|<public_id>",
    "source_split": "…", "candidate_kind": "…", "candidate_pointer": "…",
    "hard_negative": false, "label_confidence": "high",
    "labeler": "owner|second|adjudicated", "selection_method": "deterministic"
  }
}
```

Hard requirements:

- `questions.irrelevant.instructions` preserved verbatim
  (`instructions_preserved_verbatim: true` in the manifest);
- `request.state` remains the serialized JSON string form the valen compiler
  and sidecar scorer already accept;
- one record per candidate pointer; `group_id` = transcript/scenario id, so
  all records of one conversation share one group;
- `meta.source_dataset` uses a new lineage prefix
  (`real_context_eval_v1:` / `context_relevance_v4:`) — fresh namespace,
  strict isolation from v2/v3 group ids;
- label-consistency invariant holds set-wide: one label per normalized state
  across the whole set (extend `validate_valen_label_consistency_v1.py`).

Manifest (mirroring `valen_nano_v3/manifest.json`): `schema_version`,
`builder`, per-file `sha256` + `bytes`, record/group/label/hard-negative
counts, per-source counts, split rule, excluded-sources list. **The eval set is
evaluation-only**: `training_allowed=false`, recorded in the manifest; there is
no `train.jsonl`. A small `dev.jsonl` (proposed ≤ 20% of groups, same
`sha256(group_id) mod 100` rule) may exist for debugging/report iteration; the
frozen `eval.jsonl` is the reported split and must not be used for threshold,
calibration, or model selection.

## 6. Size, composition, and CI targets

Manual labeling is the cost driver. Proposed v1 targets:

| Quantity | Minimum | Target |
|---|---:|---:|
| scored eval rows | 400 | 600 |
| distinct transcripts/groups | 40 | 60 |
| families (§3 taxonomy) | 8 | 12 |
| keep (`false`) share | ≥ 40% | ~55% |
| hard-negative / ambiguous share | ≥ 25% | ~35% |
| double-labeled subset | ≥ 25% | 100 rows |
| tool-linked rows | ≥ 15% | — |
| multilingual rows | ≥ 10% | — |
| dev rows (debug-only) | ≤ 20% of groups | — |

Statistical framing: at n=400 eval rows, a 95% Wilson CI on accuracy is
±~4.9pp at p=0.9 — enough to separate variants like sft_v3 (0.9156) from
rlcd_v3 (0.9858) but **not** to certify < 1pp differences; discrimination
between near-equal heads is reported via paired bootstrap CIs on the per-item
difference (McNemar-style), same-cohort pairing mandatory. If v1 labeling lands
below 400 rows, the doc must state the wider CI rather than tighten claims.

## 7. Scoring protocol

- **Precision/device:** fp32 on MPS is the reference configuration (deployment
  route validated W135: fp32 A100↔MPS flips 1/678, max Δp 0.051; bf16
  cross-device drift was 50/678). Scores are comparable only when measured on
  the **same device and precision** — the same-device parity rule from
  `VALEN_NANO_V2_REEVAL_V1.md` applies: cross-device numbers are never compared
  directly; a candidate's receipt must record checkpoint dir + config SHA-256,
  device, dtype, and eval-file SHA-256.
- **Metrics (repo conventions):** accuracy, NLL, Brier, ECE (15-bin
  top-label-confidence ECE as in `winnow_calibration_v1.py`), reported overall
  and **per family/domain** — the average must not hide a protected-family
  regression (V4-S0 contract §3 rule).
- **Abstention-band analysis:** recompute coverage/accuracy at the fixed grid
  {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} over the same raw probabilities (the
  `SKILL_ABSTENTION_DIAGNOSIS_V1.md` protocol — thresholds are post-hoc
  re-readings, none presented as selected). Report confident-wrong counts at
  ≥ 0.9 (CW@0.9), the keep-side analogue (confident drops of required
  evidence = safety-relevant errors, reported separately and capped at zero for
  any "safe operating region" claim), and point-biserial correlation.
- **Selective-risk view:** coverage-vs-error curve + per-band accuracy table;
  abstention on `label_confidence=low` rows is reported but excluded from
  scored metrics.
- **Receipts:** content-free result JSON under `results/` (gitignored);
  per-item probabilities stay local-only; the public summary carries
  aggregates + data/manifest hashes only.

## 8. Publication boundaries

Public (docs/results eligible for publication):

- aggregate metrics, per-family aggregates, abstention-band tables, CIs;
- content-free manifest (hashes, counts, family/family-license metadata);
- the synthetic `context_relevance_v4` portion may be published CC0 if the
  owner selects it, under the same generator/builder disclosure as v3;
- dataset SHA-256s and reproduction commands.

Local-only (never published):

- raw transcripts, request bodies, candidate text, tool outputs — including
  inside public manifests or receipts;
- per-item model probabilities and receipts on the real subset;
- NC-licensed public rows and anything derived from their content;
- provider outputs of any kind (also barred from labels, training, and
  calibration);
- labeling notes containing transcript text.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Privacy leakage from real transcripts | owner-only source selection; deterministic credential scan + manual redaction; raw text confined to gitignored root; manifest content-free; stop conditions inherited from the holdout protocol |
| Single-owner distribution ≠ production | declare scope honestly: this estimates performance on *this owner's* agent traffic; report per-family metrics; do not claim general production numbers |
| Label noise / labeler bias | double-label ≥ 25%, κ report, adjudication, conservative uncertain→keep, low-confidence exclusion |
| Selection bias from scorer-guided sampling | v1 uses deterministic stratification only; any scorer-influenced sampling is flagged in `meta` |
| Train/eval leakage | `request_sha256` + near-duplicate screening vs v2/v3 and intra-set; transcript-level group isolation; fresh lineage namespace |
| License contamination | license screen before inclusion; NC rows local-only and flagged; snapshot dates + provenance recorded |
| Saturating again | harder synthetic families + real transcript tail; monitor: if any variant hits ≥ 0.99 acc with zero borderline predictions, treat as saturation signal and extend the set rather than claim victory |
| Small-n overclaiming | paired bootstrap CIs mandatory; report per-family worst case; no operating-point or production claims from this eval |
| Eval data leaking into training | manifest records `training_allowed=false`; eval dirs are excluded in builders by explicit denylist (as `valen_nano_v3/manifest.json` `excluded` does for JevBench/receipts) |

## 10. Non-goals and open questions

Non-goals: this eval does not authorize or measure active filtering, does not
select a production threshold, does not promote `nano_sft_text_v3` or any head
to production (synthetic 1.0000 ≠ production performance), and does not replace
the real-context holdout's gate-safety role.

Open questions for owner/reviewer before build:

1. Authorize collection scope: which transcript roots (`~/.claude/projects`,
   `~/.codex/sessions`, service captures) and what expiry?
2. Second labeler: owner re-label vs independent reviewer — staffing decision.
3. Whether a `dev.jsonl` slice is wanted at all (adds labeling cost, enables
   debug iteration without touching frozen eval).
4. Public-release appetite for the synthetic `context_relevance_v4` subset.

## 11. Reproduction path (after authorization — not implemented)

```bash
# 1. scoped collection authorization (new artifact, owner fills fields)
#    per docs/REAL_CONTEXT_COLLECTION_RUNBOOK_V1.md
# 2. builder (future script) → data/real_context_eval_v1/{eval,dev,review}.jsonl + manifest.json
# 3. validators: content-free manifest check, credential scan re-run,
#    valen label-consistency check extended to this root, isolation audit
# 4. scoring (same-device fp32 MPS, per-candidate):
#    external/valen eval harness or valen-head sidecar over eval.jsonl
# 5. aggregate receipt → results/real_context_eval_v1_<candidate>.json
```
