# V5_DATA_DESIGN_V1 — v5 context-filter data design spec

**Status:** design spec (local working doc). No training, collection expansion,
provider calls, or production changes are authorized by this document.
**Date:** 2026-09-30
**Inputs:** `docs/REAL_CONTEXT_EVAL_RESULTS_V1.md`, `docs/KEV_FP_PATTERN_V1.md`,
`docs/ECOSYSTEM_RESEARCH_V1.md` (§2/§4), `docs/REAL_CONTEXT_EVAL_V1.md`,
`docs/REAL_CONTEXT_LABELING_GUIDE_V1.md`, `docs/PRODUCTION_SWITCH_REVIEW_V1.md`,
`data/valen_nano_v4/manifest.json`, `scripts/build_real_context_eval_v1.py`,
`scripts/mine_drop_supp_v1.py`, `scripts/gen_context_relevance_v4_v1.py`,
`docs/L40_MIGRATION_V1.md`. Roadmap: T180/T181/T182 follow-up ("v5 数据设计").
**Privacy:** this doc carries compressed descriptions only — no raw transcript
text. Real-corpus artifacts stay under the gitignored `data/` root; manifests
are content-free.

## 0. What v5 must fix (evidence recap)

Measured on the frozen real eval (590 scored labels = 525 main + 65 mined
drop-supplement; agent-adjudicated under owner delegation):

| failure | scorer | evidence | mechanism |
|---|---|---|---|
| **F-a. short/empty/negative-result over-drop** | lora_v4 | FP incl. noul=**0.919** on a "total 0 / not found" tool result that *was* the answer; correction/research segments over-dropped (0.55–0.92 band) | synthetic corpora taught "no content → irrelevant"; empty/negative results and ack/correction segments co-occurred with drop labels in the generator's weak mode |
| **F-b. drop-side under-recall** | all heads | lora drop-recall **62.5%** (FN=21), kev 48.2%, winnow 23.2% | real cross-task/stale/superseded segments are absent from training distribution; only 2 natural drop labels existed before mining |
| **F-c. boilerplate surface prior** | kev | 30 FPs, all in noul [0.50, 0.69]: exec envelopes / file dumps / search shells / rejected+interrupt control records (21/30) and low-content micro-turns (9/30); transcript-level offsets (top-5 groups = 21/30) | scorer judges "does this text look like standalone substance" instead of anchor-conditioned relevance; envelope≈constant, payload≈variable, yet payload-bearing segments get dropped |
| **F-d. synthetic memorization ceiling** | all | lora_v4 = 1.0000 on v4 synthetic eval; kev public tables show 0.83 train-source → 0.65–0.70 new-source (15–18pp transfer gap) | any fully programmatic family is learnable to the boundary; real/mined data is the only remaining discriminator |

Design stance (from `ECOSYSTEM_RESEARCH_V1.md` §4 + T182 verdict): **EXIT data
recipe on the valen contract** — real/outcome-derived labels, hard+random
negatives — plus a Provence-style asymmetric operating point ("drop only when
certainly irrelevant"). The `state`+`irrelevant`-question shape is already
EXIT-style conditional; no Gemma backbone is imported.

## 1. Invariants (apply to every family)

1. **Record schema** = byte-compatible with `data/valen_nano_v4/eval.jsonl`:
   `request.state` serialized JSON `{conversation, candidate_pointer,
   user_messages_in_order}`; `questions.irrelevant` type `noul` with the
   **verbatim** instructions string (sha256 `46598487…` in the v1 manifest);
   `targets.irrelevant.probabilities {true,false}`.
2. **Asymmetric semantics, unchanged.** `true` = certainly irrelevant = drop.
   Uncertain never becomes a scored keep — see §6.
3. **Group = transcript/scenario.** One `group_id` per source conversation or
   synthetic scenario; a group never crosses splits. Fresh lineage namespaces
   `context_relevance_v5:` and `real_context_v5:`; new seed **20261201**
   (v1/v2/oracle=20260919, v3=20261005, v4=20261115 all reserved).
4. **Label is a pure function of state content** where synthetic (v4 contract);
   for mined-real records the label is adjudicated per the labeling guide and
   `meta.evidence_basis` is mandatory.
5. **Windowing identical to eval v1**: ≤7,000 packed-token budget, keep
   control+candidate+last-user, expand nearest-first, `<elided N earlier
   segments>` markers — ~90% of real candidates are windowed, so training must
   contain the same windowed distribution.
6. **Forbidden inputs**: provider/scorer outputs as labels or state content;
   JevBench/frozen splits; the 590 real labels (§7.2); transcripts already
   admitted to `real_context_eval_v1` (all 60) for any *train* row.
7. **Contrastive pairs are first-class**: every family emits `meta.pair_id`
   grouping minimal-difference opposite-label records so leak/control audits
   can verify the pair actually differs only where intended.

## 2. Family F1 — `anti_shortcut` (negative-result ≠ irrelevant)

**Targets F-a** (lora FP pattern; also kev cluster B, 2 FPs).

| field | spec |
|---|---|
| Source | (a) **mined-real** from unused local transcripts (~290 scanned-but-unadmitted of 351 files; never the 60 eval transcripts); (b) synthetic extension under `context_relevance_v5:` |
| Generation method | **Mined-real primary.** Pattern-miner finds tool_results with empty/negative surface forms (`total 0`, `not found`, `no matches`, `0 hits`, `(Bash completed with no output)`, `"output":""`, empty search stubs) **and** assistant correction/acknowledgment segments, then keeps only candidates where a downstream segment demonstrably depends on them (later text cites/quotes the empty result as evidence, or the correction is referenced). **Synthetic secondary**: tasks of shape "does X exist / prove X is absent" where the empty/negative result is the sole answer; drop members = genuinely task-unrelated empty outputs. |
| Pos/neg ratio | **keep:drop ≈ 2:1** — deliberately keep-heavy; this is a keep-side protection family. Drop members are real short/empty outputs that are truly irrelevant (prevents the reverse shortcut "empty → keep"). |
| De-template variation | Mined rows carry real envelope diversity by construction. Synthetic rows draw payload entities from rotating pools (tool names, file kinds, result phrasings, zh/en mix); keep/drop pair members are length-matched and differ only in payload-task linkage. Envelope strings are harvested into a dedup'd skeleton inventory (shared with F3) rather than hand-written. |
| Failure mode addressed | lora's "no content = no meaning" prior: empty-grep-proves-absence, file-not-found-as-answer, correction/ack segments. Keeps the 0.919 confident-wrong class at zero. |

## 3. Family F2 — `exit_hard_negative` (same-envelope / opposite-label)

**Targets F-b and F-c** (lora FN=21 under-recall; kev envelope prior).

| field | spec |
|---|---|
| Source | **Mined-real primary**: generalize `mine_drop_supp_v1.py` from a hand-written PLAN to an automatic miner over the unused transcript pool — (i) `cross_task_reanchor`: detect task-shift boundaries (user-turn anchored), re-anchor pre-shift segments against the post-shift request exactly as v1 did (56 of 65 supp rows); (ii) `stale_superseded`: `Read(file)→Edit(file)→Read(file)` chains where the earlier read is droppable only when superseded AND unreferenced post-shift (R4 rule); (iii) `same_anchor_tail` foreign-block segments. The **54 verified drop labels** from `drop_supp/eval.jsonl` seed the pattern library — they stay eval-only but define the mining heuristics. **Synthetic secondary**: same-envelope pairs plus **anchor-flip pairs** (identical candidate text; keep under anchor A, drop under anchor B) — the direct training form of anchor-conditioned discrimination. |
| Generation method | mined-real + rule/template; anchor-flip pairs are rule-derived (same state, different `user_messages_in_order` truncation). |
| Pos/neg ratio | EXIT recipe adapted: within each group, ~50% keep-positives, ~25% same-envelope hard negatives, ~25% random negatives (cross-transcript / filler). Overall family keep:drop ≈ 1:1, negatives split hard:random ≈ 1:1 internally (≈2:1:1 keep:hard:random). |
| De-template variation | Real mining is template-free by construction. Synthetic envelopes come only from the harvested real-envelope skeleton inventory (F3); scenario entities/orders randomized per group digest; position, windowing, and elision-marker placement varied. |
| Failure mode addressed | Forces the head to read payload-vs-anchor, not envelope-vs-nothing; supplies the real-distribution drop positives the v1 set lacked (2→54 natural+mined); directly trains "stale read superseded by re-read" and "pre-shift segment after explicit task switch". |

## 4. Family F3 — `boilerplate_prior_breaker` (envelope contrastive pairs)

**Targets F-c** (all 30 kev FPs, clusters A–G).

| field | spec |
|---|---|
| Source | Envelope skeletons harvested from real transcripts of **both** harness dialects: codex `Script completed`/`Chunk ID…exit 0` exec envelopes, line-numbered file dumps, URL-probe/git-status lists, empty polls; claude `Web search results for query:` shells + REMINDER tails, domain-verify blocks, tool-use-rejected / `[Request interrupted]` control records; plus low-content conversational forms (short openers/directives, micro-acks/scope declarations, completion reports). The fc3e cluster showed Claude-harness control-surface text is absent from the synthetic domain — both dialects are mandatory. |
| Generation method | **Hybrid**: harvested envelope skeletons (mined) × payloads — payloads are either (a) harvested real snippets re-contextualized, or (b) model-written paraphrases (local model, inference only — §8). Rule assembly: contrastive pair = identical envelope, payload either task-evidence (keep) or generic/cross-task noise (drop). |
| Pos/neg ratio | **keep:drop = 1:1** within every pair batch. |
| De-template variation | Pair members matched on length ±10%; envelope×payload×anchor-role factorial; the *same* payload text appears once keep (task-relevant anchor) and once drop (foreign anchor) across different groups — kills any payload-string→label shortcut; skeleton inventory dedup'd by normalized template hash. |
| Failure mode addressed | "Envelope looks generic → drop" prior. The only way to separate the pair is payload-vs-anchor semantics; low-content user/assistant turns get both labels so "short → drop" can't be learned either. |

## 5. Family F4 — `outcome_positive` (downstream-used = hard keep)

**Targets keep-side calibration + F-a reinforcement; prevents over-correction**
(after F1/F3 teach "weird-looking things are keep", the head needs strong
positive evidence for *why*, not just pattern reversal).

| field | spec |
|---|---|
| Source | (a) **Public resolved-task trajectories** with outcome markers — OpenHands-family sets (`SWE-Gym`/`SWE-rebench`/`SWE-Zero` trajectories, `resolved`+gold patch) and `SALT-NLP/SWE-chat` real CLI sessions; license screen per row before use (provenance + license string in `meta`, NC rows flagged and excluded from any future public release — same discipline as `open_corpus_v1`). (b) local unused transcripts. |
| Generation method | **Mined-real with deterministic linker, zero labeler cost for the positive claim**: a `Read`/`Grep` result is keep iff a later `Edit`/gold-patch hunk touches the same file path; a command/tool result is keep iff its output text is quoted/incorporated in a later assistant message or the final answer; a user constraint turn is keep iff a later assistant action conforms to it. Every keep row gets `meta.evidence_basis` = the downstream pointer — auditable without re-reading text (labeling guide R8). Sampled rows (≥10%) still go through adjudication to estimate linker precision. |
| Pos/neg ratio | Keep-only by construction (all rows `false`). Balance is restored at the merged-set level (§7.1): F4 keeps are offset by F2/F3 drops. Optionally emit a sibling `outcome_negative`: segments in the *same* resolved trajectory that the outcome never touched (unused reads, abandoned-branch outputs) → drop labels via the same linker, complementing F2's random negatives. |
| De-template variation | Real trajectories; MinHash near-dup pass vs train corpora and intra-family; stratified across repo language/task type so "path mention" doesn't degenerate into one surface form. |
| Failure mode addressed | Provides *ground-truth* keep evidence ("the agent actually used this downstream") that no generator bias can fake — the EXIT/HotpotQA supporting-facts analogue for agent transcripts; anchors calibration so keep-side saturation (lora p50 noul ≈ 0.000) stays earned rather than shortcut. |

## 6. Label semantics (unchanged, restated for v5)

- **Asymmetric certain-only-drop** is kept verbatim: `true` = certainly
  irrelevant; the cost asymmetry (drop required evidence ≫ keep noise) is the
  product spec, not a labeling convenience.
- **Uncertain band**: `uncertain` rows are **excluded** from scored eval and
  from SFT/RLCD targets; they may be retained in a separate
  `uncertain_band` diagnostic slice (abstention/calibration analysis only,
  `training_allowed=false`). Rationale: a middle-band target would teach the
  head to emit mid-range noul on genuinely-ambiguous real segments, which is
  fine for calibration diagnostics but must not shift the keep decision.
- **Label authority**: model/scorer proposals are never labels; mined
  candidates go through the same agent-adjudication-under-owner-delegation pass
  (`labeler_type=human`, evidence_basis per row, conservative uncertain→keep);
  provider outputs never appear in labels, state content, or training.
- **Pair integrity**: for contrastive pairs, an adjudicator spot-check (≥5% of
  pairs) verifies the pair member labels are self-consistent.

## 7. Sizes, splits, budget

### 7.1 Train set (`data/valen_nano_v5` = frozen v4 train + v5 families)

| slice | records (target) | method share |
|---|---:|---|
| v4 train retained verbatim | 17,124 | rule/template (v1–v4 families) |
| F1 anti_shortcut | ~1,500 (≈1,000 mined + ~500 synth) | mined:synth 2:1 |
| F2 exit_hard_negative | ~3,000 (≈2,000 mined + ~1,000 synth incl. anchor-flips) | mined:synth 2:1 |
| F3 boilerplate_prior_breaker | ~1,500 (pairs ⇒ ~750 unique payloads) | hybrid |
| F4 outcome_positive | ~2,000 (≈1,500 public-trajectory + ~500 local) | mined |
| **total train** | **~25,000** | ~36% mined/real-derived |

Merged label balance target: keep:drop ≈ 55:45 (v4 train is 52:48; F1/F4 push
keep up, F2/F3 pull back). `hard_negative` flag retained on F2/F3 pair members
and F1 synthesized pairs.

### 7.2 Eval composition — four disjoint tiers

| tier | content | role | size |
|---|---|---|---|
| **real-frozen** | existing 590 labels (525 main + 65 drop-supp) | **fixed holdout, never trained on** — gitignored `data/`, `training_allowed=false` in both manifests; primary acceptance instrument | 590 |
| **real-v5-ext** | new mined candidates from transcripts **not** among the 60 eval transcripts and disjoint from v5-train transcripts; same labeling pass | extends drop-positive coverage + fresh-source check on mining itself | ~300–400 candidates → labeled |
| **synthetic eval** | `context_relevance_v5` eval/dev pools (gidx disjoint from train range, same mod-rule scheme as v4) | family-wise discrimination + saturation tripwire | ~1,500 eval + ~200 dev |
| **transfer holdout** | an **entire source family withheld from train**: e.g. train on codex_sessions+SWE-Gym, hold out claude_projects+SWE-chat (or vice versa for a second arm) | measures train→new-source gap like kev's frozen manifests (decision-v7/transfer-v4/v9); reported as seen-source acc − new-source acc | ~500–800 |

### 7.3 Split protocol (normative)

1. `group_id` never straddles train/eval/dev/transfer; assert pairwise
   disjointness in code (v4 precedent).
2. Dedup vs **all** prior corpora — `valen_nano_v2/v3/v4` (all splits),
   `real_context_eval_v1` (main + drop_supp + excluded), JevBench bundle —
   via `request_sha256` + normalized-state hash (existing helpers) **plus** a
   MinHash/LSH near-dup pass on normalized candidate text (`text-dedup`,
   Apache-2.0) as a second layer; intra-set dedup same way.
3. Transfer holdout is selected **before** family emitters run; its source
   transcripts are excluded from every mining/generation input list.
4. Manifest: content-free for real-derived splits (hashes + counts only),
   per-family/per-source/per-method counts, license strings for public rows,
   `training_allowed` per split, `excluded` denylist including
   `data/real_context_eval_v1/**`, `data/jevbench_offline_bundle_v1`,
   `*_test.jsonl`, `*_ood.jsonl`, and results/shadow receipts.
5. Known-tooling caveat (handoff): `label_real_context_v1.py --finalize`
   ignores `--out-dir` and would overwrite the main `eval.jsonl` — always pass
   `--eval-out` explicitly for v5-ext.

## 8. Build plan — what runs where

| piece | compute | location | notes |
|---|---|---|---|
| `mine_real_v5.py` (generalize `mine_drop_supp_v1.py`: auto re-anchor detection, stale-read linker, envelope harvester) | CPU | local | reuses `build_real_context_eval_v1.py` extraction/windowing/redaction verbatim; same credential-scan-drops-transcript rule |
| `gen_context_relevance_v5_v1.py` (F1/F2/F3 synthetic emitters + anchor-flip machinery) | CPU | local | clone of v4 generator contract (constant-label kinds where possible, validator run on own output, 0 violations required) |
| public-trajectory linker (F4) + license screen | CPU | local | download/public data only; no provider calls |
| dedup pipeline (hash + MinHash) | CPU | local | `ChenghaoMou/text-dedup` |
| **model-written payload paraphrases** (F3 option; small volume, ≤~2k generations) | MPS inference | local OK | generation is inference, not training — allowed locally; precedent: `run_real_context_holdout_localgen_v1.py` local-generation backend (Qwen3.5-4B on MPS). Synthetic-only; never rewrite real transcript text. |
| larger teacher generation (optional quality bump for F3 payloads, e.g. 27B-class) | GPU inference | **L40×2 server** if used | flag: still inference; needs `HF_ENDPOINT=https://hf-mirror.com`; only worth it if local-MPS paraphrases show measurable template leakage (pair-dedup audit fails) |
| labeling pass on mined candidates (~2.5–3k train + ~350 eval-ext) | human/agent adjudication | local | same guide + delegation precedent; evidence_basis mandatory |
| **v5 SFT/LoRA/RLCD training** | GPU | **L40×2 required** | v4-scale run (~17k→25k rows); configs bumped to `nano_*_v5` output dirs per `L40_MIGRATION_V1.md` §5 note — never overwrite v4 artifacts |
| eval scoring sweeps | MPS fp32 | local | same-device rule; `eval_systemone_backend_v1.py` |

**Explicitly not needed locally / not allowed**: provider API calls on real
transcript content (privacy boundary); official Jev outputs anywhere in the
data path; JevBench rows in train.

## 9. Acceptance protocol (v5 head vs lora_v4)

All gates must hold; same-device fp32 MPS reference; paired bootstrap CIs on
per-item deltas; thresholds reported on the fixed diagnostic grid only — this
eval never selects a production τ.

| gate | criterion |
|---|---|
| **G1 real-frozen superiority** | on the frozen 590: v5 acc > lora_v4's 0.9453 AND drop-recall > 62.5% AND FP < 11; and **FP=0 at τ≥0.9** with **zero confident-wrong drops of required evidence** (no noul≥0.9 FP — the 0.919 class must be eliminated, not just reduced) |
| **G2 consensus preserved** | `lora_v5 ∧ winnow` AND-drop keeps **FP=0 / precision 1.000** on the double-covered intersect, with recall ≥ the current 17.9% — the deployed `?backend=consensus` safety property must not regress |
| **G3 transfer gap** | seen-source eval acc − transfer-holdout acc ≤ **10pp** (kev's published 15–18pp gap is the anti-target); reported per source family |
| **G4 synthetic non-regression + no saturation** | `valen_nano_v4` eval acc ≥ 0.99 (lora_v4 = 1.0000); on the new `context_relevance_v5` eval, ≥0.99 with zero borderline predictions is a *memorization tripwire* → extend the set rather than claim victory |
| **G5 asymmetric posture** | error structure stays safe-sided: FN (missed drops) may exceed FP; any FP on F1/F4-type keep rows is a protected-family regression (V4-S0 rule) |
| **G6 calibration** | keep-side noul tail monitored — the 0.55–0.92 FP tail band must not grow; report ECE (15-bin) + recall-at-low-p like winnow's public band |

## 10. Non-goals / risks

- This spec does not authorize training, collection-scope expansion, or any
  production/canary step; T175's no-switch decision stands (consensus backend
  remains shadow/advisory).
- Public-trajectory licenses are screened per row before inclusion; NC content
  never leaves local artifacts.
- Mined labels inherit the v1 provenance caveat (agent-adjudicated under owner
  delegation); owner spot-check of the ≥25% double-labeled subset remains the
  escalation path (κ ≥ 0.75 target per the labeling guide).
- If the unused-transcript pool yields < ~1k mined rows/family, downscale the
  mined share and say so in the manifest — do not backfill with extra
  templates (that recreates the v3/v4 memorization problem).
