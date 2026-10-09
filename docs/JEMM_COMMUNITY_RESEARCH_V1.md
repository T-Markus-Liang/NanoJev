# JEMM / Jev-Omni community research V1

Status: desk research complete; no code or training change implied.
Research date: 2026-09-26. All numbers below are self-reported by model cards,
community pages, or preprints — cite as snapshot, not ground truth.

## 0. Naming caveat — resolved

**`MaestroYan/JEMM` exists** (https://huggingface.co/MaestroYan/JEMM, Apache-2.0,
published ~2026-09-27, trending #1 on the `search=JEMM` result page). The earlier
desk pass missed it because the HF API search did not surface a same-day upload.
JEMM = **Judgment Engine for MultiModal decisions**. The Gemma-based
`akhilaaa3/Jev-Omni` covered below remains relevant as a second architecture point.

## 0.5. JEMM (MaestroYan) — the actual artifact

- **Form**: LoRA adapter on `Qwen/Qwen3.8-27B` (a natively multimodal VLM) —
  weights are the adapter only; base does the multimodal lifting. Needs ~64 GB
  CUDA (27B bf16). Apache-2.0. GitHub server: https://github.com/ypcypc/JEMM —
  serves the **same `POST /v1/systemone` wire shape** as Jev and our 8876.
- **No trained decision head at all**: readout is pure LM-head over
  single-token candidate labels (`A`–`Z` + `0`–`9` = max 32 options); softmax
  over `logits[label_ids]` at the last position, `logits_to_keep=1`. This is
  the simplest possible Jev-interface design — and directly contradicts our
  assumption that a trained head is needed.
- **`decision_config.json` ships calibrated inference-time scalars**:
  `temperature: 1.348`, `mm_temperature: 1.395` (**separate temperature for
  multimodal inputs**), `threshold: 0.987` (top prob below this → undecided —
  an explicit abstention gate baked into the published artifact).
- **Question types**: choice (2–32), noul, score — same typed triplet.
- **Images**: up to 4 base64 PNG/JPEG/WebP per request, seen by every question;
  training used 1280×800 screenshots.
- **Training data — all real public corpora, no synthetic generator**:
  Mind2Web, xLAM-function-calling-60k (APIGen), xlam-irrelevance, When2Call,
  Banking77, MASSIVE, Aegis 2.0 (CC-BY-4.0), CLINC150 (CC-BY-3.0), BFCL,
  glaive-function-calling-v2, ToolACE, hermes-function-calling-v1,
  Multimodal-Mind2Web (OpenRAIL). Card explicitly states "No Jev outputs were
  used" — same provenance posture as our repo rule.
- **Prompt contract**: system = "Choose the best available candidate for the
  question using only the supplied state. Return exactly one candidate label.";
  user = `State: ... Question: ... Candidates: A) ... \n Answer with exactly
  one candidate label.` — a *serialization* contract, not a trained-head one.

### JEMM-specific lessons for NanoJev

1. **LM-head readout vs trained head**: JEMM gets competitive JevBench-tier
   results with zero head parameters — just label-token softmax + per-modality
   temperature. Combined with Visual Jev's "no consistent typed-head advantage"
   finding, a head-free readout arm is now a **mandatory cheap baseline** in our
   harness before crediting the trained head with any gain.
   **Resolved 2026-09-28 (W135):** we measured both readout arms on
   valen_nano_v3 (2,950 q, CPU fp32, `scripts/lm_head_readout_eval_v1.py`) —
   raw 0.8B base: **0.553** (0/2,950 drops; pure majority-class collapse);
   LoRA-adapted backbone: **0.562** (68/2,950 drops). At 0.8B the trained head
   is *required*, not optional — JEMM's head-free success needs both a ~27B
   backbone and label-format SFT. Untested remainder: label-token SFT per the
   literal JEMM recipe (A100; upside is simplification, not accuracy).
2. **Per-modality temperature**: their `mm_temperature` admits that adding
   images shifts the logit scale — if we go multimodal-by-splice, fit a
   separate temperature per modality, not one global.
3. **Published abstention threshold**: `threshold: 0.987` in the config —
   the artifact itself declares where it stops answering. We should ship a
   per-checkpoint `decision_config.json`-equivalent (our abstention band is
   currently policy-in-docs, not artifact-bound).
4. **Real-data shopping list**: their dataset mix (xlam-irrelevance,
   When2Call, Multimodal-Mind2Web) is literally an "is this context/tool
   relevant" corpus — candidate sources for the REAL_CONTEXT_EVAL tier,
   subject to per-dataset license screen (note: their card labels ToolACE
   Apache-2.0 while prior references list it NC — verify before use).
5. **Wire-compatible**: their `jemm.serve` speaks `/v1/systemone` — if we ever
   want a same-protocol external baseline on our own eval sets, JEMM can be
   pointed at valen_nano eval.jsonl with only a field adapter; likewise our
   8876 service could host a JEMM-style readout arm.

## 1. What it is

- **Jev-Omni** — author `akhilaaa3` (independent; explicitly *not* affiliated
  with TypeSafe AI, and the card states nothing was trained on Jev output).
  Base: `google/gemma-4-12B-it`. License: Apache-2.0 (dataset rights noted as
  separate). Weights: merged checkpoint, no PEFT needed at inference.
  - Repo layout: `backbone/` FP32 merged text model (~50 GB), `unified/` bf16
    checkpoint (~24 GB) combining stock Gemma 4 vision/audio with the
    fine-tuned text model, `head.pt`, `decision_config.json`,
    `runtime_buffers.pt`, `load_model.py`, `jev_omni.py`.
  - Interface: `predict(state, question, options, media=None, modality=...)`
    → `{prediction, prediction_index, confidence, probabilities}` — one
    forward pass, zero generated tokens. Question types: noul / choice /
    score, i.e. the Jev "System One" typed-decision interface.
  - Requires CUDA. Demo Space: https://huggingface.co/spaces/akhilaaa3/jev-omni
  - Eval dataset: https://huggingface.co/datasets/akhilaaa3/decision-bench

- **Adjacent ecosystem** (for recipe comparison):
  - `vagmi/jev-lite` — QLoRA adapter on Gemma 4 E4B, letter-token readout,
    `/v1/systemone` server at github.com/vagmi/jevlite. ECE 0.019 on 1,898
    held-out rows; accuracy flat while ECE fell 0.086→0.019 during training.
  - `autotrust/JEV` (9B) / `JEV-27B` — distilled from TypeSafe Jev 1.13 output
    distributions onto Qwen3.5; two-head packaging (decision LoRA + stock
    lm_head served by one vLLM engine). **Note: trained on Jev outputs —
    off-limits for us per repo rules, useful only as architecture reference.**
  - `shgao/rsi-jev-v1.0-qwen3.5-{0.8b,2b}` — full-tower SFT on soft
    distributions, 7.3M `option_xattn` scorer head, 3 seeds, typed-decisions
    test: 0.614 / 0.662 pooled top-1 vs Jev 0.727. Reports an
    "options reversed" robustness column — worth copying.
  - `junetask/Open-Jev-9B`, `rAVEUK/open-jev-deberta-v3-large` (encoder, 512
    tok ctx, option-group softmax with CE+Brier loss and gold-preserving
    augmentation), `frontier-infra/jebadiah-4b-v1`, `davidburhans/gevva-e2b/e4b`
    (cross-encoder on Gemma 4 E2B/E4B, frozen SigLIP tower, group-atomic
    ranking loss, per-question temperature), Nokia's `anyjev` (training-free
    cyclic-shift option debiasing + label-prior removal + temp scaling).
  - Papers: Visual Jev (arXiv 2609.25845), PixelJev (arXiv 2609.29283),
    RLCDAlignBench (arXiv 2609.29429, uses official Jev for zero-shot
    alignment-failure detection, AUROC 0.886 median).

## 2. Architecture and training recipe (Jev-Omni)

Decision head (`Head256`, from the public loader/Space code):

    z = Linear(hidden=3840 -> 256, fp32)((h_last.float() - mu) / sd)
    z masked_fill(option_index >= k, -1e30); softmax over first k

- `mu`/`sd` are registered buffers (feature standardization before the linear).
- Readout: last-position hidden state captured by a forward hook on the
  decoder; `logits_to_keep: 1` where supported. No generation anywhere.
- Prompt: state + question + options serialized into one user message via
  `apply_chat_template(..., enable_thinking=False)`.

Training (publicly quoted `decision_config.json`, via
https://bittide.aicompass.dev/article/134fa91a-2808-40b6-ab3c-5e6f87575196):

- ~30k-question run total; final stage: 24,000 examples, 1 epoch, 750 steps,
  effective batch 32 (microbatch 8), lr 1e-5 (head and LoRA), warmup 10%,
  seed 3407.
- LoRA **rank 512, alpha 512** — unusually high — applied on top of an
  earlier *already-merged* FP32 v1 + trained head, then merged again
  (`adapters_merged`, checkpoint path `/out/hard4k-20260919/rank128` suggests
  a hard-subset run started from a rank-128 predecessor). Iterative
  merge-then-adapt stacking.
- `verification.json`: argmax(merged) == argmax(adapter), max prob diff
  3.39e-05, exact save/reload parity — they publish numeric merge-provenance
  checks with the weights.

## 3. Multimodal handling — the key design

- Only the **text decoder is fine-tuned**; the Gemma 4 vision and audio
  components are the stock released weights.
- At load time (`jev_omni.py`), the fine-tuned text model is **spliced into
  the full `Gemma4*ForConditionalGeneration` multimodal chassis** via
  `setattr` on the backbone path — i.e., multimodality is inherited, never
  trained. Decision ability transfers to image/audio/video inputs with zero
  multimodal decision data.
- Input shaping: image → one RGB PIL image; video → 16 sampled frames
  inserted as image content items; audio → ffmpeg to 16 kHz mono WAV capped
  at 30 s. One media item per request.
- Reported warm H200 latency: 83 ms (~2k-token text), 26 ms (image),
  31 ms (13 s audio), 504 ms (16-frame video).
- Contrast: Gevva uses a frozen SigLIP tower + cross-encoder head;
  Visual Jev uses a Qwen-VL backbone with LM-head readout and shared visual
  prefix across batched question suffixes; PixelJev reads candidate-token
  logits at the first assistant position on Qwen3.5-VL with optional
  language-side LoRA.

## 4. Evaluation methodology and numbers

Jev-Omni card (self-reported, merged-model results):

| Benchmark | Scope | Macro acc | Micro acc |
|---|---|---:|---:|
| DecisionBench Medium | 80 states / 293 q | 87.57% | 86.01% |
| JevBench matched subset | 195 groups / 231 decisions | 86.15% | 87.45% |
| MMAU (audio) | 1,000 q | — | 63.10% |
| MVBench (video) | 14 tasks / 2,786 q | 53.10% | 53.09% |

- Calibration: DecisionBench Medium ECE **0.0400** (10 bins). For reference
  on the same dataset card: Jev 1.13 micro 88.05%/ECE 0.0423 (medium) and
  68.26%/ECE 0.0894 (hard); frontier chat models 92–99% but worse or
  comparable ECE.
- Official JevBench v1.4.1 (benchmarkheaven.com/jev-models/jev-omni):
  composite **51.3, rank #9** (~75.0% public accuracy, ~32.1% hard tier).
  Composite blends chance-corrected intelligence, calibration, speed, cost.
- Size-coverage honesty: MMAU 63.1% vs Inkling (975B MoE) 77.2%; MVBench
  53.1% vs Qwen3.5-397B-A17B 77.6% — presented as size/coverage comparison,
  not a quality claim. Card also states "best supported at ≤20 options"
  despite the 256-slot head.
- Cost methodology: priced one call per question (state re-sent per
  question) vs chat models one call per state — the card explicitly notes
  the 2.82× token asymmetry rather than hiding it.

## 5. Lessons / actionable items for NanoJev

1. **Multimodal-by-splice**: train only the text decoder + head, then swap it
   into a stock multimodal chassis (Qwen3.5-VL or Gemma 4 E2B/12B). This is
   the cheapest plausible route to image/video-bearing `state` for our
   keep/drop head — zero multimodal training data needed. Candidate: keep
   NanoJev's decision head, re-mount on `Qwen3.5-VL-*` text tower, verify
   text-benchmark parity before claiming multimodality.
2. **Merge-verification receipts**: publish `verification.json`-style checks
   (adapter↔merged argmax + max-prob-diff, save/reload parity) with every
   checkpoint — trivially cheap, directly comparable to our provenance docs.
3. **High-rank LoRA on merged base**: r=512 stacked on a merged predecessor
   worked at 12B; our head-only → LoRA → RLCD ladder is structurally similar.
   Consider reporting adapter rank/merge order in run manifests (we partially
   do via checkpoint config).
4. **Fixed-slot masked head**: Head256 (z-norm + fp32 linear + -1e30 masking)
   is a simple, defensible pattern; ours (`head.decision`/`head.candidate`
   256×1024) is comparable — worth a documented A/B vs LM-head readout,
   since Visual Jev found **no consistent typed-head advantage** over
   LM-head candidate readout. If equal, the simpler readout wins.
5. **Shared-prefix question batching**: Visual Jev's 8.9× speedup at N=32
   questions/image (prefix cached once, isolated question suffixes batched)
   maps directly onto our multi-question `{state, questions}` calls —
   currently each question likely re-encodes state. Measure on MPS.
6. **Calibration as a separate axis**: jev-lite's ECE 0.086→0.019 at flat
   accuracy, and p_max chosen as confidence by measured AUROC (0.816) —
   adopt: report ECE next to accuracy every run, select confidence
   definition by measurement, keep post-hoc temperature as a fitted,
   checkpointed artifact (DeBERTa OpenJev and Gevva both do this).
7. **Option-order robustness**: rsi-jev's "options reversed" column and
   AnyJev L0 cyclic-shift marginalization (23%→7.3% answer flips, training
   free) — add a reversal/rotation eval arm to our harness; cheap control.
8. **Eval hygiene worth copying**: DecisionBench publishes states/questions/
   design notes and prices cost-per-state; NanoJev's frozen-bundle +
   JevBench-public-231 setup already matches this. Jev-Omni's matched-231
   numbers (86.15%/87.45%) give a direct comparison target for our bundle
   runs — aggregates publishable per repo policy.
9. **Provenance discipline**: Jev-Omni's explicit "not trained on Jev output /
   Jev is a TypeSafe trademark" disclaimer is the posture we already enforce;
   avoid autotrust/JEV-style teacher-distilled artifacts entirely for both
   training and calibration fitting (repo rule reaffirmed 2026-09-21).
10. **Honest capability bounds**: "head accepts 256, quality >20 options
    unverified" — mirror this: publish supported option-count ranges and
    context caps per checkpoint instead of implied generality.

## Sources

- https://huggingface.co/MaestroYan/JEMM (model card, adapter, decision_config.json)
- https://github.com/ypcypc/JEMM (/v1/systemone HTTP server)
- https://huggingface.co/akhilaaa3/Jev-Omni (model card, code, commits)
- https://huggingface.co/datasets/akhilaaa3/decision-bench
- https://huggingface.co/spaces/akhilaaa3/jev-omni (Head256 source)
- https://benchmarkheaven.com/jev-models/jev-omni (JevBench v1.4.1 entry)
- https://arxiv.org/abs/2609.25845 (Visual Jev)
- https://arxiv.org/abs/2609.29283 (PixelJev)
- https://huggingface.co/vagmi/jev-lite
- https://huggingface.co/autotrust/JEV
- https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b
- https://huggingface.co/rAVEUK/open-jev-deberta-v3-large
- https://pypi.org/project/gevva/ , https://pypi.org/project/anyjev/
- https://thejevai.com/model/jev-omni , https://decisioneval.dev/models/jev-omni/
- https://bittide.aicompass.dev/article/134fa91a-2808-40b6-ab3c-5e6f87575196
  (decision_config.json transcription)
