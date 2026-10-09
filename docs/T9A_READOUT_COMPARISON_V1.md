# T9a — base model / fine-tuned backbone / trained-head readout

Status (2026-09-20): **READY_FOR_REVIEW — negative drop-in result**. No training, checkpoint replacement, production deployment,
active context removal, or trading is authorized by this diagnostic.

## Question and frozen design

Does the existing SemIf-inspired prompt become a usable, permutation-stable readout when
applied to the unmodified `Qwen/Qwen3-0.6B`, rather than the game-fine-tuned backbone?
This is a **reused 13-question diagnostic**, not a fresh holdout or an upstream benchmark
reproduction. A failure can reject this exact readout as a drop-in; a pass would require
new counterfactual/evidence controls and fresh labelled evaluation, not deployment.

The protocol was written before new base-model inference:
[`t9a_readout_protocol_v1.json`](../research/t9a_readout_protocol_v1.json).
It fixes the original survey bytes and model SHA-256s, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, MPS/FP32, temperature 1.0, threshold 0.9,
the prior probe's prompt, and all permutations of every 2–4-option letter question.
Labels and the seven unlabelled questions are not changed after seeing predictions.

Three arms:

1. Unmodified Qwen3-0.6B with letter-slot next-token logits.
2. Current NanoJev backbone with exactly the same letter prompt and tokenizer.
3. Current NanoJev trained heads with the existing native input contract.

The two letter arms each evaluate 104 question/order combinations. The head evaluates
74: all 66 Choice permutations plus seven Boolean questions and one ordinal Score.
**Boolean/Score permutation is not applicable to the native head**: those orders are
fixed by its contract. Do not inflate the head's control denominator to 13.

Each observation retains the semantic probability vector, selected semantic ID/letter,
confidence, top margin, and (letter arms) prompt hash, token count, and total probability
mass assigned to the selected letters in the **full vocabulary**. A conditional softmax
over letters is not a calibrated probability of correctness. Temperature is a logit
rescaling parameter, not a sampling temperature; no autoregressive tokens are generated.

Strict loading reconstructs the fine-tuned LM projection from its tied input embedding
and refuses missing/unexpected tensors. Inspection found 322 checkpoint tensors,
310 under `backbone.`, one `backbone.embed_tokens.weight`, and no separate `lm_head`.
The input/output tensor pointers must match; tokenizer vocab and chat template must match.
The heads and letter arms necessarily differ in input serialization and scoring architecture:
the comparison is not an isolated causal test of readout alone.

## Interpretation safeguards

- Any semantic argmax flip fails the predeclared drop-in permutation gate. Passing the gate
  is **necessary, not sufficient**; a constant semantic answer can pass while being wrong.
- All six declared labels are Boolean. A constant `true` baseline gets **4/6**. Report it,
  the constant `false` baseline (2/6), and a four-label sensitivity subset excluding generic
  `check_secrets` and `needs_new_test` (both constant baselines 2/4).
- Accuracy is measured only on the declared labels. Permutations are repeated observations,
  not independent samples. Order-average accuracy weights each question equally.
- Coverage at 0.9 and correctness are separate. The seven unlabelled items support no
  correctness or model-quality claim. No threshold/prompt tuning based on this survey.
- The old CPU receipts are historical controls. All new arms use the same MPS/FP32 setting;
  small numerical differences from those historical receipts are not a model improvement.

## Verification and tooling

At input HEAD `47da116`, the only pre-existing untracked file was `AGENTS.md` (preserved).
New code is [`probe_t9a_readout_v1.py`](../scripts/probe_t9a_readout_v1.py).
All result paths are exclusive-create; the earlier negative receipts remain untouched.

Official Worker route reverified: `https://api.deepseek.com/v1`, API ID `deepseek-flash`,
official mapping `DeepSeek-V4.1-Flash`, no third-party fallback. Read-only methodology
review job `1789832916-7d5dc1c422e5` returned **partial**, with concerns rather than approval;
its substantive concerns informed the stricter loader, baselines and necessary-only gate.
Implementation/review job `1789833199-d9bedad1d886` **succeeded**, produced only a test file
in its isolated directory, and ran 45 stdlib tests. The main agent reviewed the full test
file, verified the staged inputs were unchanged, adapted only fixture paths, and reran it.
Neither Worker received weights, credentials in task text, private histories, or service access.

Executed checks:

- Targeted probe suite: **47 passed** (45 Worker-authored checks plus two loading-diagnostic regressions).
- Repository suite: **523 tests OK, 2 skipped**; an existing unclosed-file ResourceWarning
  was also emitted. This is test evidence, not model-quality evidence.
- Repository skill suite: **17 passed**. The installed skill still lacks the newer repository
  scope-guard change; synchronizing that change remains a separate reviewed installation step.
- Offline tokenizer parity: vocab and chat template equal; bare A/B/C/D token IDs 32/33/34/35.
- Real fine-tuned weight-load check: strict load and tied embedding identity passed;
  596,049,920 backbone/LM parameters. No inference was performed in that loading check.

Actual installed-local-skill lifecycle calls (all MPS/FP32, zero remote model calls):

| Phase | Event ID | Outcome |
|---|---|---|
| development | `a1a57e69-86ee-46f7-a64a-da779c4917a8` | abstained, confidence 0.528; main-model fallback recorded |
| testing | `2da02196-9535-4be5-9856-0d98205944c6` | abstained, confidence 0.529; tested independently and feedback recorded |
| optimization | `20ce0b4c-3acc-4656-b8c0-3d21066cdd42` | abstained, confidence 0.552; no threshold change |
| optimization after results | `3d336817-7095-4d63-80c2-3fbece1b39b1` | abstained, confidence 0.519; failed gate determines no replacement, fallback recorded |

No deployment phase has been executed.

## Reproduction

```bash
.venv/bin/python -m unittest discover -s scripts -p test_probe_t9a_readout_v1.py -v
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_t9a_readout_v1.py \
  --base /Users/markus/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir results/t9a_readout_v1_run2
```

The base file must first be complete and match the protocol hash; incomplete downloads are
not accepted. The probe never downloads or changes a service/checkpoint.

## Results

Complete raw receipts and identity/hashes:
[`results/t9a_readout_v1_run2`](../results/t9a_readout_v1_run2/summary.json).
Protocol SHA-256: `be35052d7d75b548087e0de5bc8a803813be28d8c15625cd4bbab8ee2af18561`.
All three arms completed with local-only loading and the same device/precision. The base
download is now complete (1,503,300,328 bytes) and its full SHA-256 matches the protocol.

| Arm | Original median confidence | Original coverage ≥0.9 | Coverage in every order | Semantic invariance | Declared-label accuracy |
|---|---:|---:|---:|---:|---:|
| Base + letter | 0.999963 | 12/13 | 10/13 | **10/13** | 4/6 |
| Fine-tuned + letter | 0.898769 | 6/13 | 5/13 | **0/13** | 4/6 |
| Trained head | 0.504843 | 0/13 | 0/13 | **5/5 Choice**; 8 other items N/A | 3/6 |

The base's six labelled answers are **all `true`** in both orders. Its 4/6 therefore exactly
matches the constant-true baseline, and its four-label sensitivity score is 2/4 (also the
constant baseline). The head scores 1/4 on that subset. None establishes a general accuracy
rate or superiority; these are small, already-inspected diagnostic samples.

Most importantly, the base confidently answers two declared labels incorrectly:

| Question | Expected | Base answer | Original confidence | Stable across reversal? |
|---|---|---|---:|---|
| `safe_to_drop` without a validated gate | false | true | **0.999989** | yes |
| `token_savings` with zero proposed removals | false | true | **0.999974** | yes |

The base's unstable items are `next_phase`, `route`, and `kind`. They are unlabelled;
their correctness is not scored, but their semantic flips do fail the predeclared permutation
gate. The fine-tuned letter arm always selects slot A in **all 104 observations**, reproducing
the old position artifact. The head's five Choice items remain invariant across all 66
permutations; this does not establish engineering usefulness given its zero coverage.

The base's original letter mass is near one, so its confident wrong answers are **not explained
by merely discarding non-letter vocabulary mass**. The fine-tuned arm assigns only about
0.041–0.086 of full-vocabulary mass to the letter slots in original order; its reported
conditional confidence must not be confused with absolute letter probability.

**Conclusion:** reject this exact base-model prompt/readout as a drop-in repair. It raises
coverage but introduces confident wrong answers and does not pass permutation invariance.
Do not swap the production model or lower thresholds. This does **not** prove that all
schema-conditioned approaches fail, that task-fitted heads are intrinsically necessary,
or that new training will succeed. The earlier claim that the root cause was definitively
architectural rather than data-related is withdrawn as unsupported.

Independent official Worker audit `1789833992-f027a69bc587` succeeded: recomputed summaries
directly from raw rows without calling the probe's summary helper; **zero mismatches** in
metrics, permutation sets, argmax, probability sums, input identities or the 104 paired prompt
hashes. Its compact audit is retained beside the results. This validates the arithmetic and
controls, not a production readiness decision.

The next safe work remains the corpus→trainer compatibility/split review (T8g) and a
decision-ready protocol amendment (T9d); actual training still requires its explicit review.
Any future schema-readout variant needs a new protocol and fresh evidence/counterfactual
controls, not tuning this reused survey until it passes.

### Preserved failed execution

The first run computed the base arm but failed while serializing Transformers' `set`-typed
loading diagnostics, before storing any prediction rows. It is preserved in
[`results/t9a_readout_v1/FAILED.md`](../results/t9a_readout_v1/FAILED.md), with the incomplete
file clearly named `.json.partial`. The fix normalizes checked loading diagnostics and
serializes before creating a result file. Two regressions were added. The full run was
repeated in a new directory with **identical protocol bytes**, no prompt/threshold/label
changes, and no predictions used for selection.
