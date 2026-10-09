#!/usr/bin/env python3
"""X2 Arm B — SemIf-style direct native-logit readout (no learned parameters).

Reference: github.com/TheoLeeCJ/SemIf (MIT, pinned ca3ba65f142967030ecb453346e94d6f476a69df).
For each question the candidates are rendered once as labelled options inside the
house serialization prefix (identical State:/Question type:/Question: segments as
predict_toy_decisions.prepare_examples — NOT the chat template used by J-D5/T9a),
followed by a `Decision:` marker. One forward pass yields next-token logits at the
decision position; softmax over the label-token slots is the readout distribution.
Nothing is generated; no learned parameters exist.

Label scheme (all verified single clean tokens at load time):
  boolean -> "False"/"True" mapped to candidate ids "false"/"true" (candidate order)
  choice  -> "A".."Z" mapped to criteria keys in insertion order
  score   -> "0".."k-1" mapped to level index strings

255-candidate contract: SemIf's letter path caps at 26 options (one token per
letter). For choice questions with k > 26 this module takes a documented fallback:
each candidate is scored on its own leaf path (byte-identical to prepare_examples'
`prefix + "Candidate:\\n{text}\\nDecision:" + EOS`) by the logit of EOS at the
decision position — i.e. P(model continues the path with EOS) as a per-candidate
scalar — then softmaxed over the candidate set. This keeps the 255 contract with
zero learned parameters; its limitation is that EOS-acceptance is a path-plausibility
proxy, not a label readout, and is unvalidated as a semantic scorer. The option-text
first-token alternative was rejected: first tokens of "key: desc" texts can collide
across candidates, which would silently merge two options' mass.

Also measured per question: label_vocabulary_mass — the share of the full-vocab
softmax sitting on the label slots (J-D5 readout-health diagnostic).

Everything is local-only: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set before any
transformers import; `--source base` loads the pinned Qwen3-0.6B snapshot,
`--source checkpoint` rebuilds a causal LM from a DecisionModel checkpoint by
remapping `backbone.*` weights and reconstructing the tied-embedding LM head
(exactly the J-D5 finetuned_letter method).

Usage:
  .venv/bin/python scripts/x2_semif_readout.py run \
      --source checkpoint --checkpoint-dir checkpoints/domain_adaptation_v4_lora_seed18 \
      --input research/engineering_judgment_corpus_v3/trainer_view/dev.jsonl \
      --rows-out results/x2_semif_transfer/rows_b_lora18_dev.jsonl \
      --metrics-out results/x2_semif_transfer/metrics_b_lora18_dev.json
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_toy_decisions import (  # noqa: E402
    prepare_examples, question_prefix_segments, read_json,
)

BASE_SNAPSHOT = Path(
    "/Users/markus/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOG_FLOOR = 1e-12  # matches scripts/evaluate_pipeline_decisions.py


# ---------------------------------------------------------------- pure core --
def candidate_ids(question):
    if question["type"] == "boolean":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def label_scheme(question):
    """[(candidate_id, label_text)] in candidate order. Raises for choice k>26."""
    typ = question["type"]
    if typ == "boolean":
        return [("false", "False"), ("true", "True")]
    if typ == "choice":
        ids = list(question["criteria"])
        if not 2 <= len(ids) <= len(CHOICE_LABELS):
            raise ValueError(f"choice option count {len(ids)} exceeds letter-label range 2..26")
        return [(cid, CHOICE_LABELS[i]) for i, cid in enumerate(ids)]
    if typ == "score":
        return [(str(i), str(i)) for i in range(len(question["criteria"]))]
    raise ValueError(f"unsupported type: {typ}")


def option_texts(question):
    """Option display text per candidate, matching the leaf-path candidate content."""
    typ = question["type"]
    if typ == "boolean":
        return {"false": "The proposition is false.", "true": "The proposition is true."}
    if typ == "choice":
        return {cid: f"{cid}: {text}" for cid, text in question["criteria"].items()}
    return {str(i): text for i, text in enumerate(question["criteria"])}


def readout_mode(question):
    """'label' for the single-pass SemIf path; 'leaf_eos' for the >26 fallback."""
    if question["type"] == "choice" and len(question["criteria"]) > len(CHOICE_LABELS):
        return "leaf_eos"
    return "label"


def build_prompt_tokens(state, question, tokenizer, marker="Decision:"):
    """Single-pass labelled-option prompt; prefix segments identical to the trainer."""
    labels = label_scheme(question)
    texts = option_texts(question)
    segments = question_prefix_segments(state, question)
    block = ("Candidates:\n" + "".join(f"{label}: {texts[cid]}\n" for cid, label in labels)
             + marker)
    ids = sum([tokenizer.encode(t, add_special_tokens=False) for t in segments], [])
    ids += tokenizer.encode(block, add_special_tokens=False)
    return ids, [cid for cid, _l in labels], [label for _c, label in labels]


def label_token_ids(tokenizer, labels):
    """Verified single clean token per label (J-D5 convention)."""
    slots = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label:
            raise ValueError(f"Not a single clean token: {label!r}")
        slots.append(ids[0])
    return slots


def probs_from_selected_logits(selected):
    """Pure-Python softmax over the candidate slots only; returns probabilities."""
    largest = max(selected)
    exps = [math.exp(z - largest) for z in selected]
    total = math.fsum(exps)
    return [v / total for v in exps]


def gold_key(qtype, gold):
    if qtype == "boolean":
        return "true" if gold else "false"
    if qtype == "score":
        return str(gold)
    return str(gold)


def leaf_paths_for(row, qid, question, tokenizer, max_length):
    """Byte-identical leaf paths via the production serializer (>26 fallback)."""
    payload = {"states": [{"id": row["id"], "state": row["state"],
                           "questions": {qid: question}}]}
    ex = prepare_examples(payload, tokenizer, max_length)[0]
    return ex["leaf_tokens"], ex["candidate_ids"]


# --------------------------------------------------------------- model side --
def load_native_model(source, checkpoint_dir, device):
    """Return (model, tokenizer, metadata). torch/transformers imported lazily."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if source == "base":
        base = BASE_SNAPSHOT.resolve(strict=True)
        if base.name != "c1899de289a04d12100db370d81485cdf75e47ca":
            raise ValueError("Base snapshot revision mismatch")
        model = AutoModelForCausalLM.from_pretrained(
            str(base), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True,
                                                  trust_remote_code=False)
        metadata = {"source": "base", "snapshot": str(base),
                    "weights_sha256": sha256(base / "model.safetensors")}
    elif source == "checkpoint":
        from safetensors.torch import load_file
        root = Path(checkpoint_dir).resolve(strict=True)
        cfg = AutoConfig.from_pretrained(str(root / "backbone_config"),
                                         local_files_only=True, trust_remote_code=False)
        if not cfg.tie_word_embeddings:
            raise ValueError("Cannot reconstruct untrained LM head without tied embeddings")
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa",
                                                 trust_remote_code=False).float()
        weights = load_file(str(root / "best.safetensors"))
        body = {"model." + key[len("backbone."):]: value
                for key, value in weights.items() if key.startswith("backbone.")}
        body["lm_head.weight"] = body["model.embed_tokens.weight"]
        model.load_state_dict(body, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"),
                                                  local_files_only=True,
                                                  trust_remote_code=False)
        metadata = {"source": "checkpoint", "checkpoint": str(root),
                    "weights_sha256": sha256(root / "best.safetensors"),
                    "lm_head_source": "backbone.embed_tokens.weight (tied)"}
        del weights, body
    else:
        raise ValueError(f"unknown source: {source}")
    model.tie_weights()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Expected tied input/output embeddings")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    metadata["tied_embeddings_verified"] = True
    return model, tokenizer, metadata


def _last_logits(model, batch_ids, pad_id, device):
    """One padded forward; return per-row next-token logits at the final real token."""
    import torch
    lengths = torch.tensor([len(ids) for ids in batch_ids], device=device)
    width = int(lengths.max())
    tokens = torch.full((len(batch_ids), width), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_ids):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    rows = logits[torch.arange(len(batch_ids), device=device), lengths - 1].float()
    if not torch.isfinite(rows).all():
        raise ValueError("Nonfinite model logits")
    return rows


def predict_batch_label(model, tokenizer, device, batch, max_length, marker="Decision:"):
    """batch = [(row, qid, question)]; single-pass label readout per question."""
    import torch
    encoded, slots_per_q = [], []
    for row, qid, q in batch:
        ids, cids, labels = build_prompt_tokens(row["state"], q, tokenizer, marker)
        if len(ids) > max_length:
            raise ValueError(f"{row['id']}:{qid} prompt {len(ids)} tokens exceeds {max_length}")
        slots_per_q.append((ids, cids, labels, label_token_ids(tokenizer, labels)))
        encoded.append(ids)
    rows_logits = _last_logits(model, encoded, tokenizer.pad_token_id, device)
    out = []
    for (row, qid, q), (ids, cids, labels, slots), logits in zip(batch, slots_per_q, rows_logits):
        selected = logits[slots]
        probs = probs_from_selected_logits(selected.cpu().tolist())
        mass = float(torch.exp(torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)))
        best = max(range(len(probs)), key=probs.__getitem__)
        out.append({"id": row["id"], "qid": qid, "type": q["type"], "mode": "label",
                    "candidate_ids": cids, "labels": labels,
                    "probabilities": dict(zip(cids, probs)),
                    "selected_id": cids[best], "confidence": probs[best],
                    "label_vocabulary_mass": mass,
                    "input_tokens": len(ids),
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(ids).encode()).hexdigest()})
    return out


def predict_batch_leaf_eos(model, tokenizer, device, batch, max_length):
    """k>26 fallback: per-candidate leaf path, scalar = EOS logit at the decision slot.

    Each leaf path ends "...Decision:" + EOS. The scalar per candidate is the logit
    the model assigns to EOS at the second-to-last position — the slot where the
    model must decide to close this candidate path. Softmax over candidates yields
    a contract-valid distribution for any k with zero learned parameters.
    """
    import torch
    flat, spans = [], []
    for row, qid, q in batch:
        leaves, cids = leaf_paths_for(row, qid, q, tokenizer, max_length)
        flat.extend(leaves)
        spans.append((cids, len(leaves)))
    rows_logits = _last_logits_pre(model, flat, tokenizer.pad_token_id, device)
    eos = tokenizer.eos_token_id
    results, offset = [], 0
    for (row, qid, q), (cids, n) in zip(batch, spans):
        selected = [float(rows_logits[offset + j][eos]) for j in range(n)]
        offset += n
        probs = probs_from_selected_logits(selected)
        best = max(range(len(probs)), key=probs.__getitem__)
        results.append({"id": row["id"], "qid": qid, "type": q["type"],
                        "mode": "leaf_eos_fallback",
                        "candidate_ids": cids, "labels": ["<eos>"] * n,
                        "probabilities": dict(zip(cids, probs)),
                        "selected_id": cids[best], "confidence": probs[best],
                        "label_vocabulary_mass": None,
                        "input_tokens": None,
                        "prompt_sha256": None,
                        "fallback_note": "EOS logit at the pre-EOS decision position per "
                                         "candidate leaf path; path-plausibility proxy, "
                                         "not a label readout"})
    return results


def _last_logits_pre(model, batch_ids, pad_id, device):
    """Like _last_logits but returns logits at the second-to-last real token —
    the position whose next-token prediction is the leaf's final EOS."""
    import torch
    lengths = torch.tensor([len(ids) for ids in batch_ids], device=device)
    width = int(lengths.max())
    tokens = torch.full((len(batch_ids), width), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_ids):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    rows = logits[torch.arange(len(batch_ids), device=device), lengths - 2].float()
    if not torch.isfinite(rows).all():
        raise ValueError("Nonfinite model logits")
    return rows


def run_readout(model, tokenizer, device, rows, batch_size, max_length, marker="Decision:"):
    import torch
    flat = [(row, qid, q) for row in rows for qid, q in row["questions"].items()]
    for row, qid, q in flat:
        if q["type"] not in {"boolean", "choice", "score"}:
            raise ValueError(f"unsupported type {q['type']}")
    results, n_forward = [], 0
    label_batch = [t for t in flat if readout_mode(t[2]) == "label"]
    leaf_batch = [t for t in flat if readout_mode(t[2]) == "leaf_eos"]
    with torch.inference_mode():
        for fn, group in ((predict_batch_label, label_batch),
                          (predict_batch_leaf_eos, leaf_batch)):
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                if fn is predict_batch_label:
                    results.extend(fn(model, tokenizer, device, chunk, max_length, marker))
                else:
                    results.extend(fn(model, tokenizer, device, chunk, max_length))
                n_forward += len(chunk) if fn is predict_batch_label else \
                    sum(len(t[2]["criteria"]) for t in chunk)
    by_key = {(r["id"], r["qid"]): r for r in results}
    return [by_key[(t[0]["id"], t[1])] for t in flat], n_forward


# ------------------------------------------------------------------ scoring --
def score_rows(result_rows, gold_by_id):
    """Accuracy / cov@0.9 / per-type breakdown vs gold (score_t9d_v3 conventions)."""
    by_type, confs, masses = {}, [], []
    n = n_correct = answered = confident_wrong = 0
    for r in result_rows:
        gold = gold_key(r["type"], gold_by_id[r["id"]][r["qid"]])
        probs = r["probabilities"]
        pred = max(probs, key=probs.__getitem__)
        conf = probs[pred]
        correct = int(pred == gold)
        n += 1
        n_correct += correct
        confs.append(conf)
        if r.get("label_vocabulary_mass") is not None:
            masses.append(r["label_vocabulary_mass"])
        if conf >= 0.9:
            answered += 1
            confident_wrong += (1 - correct)
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "answered": 0,
                                           "confident_wrong": 0})
        b["n"] += 1
        b["correct"] += correct
        b["answered"] += int(conf >= 0.9)
        b["confident_wrong"] += (1 - correct) if conf >= 0.9 else 0
    return {"questions": n,
            "accuracy": n_correct / n if n else None,
            "answered_at_0.90_rate": answered / n if n else None,
            "answered_count": answered,
            "confident_wrong_count": confident_wrong,
            "confidence_mean": statistics.fmean(confs) if confs else None,
            "label_vocabulary_mass_mean": statistics.fmean(masses) if masses else None,
            "by_type": {t: {"n": b["n"], "accuracy": b["correct"] / b["n"],
                            "answered_at_0.90": b["answered"],
                            "confident_wrong": b["confident_wrong"]}
                        for t, b in sorted(by_type.items())}}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_gold_map(input_path):
    gold = {}
    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["id"]] = row["gold"]
    return gold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--source", choices=["base", "checkpoint"], required=True)
    p.add_argument("--checkpoint-dir", help="required for --source checkpoint")
    p.add_argument("--input", required=True, help="JSONL rows with id/state/questions/gold")
    p.add_argument("--rows-out", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="mps")
    p.add_argument("--marker", default="Decision:",
                   help="prompt-closing marker; 'Decision:' matches the leaf-path "
                        "serialization, alternatives (e.g. 'Answer:') are diagnostics")
    p.add_argument("--limit", type=int, default=0, help="debug: first N rows only")
    args = p.parse_args()
    if args.source == "checkpoint" and not args.checkpoint_dir:
        p.error("--checkpoint-dir is required for --source checkpoint")
    if min(args.batch_size, args.max_length) <= 0 or args.limit < 0:
        p.error("invalid limits")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    torch.manual_seed(17)

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    gold = load_gold_map(args.input)
    missing = [r["id"] for r in rows if r["id"] not in gold]
    if missing:
        raise ValueError(f"rows missing gold: {missing[:3]}")

    model, tokenizer, metadata = load_native_model(args.source, args.checkpoint_dir,
                                                 torch.device(args.device))
    started = time.monotonic()
    results, n_forward = run_readout(model, tokenizer, torch.device(args.device),
                                     rows, args.batch_size, args.max_length,
                                     marker=args.marker)
    metrics = score_rows(results, gold)
    metrics.update({
        "schema_version": "nanojev-x2-semif-readout-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": metadata,
        "input": {"path": str(args.input), "sha256": sha256(args.input),
                  "rows": len(rows)},
        "methodology": {
            "serialization": "identical State:/Question type:/Question: prefix segments "
                             "as predict_toy_decisions.prepare_examples; single-pass "
                             "Candidates: block + closing marker (no chat template)",
            "closing_marker": args.marker,
            "labels": {"boolean": "False/True -> ids false/true",
                       "choice": "A..Z -> criteria keys in insertion order",
                       "score": "0..k-1 -> level index strings"},
            "fallback_over_26_choice": "per-candidate leaf path, EOS logit at the "
                                     "decision position; proxy readout, see module docstring",
            "temperature": 1.0, "dtype": "float32", "device": args.device,
            "autoregressive_decode_steps": 0, "learned_parameters": 0,
            "forward_passes": n_forward,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "network_model_calls": 0,
        },
        "runtime": {"torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "safetensors": importlib.metadata.version("safetensors")},
        "modes": {m: sum(1 for r in results if r["mode"] == m)
                  for m in sorted({r["mode"] for r in results})},
    })
    rows_path, metrics_path = Path(args.rows_out), Path(args.metrics_out)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", encoding="utf-8") as stream:
        for r in results:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": str(rows_path), "metrics": str(metrics_path),
                      "accuracy": metrics["accuracy"],
                      "by_type": {k: v["accuracy"] for k, v in metrics["by_type"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
