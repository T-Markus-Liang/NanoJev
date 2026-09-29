#!/usr/bin/env python3
"""J-D5 backbone-vs-head attribution: base-model label readout on labelled cohorts.

Extends the T9a three-arm readout method (scripts/probe_t9a_readout_v1.py) from the
13-question abstention survey to the 84-item engineering-judgment test cohort and the
88-row workflow subset. For every question the candidate set is rendered as labelled
options inside a chat prompt; a single fp32 forward pass yields next-token logits, and
the softmax over the option-label token slots is the readout distribution. No tokens are
generated. Label scheme per question type:

  boolean -> labels "True"/"False" mapped to ids "true"/"false" (True shown first)
  choice  -> labels "A".."T" mapped to criteria keys in insertion order
  score   -> labels "0".."k" mapped to level index strings

Arms: ``base_letter`` (pinned Qwen3-0.6B snapshot) and ``finetuned_letter`` (NanoJev
backbone with the tied-embedding LM head reconstructed exactly as in T9a). The trained
head and Jev arms are not re-run here; ``score`` merges their frozen receipts.

Everything is local-only: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set before any
transformers import, ``--base`` must be a resolved snapshot directory, and all outputs
are written inside results/jd5_backbone_attribution/.

Usage:
  .venv/bin/python scripts/jd5_letter_readout.py run --output-dir results/jd5_backbone_attribution
  .venv/bin/python scripts/jd5_letter_readout.py score --output-dir results/jd5_backbone_attribution
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
LOG_FLOOR = 1e-12  # matches scripts/evaluate_pipeline_decisions.py

BASE_SNAPSHOT = Path(
    "/Users/markus/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
CHECKPOINT = ROOT / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17"

SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed "
    "option. Respond with only that option's label, with no explanation or reasoning."
)

COHORTS = {
    "eng_test": {
        "input": ROOT / "research/engineering_judgment_corpus_v2/trainer_view/test.jsonl",
        "nanojev_pred": Path("/tmp/jev_runs/l3_nanojev_pred.json"),
        "jev_responses": [
            ROOT / "results/jev_sidebyside_v1/l3_eng_v2_test/responses.jsonl",
            ROOT / "results/jev_sidebyside_v1/l3_retry/responses.jsonl",
        ],
    },
    "l2_sub": {
        "input": Path("/tmp/jev_runs/l2_sub.jsonl"),
        "nanojev_pred": Path("/tmp/jev_runs/l2_nanojev_pred.json"),
        "jev_responses": [ROOT / "results/jev_sidebyside_v1/l2_workflow_test_stride8/responses.jsonl"],
    },
}

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def options_for(question):
    """Return [(candidate_id, label_token, description)] in presentation order."""
    typ = question["type"]
    criteria = question.get("criteria")
    if typ == "choice":
        ids = list(criteria.keys())
        if not 2 <= len(ids) <= len(CHOICE_LABELS):
            raise ValueError(f"choice option count out of range: {len(ids)}")
        return [(cid, CHOICE_LABELS[i], f"{cid}: {criteria[cid]}") for i, cid in enumerate(ids)]
    if typ == "score":
        return [(str(i), str(i), text) for i, text in enumerate(criteria)]
    if typ == "boolean":
        return [("true", "True", "The proposition is true."),
                ("false", "False", "The proposition is false.")]
    raise ValueError(f"Unsupported type: {typ}")


def gold_dist(row, qid, question):
    """Return (q dict over candidate ids, kind) from gold_probs or hard gold."""
    ids = candidate_ids_for(question)
    gp = (row.get("gold_probs") or {}).get(qid)
    if gp is not None:
        vec = norm_probs(gp, ids)
        if vec is None:
            raise ValueError(f"gold_probs has no mass: {row['id']}:{qid}")
        q = dict(zip(ids, vec))
        kind = "deterministic_truth" if sum(v != 0 for v in vec) == 1 else "soft_distribution"
        return q, kind
    gold = row["gold"][qid]
    gid = ("true" if gold is True else "false" if gold is False else str(gold))
    return {cid: float(cid == gid) for cid in ids}, "deterministic_truth"


def gold_id(row, qid, question):
    q, _kind = gold_dist(row, qid, question)
    return max(q, key=q.__getitem__)


def checked_loading_info(info):
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(info.get(key) for key in fields):
        raise ValueError(f"Base weight loading failed: {info}")
    return {key: sorted(info.get(key, [])) for key in fields}


def load_letter_model(arm, base, checkpoint):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if arm == "base_letter":
        model, info = AutoModelForCausalLM.from_pretrained(
            str(base), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        metadata = {"loading_info": checked_loading_info(info)}
    elif arm == "finetuned_letter":
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(str(checkpoint / "backbone_config"),
                                         local_files_only=True, trust_remote_code=False)
        if not cfg.tie_word_embeddings:
            raise ValueError("Cannot reconstruct untrained LM head without tied embeddings")
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa",
                                                 trust_remote_code=False).float()
        weights = load_file(str(checkpoint / "best.safetensors"))
        body = {"model." + key[len("backbone."):]: value
                for key, value in weights.items() if key.startswith("backbone.")}
        body["lm_head.weight"] = body["model.embed_tokens.weight"]
        model.load_state_dict(body, strict=True)
        metadata = {"strict_checkpoint_load": True,
                    "lm_head_source": "backbone.embed_tokens.weight (tied)"}
        del weights, body
    else:
        raise ValueError(f"Unknown letter arm: {arm}")
    model.tie_weights()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Expected tied input/output embeddings")
    metadata["tied_embeddings_verified"] = True
    return model.eval(), metadata


def label_token_ids(tokenizer, options):
    slots = []
    for _cid, label, _desc in options:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label:
            raise ValueError(f"Not a single clean token: {label!r}")
        slots.append(ids[0])
    return slots


def run_letter_arm(arm, base, checkpoint, cohort_name, rows, device):
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True,
                                              trust_remote_code=False)
    other = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"),
                                          local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != other.get_vocab() or tokenizer.chat_template != other.chat_template:
        raise ValueError("Base/checkpoint tokenizer vocab or chat template differ")
    model, metadata = load_letter_model(arm, base, checkpoint)
    model.to(device)
    metadata["tokenizer_parity"] = True
    out_rows = []
    n_forward = 0
    with torch.inference_mode():
        for row in rows:
            for qid, question in row["questions"].items():
                options = options_for(question)
                slots = label_token_ids(tokenizer, options)
                payload = {"evidence": row["state"], "criterion": question["instructions"],
                           "options": [{"label": label, "description": desc}
                                       for _cid, label, desc in options]}
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True,
                                                       enable_thinking=False)
                inputs = tokenizer(prompt, return_tensors="pt")
                if inputs["input_ids"].shape[-1] > 2048:
                    raise ValueError("Refusing prompt truncation")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                n_forward += 1
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite model logits")
                selected = logits[slots]
                probs = torch.softmax(selected, -1).cpu().tolist()
                mass = torch.exp(torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)).item()
                best = max(range(len(probs)), key=probs.__getitem__)
                out_rows.append({
                    "id": row["id"], "qid": qid, "type": question["type"],
                    "gold_id": gold_id(row, qid, question),
                    "candidate_ids": [cid for cid, _l, _d in options],
                    "labels": [label for _c, label, _d in options],
                    "probabilities": {cid: p for (cid, _l, _d), p in zip(options, probs)},
                    "selected_id": options[best][0],
                    "selected_label": options[best][1],
                    "confidence": probs[best],
                    "label_vocabulary_mass": mass,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                })
            print(f"{arm} {cohort_name} {row['id']} done", flush=True)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out_rows, metadata, n_forward


# ---------------------------------------------------------------- scoring ----

def norm_probs(probs, ids):
    """Fill missing candidate ids with 0 and renormalize; None if no mass."""
    vec = [max(0.0, float(probs.get(cid, 0.0))) for cid in ids]
    total = math.fsum(vec)
    if total <= 0:
        return None
    return [v / total for v in vec]


def nanojev_probs(answer, ids, typ):
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        return None
    return norm_probs(probs, ids)


def jev_probs(answer, ids, typ):
    if typ == "boolean":
        p = answer.get("probability")
        if type(p) not in {int, float} or not 0 <= p <= 1:
            return None
        raw = {"false": 1.0 - p, "true": p}
    else:
        raw = answer.get("probabilities")
        if not isinstance(raw, dict):
            return None
    return norm_probs(raw, ids)


def candidate_ids_for(question):
    typ = question["type"]
    criteria = question.get("criteria")
    if typ == "boolean":
        return ["false", "true"]
    if typ == "choice":
        return list(criteria.keys())
    if typ == "score":
        return [str(i) for i in range(len(criteria))]
    raise ValueError(typ)


def score_arms(cohort_name):
    spec = COHORTS[cohort_name]
    rows = [json.loads(line) for line in spec["input"].read_text().splitlines() if line.strip()]

    arms = {}

    # Letter arms produced by `run`.
    for arm in ("base_letter", "finetuned_letter"):
        path = spec["_rows_dir"] / f"rows_{arm}_{cohort_name}.jsonl"
        if not path.is_file():
            continue
        letter = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            letter[(r["id"], r["qid"])] = r
        arms[arm] = lambda row, qid, q, _l=letter: (
            None if (row["id"], qid) not in _l
            else [_l[(row["id"], qid)]["probabilities"][cid] for cid in candidate_ids_for(q)])

    # NanoJev trained head (frozen receipt).
    pred = json.loads(spec["nanojev_pred"].read_text())
    nano = {}
    for state in pred["states"]:
        for qid, answer in state["answers"].items():
            nano[(state["id"], qid)] = answer
    arms["nanojev_head"] = lambda row, qid, q: nanojev_probs(
        nano.get((row["id"], qid), {}), candidate_ids_for(q), q["type"])

    # Jev gateway baseline (frozen receipts).
    jev = {}
    for path in spec["jev_responses"]:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for qid, answer in r.get("answers", {}).items():
                jev[(r["id"], qid)] = answer
    arms["jev"] = lambda row, qid, q: jev_probs(
        jev.get((row["id"], qid), {}), candidate_ids_for(q), q["type"])

    metrics = {}
    per_item = {}
    for arm, getter in arms.items():
        by_type = {}
        for row in rows:
            for qid, q in row["questions"].items():
                ids = candidate_ids_for(q)
                p = getter(row, qid, q)
                qdist, qkind = gold_dist(row, qid, q)
                qv = [qdist[cid] for cid in ids]
                gold = ids[max(range(len(qv)), key=qv.__getitem__)]
                rec = per_item.setdefault((row["id"], qid), {
                    "id": row["id"], "qid": qid, "type": q["type"], "gold_id": gold,
                    "gold_kind": qkind})
                if p is None:
                    rec[arm] = None
                    continue
                best = max(range(len(p)), key=p.__getitem__)
                ce = -math.fsum(a * math.log(max(b, LOG_FLOOR))
                                for a, b in zip(qv, p) if a)
                brier = math.fsum(x * x for x in p) - 2 * math.fsum(
                    a * b for a, b in zip(qv, p)) + 1
                rec[arm] = {"selected_id": ids[best], "confidence": p[best],
                            "correct": ids[best] == gold,
                            "cross_entropy": ce, "expected_brier": brier}
                bucket = by_type.setdefault(q["type"], {"n": 0, "answered": 0, "correct": 0,
                                                        "nll": [], "brier": [], "conf": []})
                bucket["n"] += 1
                bucket["answered"] += 1
                bucket["correct"] += int(ids[best] == gold)
                bucket["nll"].append(ce)
                bucket["brier"].append(brier)
                bucket["conf"].append(p[best])
        metrics[arm] = {}
        for typ, b in sorted(by_type.items()):
            metrics[arm][typ] = {
                "n": b["n"], "answered": b["answered"],
                "accuracy": b["correct"] / b["n"] if b["n"] else None,
                "nll": math.fsum(b["nll"]) / len(b["nll"]) if b["nll"] else None,
                "brier": math.fsum(b["brier"]) / len(b["brier"]) if b["brier"] else None,
                "confidence_mean": statistics.fmean(b["conf"]) if b["conf"] else None,
            }
    return metrics, per_item, {"rows": len(rows),
                               "jev_items": len(jev), "nanojev_items": len(nano)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "score"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE_SNAPSHOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--cohorts", default="eng_test,l2_sub")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    if set(cohorts) - set(COHORTS):
        raise ValueError(f"unknown cohort: {cohorts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in COHORTS.values():
        spec["_rows_dir"] = args.output_dir

    if args.command == "run":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1")
        base = args.base.resolve(strict=True)
        checkpoint = args.checkpoint.resolve(strict=True)
        if base.name != "c1899de289a04d12100db370d81485cdf75e47ca":
            raise ValueError("Base snapshot revision mismatch")
        import torch
        import transformers
        torch.manual_seed(17)
        protocol = {
            "schema_version": "nanojev-jd5-letter-readout-v1",
            "extends": "nanojev-t9a-readout-protocol-v1",
            "purpose": "J-D5 backbone-vs-head attribution on labelled cohorts; diagnostic only, "
                       "no training/deployment/promotion authorized.",
            "base_model": "Qwen/Qwen3-0.6B",
            "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "base_weights_sha256": sha256(base / "model.safetensors"),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_weights_sha256": sha256(checkpoint / "best.safetensors"),
            "system_prompt": SYSTEM_PROMPT,
            "label_scheme": {"boolean": "labels True/False -> ids true/false (True first)",
                             "choice": "labels A.. -> criteria keys in insertion order",
                             "score": "labels 0..k -> level index strings"},
            "deviation_from_t9a_prompt": "T9a used uppercase letters for every type and the "
                "key 'letter'; J-D5 uses per-type labels (True/False, digits, letters) under "
                "the key 'label' with a matching one-word system-prompt change.",
            "permutations": "none - single canonical presentation order per question",
            "temperature": 1.0, "dtype": "float32", "device": args.device,
            "max_length": 2048, "enable_thinking": False, "autoregressive_decode_steps": 0,
            "metrics": ["per-type accuracy vs gold", "nll with 1e-12 log floor",
                        "multiclass brier", "label vocabulary mass"],
            "cohorts": {name: {"input": str(COHORTS[name]["input"]),
                               "input_sha256": sha256(COHORTS[name]["input"])}
                        for name in cohorts},
            "script_sha256": sha256(__file__),
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__},
            "network_model_calls": 0,
        }
        protocol_path = args.output_dir / "protocol_jd5.json"
        if not protocol_path.is_file():
            write_json(protocol_path, protocol)
        for arm in ("base_letter", "finetuned_letter"):
            for name in cohorts:
                spec = COHORTS[name]
                rows = [json.loads(l) for l in spec["input"].read_text().splitlines() if l.strip()]
                rows_path = args.output_dir / f"rows_{arm}_{name}.jsonl"
                receipt_path = args.output_dir / f"receipt_{arm}_{name}.json"
                if rows_path.is_file() and receipt_path.is_file():
                    print(json.dumps({"arm": arm, "cohort": name, "skipped": "exists"}))
                    continue
                start = time.monotonic()
                out_rows, metadata, n_forward = run_letter_arm(
                    arm, base, checkpoint, name, rows, args.device)
                write_jsonl(rows_path, out_rows)
                write_json(receipt_path, {
                    "arm": arm, "cohort": name, "load": metadata,
                    "forward_passes": n_forward, "questions": len(out_rows),
                    "elapsed_seconds": time.monotonic() - start})
                print(json.dumps({"arm": arm, "cohort": name,
                                  "questions": len(out_rows)}), flush=True)
    else:
        result = {}
        for name in cohorts:
            metrics, per_item, counts = score_arms(name)
            result[name] = {"counts": counts, "metrics": metrics}
            write_jsonl(args.output_dir / f"per_item_{name}.jsonl",
                        [per_item[k] for k in sorted(per_item)])
        write_json(args.output_dir / "comparison.json", {
            "schema_version": "nanojev-jd5-comparison-v1",
            "nll_definition": f"cross-entropy -sum(q*log(max(p,{LOG_FLOOR}))) vs gold_probs; "
                              "equals -log p_gold for one-hot deterministic_truth targets",
            "brier_definition": "expected Brier sum(p^2) - 2*dot(q,p) + 1 (repo convention)",
            "jev_note": "Jev probability vectors renormalized over candidate ids; missing "
                        "candidates count as 0 mass. Boolean arm uses p_true.",
            "cohorts": result})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
