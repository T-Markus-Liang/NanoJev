#!/usr/bin/env python3
"""Frozen, local-only three-arm readout diagnostic; never modifies a checkpoint."""
import argparse
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    # Exclusive creation protects previous positive and negative receipts.
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def checked_loading_info(info):
    # Transformers versions return lists or sets for these diagnostics.
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(info.get(key) for key in fields):
        raise ValueError(f"Base weight loading failed: {info}")
    return {key: sorted(info.get(key, [])) for key in fields}


def options_for(question):
    typ = question["type"]
    if typ == "choice":
        options = [(key, f"{key}: {text}") for key, text in question["criteria"].items()]
    elif typ == "score":
        options = [(str(i), text) for i, text in enumerate(question["criteria"])]
    elif typ == "boolean":
        options = [("true", "The proposition is true."), ("false", "The proposition is false.")]
    else:
        raise ValueError(f"Unsupported type: {typ}")
    if not 2 <= len(options) <= 4 or len({key for key, _ in options}) != len(options):
        raise ValueError("This bounded diagnostic requires 2-4 unique options")
    return options


def orders_for(question, arm):
    size = len(options_for(question))
    if arm == "trained_head" and question["type"] != "choice":
        return [tuple(range(size))]
    return list(itertools.permutations(range(size)))


def observation(options, order, probabilities, letter_mass=None):
    if len(probabilities) != len(options) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probabilities")
    if abs(math.fsum(probabilities) - 1) > 1e-5:
        raise ValueError("Probabilities do not sum to one")
    if sorted(order) != list(range(len(options))):
        raise ValueError("Not a permutation")
    if letter_mass is not None and (not math.isfinite(letter_mass) or not 0 <= letter_mass <= 1):
        raise ValueError("Invalid vocabulary mass")
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    ranked = sorted(probabilities, reverse=True)
    return {
        "order": [options[i][0] for i in order],
        "probabilities": {options[i][0]: p for i, p in zip(order, probabilities)},
        "selected_id": options[order[best]][0],
        "selected_slot": chr(65 + best),
        "confidence": probabilities[best],
        "top_margin": ranked[0] - ranked[1],
        "letter_vocabulary_mass": letter_mass,
    }


def question_summary(rows, permutation_applicable):
    first = rows[0]
    return {
        "permutation_applicable": permutation_applicable,
        "permutations": len(rows),
        "semantic_stable": len({r["selected_id"] for r in rows}) == 1 if permutation_applicable else None,
        "slot_stable": len({r["selected_slot"] for r in rows}) == 1 if permutation_applicable else None,
        "max_probability_delta": max(abs(row["probabilities"][key] - value)
                                     for row in rows for key, value in first["probabilities"].items()),
        "min_top_margin": min(row["top_margin"] for row in rows),
    }


def summarize(rows, labels, threshold):
    applicable = [row for row in rows if row["control"]["permutation_applicable"]]
    originals = [row["observations"][0] for row in rows]
    result = {
        "questions": len(rows),
        "confidence_median": statistics.median(r["confidence"] for r in originals),
        "confidence_max": max(r["confidence"] for r in originals),
        "original_answered": sum(r["confidence"] >= threshold for r in originals),
        "answered_in_every_order": sum(all(r["confidence"] >= threshold for r in row["observations"]) for row in rows),
        "permutation_eligible": len(applicable),
        "semantic_stable": sum(row["control"]["semantic_stable"] for row in applicable),
        "slot_stable": sum(row["control"]["slot_stable"] for row in applicable),
        "drop_in_permutation_gate": bool(applicable) and all(row["control"]["semantic_stable"] for row in applicable),
    }
    result["by_type"] = {}
    for typ in sorted({row["type"] for row in rows}):
        subset = [row for row in rows if row["type"] == typ]
        result["by_type"][typ] = {"n": len(subset),
            "original_answered": sum(row["observations"][0]["confidence"] >= threshold for row in subset)}
    for name, excluded in (("declared_6", set()), ("sensitivity_4", {"check_secrets", "needs_new_test"})):
        selected = [row for row in rows if row["qid"] in labels and row["qid"] not in excluded]
        covered = [row for row in selected if row["observations"][0]["confidence"] >= threshold]
        result[name] = {
            "n": len(selected),
            "original_correct": sum(row["observations"][0]["selected_id"] == labels[row["qid"]] for row in selected),
            "original_covered": len(covered),
            "original_covered_correct": sum(row["observations"][0]["selected_id"] == labels[row["qid"]] for row in covered),
            # Weight each original question equally, not its number of permutations.
            "order_mean_accuracy": statistics.mean(statistics.mean(r["selected_id"] == labels[row["qid"]]
                for r in row["observations"]) for row in selected) if selected else None,
            "all_orders_correct": sum(all(r["selected_id"] == labels[row["qid"]] for r in row["observations"]) for row in selected),
            "constant_true_correct": sum(labels[row["qid"]] == "true" for row in selected),
            "constant_false_correct": sum(labels[row["qid"]] == "false" for row in selected),
        }
    return result


def readout_row(state, qid, question, observations, arm):
    return {"state_id": state["id"], "qid": qid, "type": question["type"],
            "observations": observations,
            "control": question_summary(observations, arm != "trained_head" or question["type"] == "choice")}


def run_head(checkpoint, survey, protocol):
    from predict_toy_decisions import DecisionPredictor
    predictor = DecisionPredictor(checkpoint, device_name=protocol["device"], precision="fp32",
                                  max_length=protocol["max_length"])
    rows = []
    for state in survey["request"]["states"]:
        for qid, question in state["questions"].items():
            options = options_for(question)
            observations = []
            for order in orders_for(question, "trained_head"):
                q = dict(question)
                if q["type"] == "choice":
                    q["criteria"] = {options[i][0]: question["criteria"][options[i][0]] for i in order}
                answer = predictor.predict({"states": [{"id": state["id"], "state": state["state"],
                    "questions": {qid: q}}]}, temperature=protocol["temperature"])["states"][0]["answers"][qid]
                observations.append(observation(options, order, [answer["probabilities"][options[i][0]] for i in order]))
            rows.append(readout_row(state, qid, question, observations, "trained_head"))
            print(f"trained_head {state['id']}:{qid} done", flush=True)
    del predictor
    return rows, {"strict_checkpoint_load": True, "boolean_score_permutation": "not applicable"}


def load_letter_model(arm, base, checkpoint):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if arm == "base_letter":
        model, info = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        metadata = {"loading_info": checked_loading_info(info)}
    else:
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(str(checkpoint / "backbone_config"), local_files_only=True,
                                         trust_remote_code=False)
        if not cfg.tie_word_embeddings:
            raise ValueError("Cannot reconstruct untrained LM head without tied embeddings")
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", trust_remote_code=False).float()
        weights = load_file(str(checkpoint / "best.safetensors"))
        body = {"model." + key[len("backbone."):]: value for key, value in weights.items() if key.startswith("backbone.")}
        # Explicitly reconstruct the tied projection; strict=True forbids random leftovers.
        body["lm_head.weight"] = body["model.embed_tokens.weight"]
        model.load_state_dict(body, strict=True)
        metadata = {"strict_checkpoint_load": True, "lm_head_source": "backbone.embed_tokens.weight (tied)"}
        del weights, body
    model.tie_weights()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Expected tied input/output embeddings")
    metadata["tied_embeddings_verified"] = True
    return model.eval(), metadata


def run_letter(arm, base, checkpoint, survey, protocol):
    import torch
    from transformers import AutoTokenizer
    # One tokenizer for both arms removes an accidental tokenizer confound.
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    other = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != other.get_vocab() or tokenizer.chat_template != other.chat_template:
        raise ValueError("Base/checkpoint tokenizer vocab or chat template differ")
    slots = []
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != letter:
            raise ValueError(f"Not a single clean token: {letter}")
        slots.append(ids[0])
    model, metadata = load_letter_model(arm, base, checkpoint)
    model.to(protocol["device"])
    metadata.update(letter_token_ids=slots, tokenizer_parity=True)
    rows = []
    with torch.inference_mode():
        for state in survey["request"]["states"]:
            for qid, question in state["questions"].items():
                options, observations = options_for(question), []
                for order in orders_for(question, arm):
                    payload = {"evidence": state["state"], "criterion": question["instructions"],
                               "options": [{"letter": chr(65 + i), "description": options[index][1]}
                                           for i, index in enumerate(order)]}
                    messages = [{"role": "system", "content": protocol["system_prompt"]},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=protocol["enable_thinking"])
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if inputs["input_ids"].shape[-1] > protocol["max_length"]:
                        raise ValueError("Refusing prompt truncation")
                    inputs = {key: value.to(protocol["device"]) for key, value in inputs.items()}
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite model logits")
                    selected = logits[slots[:len(options)]]
                    probs = torch.softmax(selected / protocol["temperature"], -1).cpu().tolist()
                    mass = torch.exp(torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)).item()
                    obs = observation(options, order, probs, mass)
                    obs["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                    obs["input_tokens"] = inputs["input_ids"].shape[-1]
                    observations.append(obs)
                rows.append(readout_row(state, qid, question, observations, arm))
                print(f"{arm} {state['id']}:{qid} done", flush=True)
    del model
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "research/t9a_readout_protocol_v1.json")
    parser.add_argument("--base", type=Path, required=True, help="Pinned local snapshot directory; never downloads")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["schema_version"] != "nanojev-t9a-readout-protocol-v1":
        raise ValueError("Unknown protocol")
    if protocol["temperature"] != 1 or protocol["dtype"] != "float32":
        raise ValueError("This diagnostic forbids temperature or precision tuning")
    base = args.base.resolve(strict=True)
    if base.name != protocol["base_revision"]:
        raise ValueError("Base snapshot revision mismatch")
    survey_path, checkpoint = ROOT / protocol["survey"], ROOT / protocol["checkpoint"]
    for path, expected in ((survey_path, protocol["survey_sha256"]),
                           (checkpoint / "best.safetensors", protocol["checkpoint_weights_sha256"]),
                           (base / "model.safetensors", protocol["base_weights_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    survey = json.loads(survey_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    torch.manual_seed(17)
    identity = {"schema_version": "nanojev-t9a-readout-v1", "protocol_sha256": sha256(args.protocol),
                "script_sha256": sha256(__file__), "protocol": protocol, "base_path": str(base),
                "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "device": protocol["device"], "dtype": "float32", "network_model_calls": 0},
                "input_files": {str(path.relative_to(ROOT)): sha256(path) for path in
                                [checkpoint / "config.json", checkpoint / "backbone_config/config.json"]},
                "base_config_sha256": sha256(base / "config.json"),
                "tokenizer_files": {path.name: sha256(path) for path in base.iterdir() if path.suffix in {".json", ".txt", ".jinja"}}}
    write_new(args.output_dir / "identity.json", identity)
    summaries = {}
    for arm in protocol["arms"]:
        start = time.monotonic()
        if arm == "trained_head":
            rows, metadata = run_head(checkpoint, survey, protocol)
        else:
            rows, metadata = run_letter(arm, base, checkpoint, survey, protocol)
        summary = summarize(rows, survey["expected_answers"], protocol["threshold"])
        receipt = {"arm": arm, "identity": identity, "load": metadata, "elapsed_seconds": time.monotonic() - start,
                   "rows": rows, "summary": summary}
        write_new(args.output_dir / f"{arm}.json", receipt)
        summaries[arm] = summary
        print(json.dumps({"arm": arm, "summary": summary}), flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    write_new(args.output_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
