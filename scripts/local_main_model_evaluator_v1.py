#!/usr/bin/env python3
"""Controlled local paired downstream evaluator V1.

This module compares an original and a reduced request without sending either one to
an external provider. The first backend is deliberately deterministic: it answers the
declared expected answer only when every required evidence string survives. It is a
stronger test harness primitive than raw byte counting, but still not a real model
quality claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_restore_v1 import canonical_bytes
from safe_dedup_v1 import load_tokenizer


SCHEMA_VERSION = "nanojev-local-main-model-evaluator-v1"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_request_text(raw):
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", errors="replace")
    return canonical_bytes(decoded).decode("utf-8")


def count_tokens(counter, text):
    if counter is None:
        return None
    try:
        return int(counter(text))
    except Exception:
        return None


def normalize_answer(text):
    text = re.sub(r"<think>.*?</think>", " ", str(text or ""),
                  flags=re.DOTALL | re.IGNORECASE)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [token for token in tokens if token not in {"a", "an", "and", "the"}]
    return " ".join(tokens)


class LocalGenerationResponder:
    """Pinned local causal-LM responder for original/reduced request comparison.

    The raw prompt and raw generation are never stored. ``generate_fn`` is injectable
    so tests can exercise the contract without loading model weights.
    """

    def __init__(self, model="Qwen/Qwen3-0.6B",
                 revision="c1899de289a04d12100db370d81485cdf75e47ca",
                 device="mps", token_counter=None, max_new_tokens=32,
                 generate_fn=None, model_path=None):
        self.model_name = model
        self.model_path = Path(model_path).resolve() if model_path else None
        self.revision = revision
        self.device = device
        self.token_counter = token_counter
        self.max_new_tokens = max_new_tokens
        self.backend_id = f"local-generation:{model}@{revision}"
        self._generate_fn = generate_fn
        self._loaded = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)

        if self.model_path is not None:
            if not (self.model_path / "config.json").exists():
                raise FileNotFoundError(self.model_path)
            model_path = str(self.model_path)
        else:
            cache_snapshot = (Path.home() / ".cache" / "huggingface" / "hub" /
                              ("models--" + self.model_name.replace("/", "--")) /
                              "snapshots" / self.revision)
            if (cache_snapshot / "config.json").exists():
                model_path = str(cache_snapshot)
            else:
                model_path = snapshot_download(self.model_name, revision=self.revision,
                                               local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float16).to(self.device)
        except (ValueError, TypeError, KeyError):
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.float16).to(self.device)
        model.eval()
        self._loaded = (tokenizer, model)
        return self._loaded

    def _generate(self, prompt):
        if self._generate_fn is not None:
            return self._generate_fn(prompt)
        import torch
        tokenizer, model = self._load()
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "Return only the shortest exact answer string."},
                {"role": "user", "content": prompt},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=4096).to(self.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def respond(self, raw, contract):
        text = canonical_request_text(raw)
        required = contract.get("required_strings") or []
        expected_answer = contract.get("expected_answer")
        missing = [value for value in required if value not in raw.decode("utf-8", errors="replace")]
        prompt = (
            "You are a controlled local downstream evaluator. Read the serialized API "
            "request and answer the user's final task. Return only the shortest exact "
            "answer string; do not explain.\n\n"
            "Serialized request:\n" + text + "\n\nFinal answer:"
        )
        generated = self._generate(prompt)
        normalized = normalize_answer(generated)
        expected = normalize_answer(expected_answer)
        answer_ok = bool(expected) and normalized.startswith(expected)
        prompt_tokens = count_tokens(self.token_counter, text)
        response = {"status": "answered" if answer_ok else "answer_mismatch",
                    "answer": generated if answer_ok else None,
                    "usage": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": len(normalized.split()) if normalized else 0}}
        return {
            "backend": self.backend_id,
            "status": response["status"],
            "answer_ok": answer_ok,
            "missing_required_string_sha256": [sha256_text(value) for value in missing],
            "prompt_tokens": prompt_tokens,
            "response_sha256": hashlib.sha256(
                json.dumps(response, ensure_ascii=False, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")).hexdigest(),
        }


class DeterministicEvidenceResponder:
    """A local responder whose answer is controlled by declared evidence requirements."""

    backend_id = "deterministic-evidence-v1"

    def __init__(self, token_counter=None):
        self.token_counter = token_counter

    def respond(self, raw, contract):
        text = raw.decode("utf-8", errors="replace")
        required = contract.get("required_strings") or []
        expected_answer = contract.get("expected_answer")
        missing = [value for value in required if value not in text]
        prompt_tokens = count_tokens(self.token_counter, canonical_request_text(raw))
        answer_ok = (not missing and isinstance(expected_answer, str)
                     and bool(expected_answer))
        response = {
            "status": "answered" if answer_ok else "insufficient_evidence",
            "answer": expected_answer if answer_ok else None,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
        }
        return {
            "backend": self.backend_id,
            "status": response["status"],
            "answer_ok": answer_ok,
            "missing_required_string_sha256": [sha256_text(value) for value in missing],
            "prompt_tokens": prompt_tokens,
            "response_sha256": hashlib.sha256(
                json.dumps(response, ensure_ascii=False, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")).hexdigest(),
        }


def evaluate_pair(original, reduced, contract, responder=None):
    """Compare original/reduced request outcomes without retaining raw text."""
    responder = responder or DeterministicEvidenceResponder()
    original_result = responder.respond(original, contract)
    reduced_result = responder.respond(reduced, contract)
    original_tokens = original_result.get("prompt_tokens")
    reduced_tokens = reduced_result.get("prompt_tokens")
    regression = bool(original_result.get("answer_ok") and not reduced_result.get("answer_ok"))
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": original_result["backend"],
        "original": original_result,
        "reduced": reduced_result,
        "prompt_tokens_removed": (None if original_tokens is None or reduced_tokens is None
                                  else original_tokens - reduced_tokens),
        "answer_regression": regression,
        "pair_status": ("passed" if original_result.get("answer_ok")
                        and reduced_result.get("answer_ok") else "failed"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True,
                        help="Original request JSON file")
    parser.add_argument("--reduced-request", type=Path,
                        help="Reduced request JSON file; defaults to the original")
    parser.add_argument("--contract", type=Path, required=True,
                        help="JSON object with expected_answer and required_strings")
    parser.add_argument("--tokenizer", type=Path,
                        help="Optional local tokenizer.json for usage estimates")
    parser.add_argument("--backend", choices=["deterministic", "local-generation"],
                        default="deterministic")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", type=Path,
                        help="Optional local model directory; bypasses HF cache lookup")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    counter = load_tokenizer(args.tokenizer)[0] if args.tokenizer else None
    responder = (LocalGenerationResponder(args.model, args.revision, args.device,
                                          counter, args.max_new_tokens,
                                          model_path=args.model_path)
                 if args.backend == "local-generation"
                 else DeterministicEvidenceResponder(counter))
    original = args.request.read_bytes()
    reduced = args.reduced_request.read_bytes() if args.reduced_request else original
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = evaluate_pair(original, reduced, contract, responder)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
