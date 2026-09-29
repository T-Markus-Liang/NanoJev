#!/usr/bin/env python3
"""Paired local-generation evaluator for real-context holdout manifests.

Reads local raw request files plus local-only downstream contracts, evaluates the
original and the deterministic-shadow reduced request in a pinned local backend,
and emits content-free receipts. No provider calls and no active forwarding.
"""

import argparse
import hashlib
import json
from pathlib import Path

from context_gate_v1 import shadow_request
from local_main_model_evaluator_v1 import (
    DeterministicEvidenceResponder, LocalGenerationResponder, evaluate_pair,
    sha256_text,
)
from main_model_gateway_v1 import build_reduced_request
from real_context_holdout_manifest_v1 import validate_manifest
from run_real_context_holdout_shadow_v1 import _sidecar, deterministic_scorer
from scorer_adapters_v1 import InProcessScorer
from safe_dedup_v1 import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _read_case_request(case_dir, case):
    request_rel = case["request_file"]
    path = (Path(case_dir) / request_rel).resolve()
    if Path(case_dir).resolve() not in path.parents:
        raise ValueError("request_file escapes case_dir")
    raw = path.read_bytes()
    if sha256_bytes(raw) != case["request_sha256"]:
        raise ValueError("request_sha256 mismatch")
    return raw


def _read_contract(contract_dir, case):
    downstream = case.get("downstream") or {}
    path = Path(contract_dir) / f"{case['case_id']}.contract.json"
    if not path.is_file():
        raise ValueError("missing downstream contract file")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    expected = contract.get("expected_answer")
    if downstream.get("expected_answer_sha256") is not None:
        if not isinstance(expected, str) or sha256_text(expected) != downstream["expected_answer_sha256"]:
            raise ValueError("expected_answer_sha256 mismatch")
    declared_hashes = downstream.get("required_evidence_sha256") or []
    required = contract.get("required_strings") or []
    if not isinstance(required, list) or any(not isinstance(value, str) for value in required):
        raise ValueError("required_strings must be a list of strings")
    if declared_hashes and sorted(sha256_text(value) for value in required) != sorted(declared_hashes):
        raise ValueError("required_evidence_sha256 mismatch")
    return contract


def _shadow_reduction(case, raw, threshold):
    _, receipt = shadow_request(raw, case["wire_format"], _sidecar(case),
                                InProcessScorer(deterministic_scorer), threshold=threshold)
    suggested = [segment["pointer"] for segment in receipt.get("segments") or []
                 if segment.get("suggestion") == "drop"]
    reduced = build_reduced_request(raw, case["wire_format"], suggested) if suggested else raw
    return receipt, suggested, reduced


def evaluate_manifest_pairs(manifest_path, case_dir=None, contract_dir=None,
                            protocol_path=DEFAULT_PROTOCOL, threshold=0.90,
                            responder=None):
    manifest_path = Path(manifest_path)
    case_dir = Path(case_dir) if case_dir else manifest_path.parent / "cases"
    contract_dir = Path(contract_dir) if contract_dir else case_dir / "contracts"
    manifest_receipt = validate_manifest(manifest_path, protocol_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    responder = responder or DeterministicEvidenceResponder()
    failures = []
    cases = []
    skipped = []
    for case in manifest.get("cases") or []:
        try:
            if case.get("expected_gate_status") != "scored":
                skipped.append({"case_id": case["case_id"],
                                "reason": case.get("expected_gate_status")})
                continue
            raw = _read_case_request(case_dir, case)
            contract = _read_contract(contract_dir, case)
            receipt, suggested, reduced = _shadow_reduction(case, raw, threshold)
            if receipt.get("status") != "scored":
                raise ValueError(f"gate did not score case: {receipt.get('reason')}")
            pair = evaluate_pair(raw, reduced, contract, responder)
            unsafe = bool(pair.get("answer_regression") or
                          (pair.get("reduced") or {}).get("missing_required_string_sha256"))
            cases.append({
                "case_id": case["case_id"],
                "request_sha256": case["request_sha256"],
                "backend": pair["backend"],
                "suggested_pointers": suggested,
                "original_status": pair["original"]["status"],
                "reduced_status": pair["reduced"]["status"],
                "original_response_sha256": pair["original"]["response_sha256"],
                "reduced_response_sha256": pair["reduced"]["response_sha256"],
                "prompt_tokens_removed": pair["prompt_tokens_removed"],
                "answer_regression": pair["answer_regression"],
                "missing_required_string_sha256":
                    pair["reduced"]["missing_required_string_sha256"],
                "unsafe": unsafe,
            })
            if unsafe:
                failures.append(f"{case['case_id']}: paired regression or missing evidence")
        except Exception as error:
            failures.append(f"{case.get('case_id')}: {error}")
            cases.append({"case_id": case.get("case_id"),
                          "request_sha256": case.get("request_sha256"),
                          "unsafe": True,
                          "error": "paired_evaluation_failed"})
    return {
        "schema_version": "nanojev-real-context-holdout-localgen-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "manifest_validation_ok": manifest_receipt["ok"],
        "backend": getattr(responder, "backend_id", "injected-responder"),
        "threshold": threshold,
        "provider_calls": 0,
        "active_filtering_applied": False,
        "case_count": len(manifest.get("cases") or []),
        "paired_count": len(cases),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "answer_regression_count": sum(1 for case in cases if case.get("answer_regression")),
        "unsafe_count": sum(1 for case in cases if case.get("unsafe")),
        "cases": cases,
        "failures": failures,
        "status": "localgen_eval_pass" if not failures else "localgen_eval_fail",
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--contract-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--backend", choices=["deterministic", "local-generation"],
                        default="deterministic")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counter = load_tokenizer(args.tokenizer)[0] if args.tokenizer else None
    responder = (LocalGenerationResponder(args.model, args.revision, args.device,
                                          counter, args.max_new_tokens,
                                          model_path=args.model_path)
                 if args.backend == "local-generation"
                 else DeterministicEvidenceResponder(counter))
    receipt = evaluate_manifest_pairs(args.manifest, args.case_dir, args.contract_dir,
                                      args.protocol, args.threshold, responder)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": sha256_bytes(encoded.encode()),
                          "ok": receipt["ok"], "status": receipt["status"],
                          "paired": receipt["paired_count"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
