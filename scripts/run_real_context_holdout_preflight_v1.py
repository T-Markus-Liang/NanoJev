#!/usr/bin/env python3
"""End-to-end preflight for the real-context holdout pipeline.

Chains protocol validation, optional manifest validation, deterministic shadow
evaluation, and paired local-generation evaluation into one content-free receipt.
No provider calls and no active filtering are performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from local_main_model_evaluator_v1 import (
    DeterministicEvidenceResponder, LocalGenerationResponder,
)
from run_real_context_holdout_localgen_v1 import evaluate_manifest_pairs
from run_real_context_holdout_shadow_v1 import evaluate_manifest
from safe_dedup_v1 import load_tokenizer
from validate_real_context_holdout_protocol_v1 import validate_protocol

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def run_preflight(protocol_path=DEFAULT_PROTOCOL, manifest_path=None,
                  case_dir=None, contract_dir=None, threshold=0.90,
                  responder=None, require_ready=False):
    protocol_receipt = validate_protocol(protocol_path)
    stages = {"protocol": protocol_receipt}
    failures = [] if protocol_receipt["ok"] else ["protocol"]
    manifest = None
    if manifest_path is None:
        status = "protocol_ready_no_manifest" if protocol_receipt["ok"] else "preflight_fail"
        return {
            "schema_version": "nanojev-real-context-holdout-e2e-preflight-v1",
            "status": status,
            "ok": protocol_receipt["ok"],
            "provider_calls": 0,
            "active_filtering_applied": False,
            "stages": stages,
            "failures": failures,
        }

    manifest_path = Path(manifest_path)
    shadow = evaluate_manifest(manifest_path, case_dir, protocol_path, threshold)
    stages["shadow"] = shadow
    manifest = shadow
    if not shadow["ok"]:
        failures.append("shadow")
    if require_ready and shadow.get("ready_for_evaluation") is not True:
        failures.append("manifest_not_ready_for_evaluation")

    if responder is None:
        responder = DeterministicEvidenceResponder()
    localgen = evaluate_manifest_pairs(manifest_path, case_dir, contract_dir,
                                       protocol_path, threshold, responder)
    stages["local_generation"] = localgen
    if not localgen["ok"]:
        failures.append("local_generation")

    return {
        "schema_version": "nanojev-real-context-holdout-e2e-preflight-v1",
        "status": "e2e_preflight_pass" if not failures else "e2e_preflight_fail",
        "ok": not failures,
        "provider_calls": 0,
        "active_filtering_applied": False,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest.get("manifest_sha256"),
            "case_count": manifest.get("case_count"),
            "ready_for_evaluation": manifest.get("ready_for_evaluation"),
        },
        "stages": stages,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--contract-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--backend", choices=["deterministic", "local-generation"],
                        default="deterministic")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--require-ready", action="store_true",
                        help="Fail if manifest coverage does not meet protocol minima")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    counter = load_tokenizer(args.tokenizer)[0] if args.tokenizer else None
    responder = (LocalGenerationResponder(args.model, args.revision, args.device,
                                          counter, args.max_new_tokens,
                                          model_path=args.model_path)
                 if args.backend == "local-generation"
                 else DeterministicEvidenceResponder(counter))
    receipt = run_preflight(args.protocol, args.manifest, args.case_dir,
                            args.contract_dir, args.threshold, responder,
                            args.require_ready)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": sha256_bytes(encoded.encode()),
                          "ok": receipt["ok"], "status": receipt["status"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
