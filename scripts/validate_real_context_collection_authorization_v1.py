#!/usr/bin/env python3
"""Validate the real-context collection authorization template.

The checked artifact is a gate template, not an authorization. It must remain
not-authorized until the owner fills the required scope fields.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUTH = ROOT / "research" / "real_context_collection_authorization_v1.json"
SCHEMA = "nanojev-real-context-collection-authorization-v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def validate_authorization(path=DEFAULT_AUTH, root=ROOT):
    path = Path(path)
    root = Path(root)
    failures = []
    auth = json.loads(path.read_text(encoding="utf-8"))
    require(auth.get("schema_version") == SCHEMA,
            "unsupported collection authorization schema", failures)
    require(auth.get("status") == "template_ready_not_authorized",
            "authorization template status mismatch", failures)

    collection = auth.get("collection") or {}
    require(collection.get("authorized") is False,
            "collection must remain unauthorized in this artifact", failures)
    for field in ("authorized_by", "authorized_at", "scope", "expires_at"):
        require(collection.get(field) is None,
                f"collection.{field} must be null until owner authorization", failures)
    require(type(collection.get("max_cases")) is int and
            1 <= collection.get("max_cases") <= 100,
            "collection.max_cases must be a bounded integer", failures)
    require(collection.get("data_root") == "data/real_context_holdout_v1",
            "collection data_root mismatch", failures)

    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    require("data/*" in gitignore.splitlines(),
            "data root must remain gitignored", failures)

    pre = auth.get("preconditions") or {}
    for field in ("protocol_must_be_valid", "raw_root_must_be_gitignored",
                  "credential_scan_must_pass", "labels_must_be_manual",
                  "v2_v3_isolation_required"):
        require(pre.get(field) is True, f"preconditions.{field} must be true", failures)
    require(pre.get("provider_calls_allowed") is False,
            "provider calls must remain disabled", failures)
    require(pre.get("active_filtering_allowed") is False,
            "active filtering must remain disabled", failures)
    for field in ("protocol", "manifest_builder", "shadow_evaluator",
                  "localgen_evaluator", "e2e_runner"):
        artifact = pre.get(field)
        require(isinstance(artifact, str) and (root / artifact).exists(),
                f"precondition artifact missing: {field}", failures)

    checklist = auth.get("operator_checklist") or []
    require(len(checklist) >= 5, "operator checklist is too short", failures)
    ids = {item.get("id") for item in checklist if isinstance(item, dict)}
    for item_id in ("scope_named", "sources_reviewed", "privacy_reviewed",
                    "labels_prepared", "manifest_built", "preflight_passed",
                    "rollback_ready"):
        require(item_id in ids, f"missing checklist item: {item_id}", failures)
    for item in checklist:
        require(item.get("required") is True,
                f"checklist {item.get('id')} must be required", failures)
        require(item.get("completed") is False,
                f"checklist {item.get('id')} must be incomplete until collection", failures)
        require(isinstance(item.get("description"), str) and item["description"],
                f"checklist {item.get('id')} lacks description", failures)

    forbidden = set(auth.get("forbidden_during_collection") or [])
    for phrase in ("provider calls", "active filtering", "remote scorer calls",
                   "using scorer outputs as labels",
                   "using holdout data for training or calibration"):
        require(phrase in forbidden, f"missing forbidden action: {phrase}", failures)

    stops = set(auth.get("stop_conditions") or [])
    for phrase in ("credential-like pattern is detected",
                   "raw context appears outside the ignored data root",
                   "manifest or E2E preflight fails",
                   "owner revokes scope or requests stop"):
        require(phrase in stops, f"missing stop condition: {phrase}", failures)

    required_fields = set(auth.get("required_owner_fields_before_collection") or [])
    for field in ("collection.authorized=true", "collection.authorized_by",
                  "collection.authorized_at", "collection.scope",
                  "collection.expires_at"):
        require(field in required_fields, f"missing owner field: {field}", failures)

    return {
        "schema_version": "nanojev-real-context-collection-authorization-check-v1",
        "authorization_path": str(path.resolve()),
        "authorization_sha256": sha256_file(path),
        "status": "template_ready_not_authorized" if not failures else "template_invalid",
        "collection_authorized": False,
        "provider_calls_allowed": False,
        "active_filtering_allowed": False,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate_authorization(args.authorization)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "ok": receipt["ok"], "status": receipt["status"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
