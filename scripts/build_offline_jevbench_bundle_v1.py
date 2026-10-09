#!/usr/bin/env python3
"""Build a provenance-preserving offline JevBench/SemIf benchmark manifest.

The official JevBench public track and SemIf supplemental tracks are kept
separate.  This command performs no network access and never merges labels or
provider outputs across tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OFFICIAL_REVISION = "83831807458d7df424a1e53e5724f3a3ffe2cf89"
SEMIF_REVISION = "ca3ba65f142967030ecb453346e94d6f476a69df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(root: Path, relative: str, *, format_name: str) -> dict:
    path = root / relative
    if not path.is_file():
        return {"path": relative, "status": "missing"}
    record = {
        "path": relative,
        "status": "present",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "format": format_name,
    }
    if format_name == "jsonl":
        record["rows"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return record


def verify(root: Path) -> int:
    manifest_path = root / "offline_bundle_manifest.json"
    if not manifest_path.is_file():
        print(f"FAIL: manifest does not exist: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    declared = set()
    for entry in manifest["files"]:
        declared.add(entry["path"])
        actual = artifact(
            root,
            entry["path"],
            format_name="jsonl" if entry["path"].endswith(".jsonl") else "file",
        )
        for key in ("status", "bytes", "sha256", "rows"):
            if key in entry and actual.get(key) != entry[key]:
                failures.append(
                    f"{entry['path']}: {key} manifest={entry[key]!r} actual={actual.get(key)!r}"
                )
    untracked = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) not in declared
        and p.name != "offline_bundle_manifest.json"
        and p.name != "README.md"
    )
    for path in untracked:
        print(f"WARN untracked file: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"OK {len(declared)} files verified against {manifest_path.name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/jevbench_offline_bundle_v1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify on-disk files against the existing manifest instead of rebuilding it",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"bundle root does not exist: {root}")
    if args.verify:
        raise SystemExit(verify(root))
    official_files = [
        "official_jevbench_v1.2.4_public/original.jsonl",
        "official_jevbench_v1.2.4_public/easy.jsonl",
        "official_jevbench_v1.2.4_public/hard.jsonl",
        "official_jevbench_v1.2.4_public/public_231.jsonl",
        "official_jevbench_v1.2.4_public/manifest.json",
        "official_jevbench_v1.2.4_public/LICENSE",
    ]
    semif_owned = [
        "semif_owned/authored144.jsonl",
        "semif_owned/perturbations108.jsonl",
        "semif_owned/shape777.jsonl",
        "semif_owned/perturbations108-manifest.json",
        "semif_owned/shape777-manifest.json",
        "licenses/SemIf-LICENSE",
    ]
    semif_manifests = [
        "semif_manifests/evaluation-matrix.jsonl",
        "semif_manifests/metric-contract.json",
        "semif_manifests/source-selection.jsonl",
    ]
    external = [
        "semif_external_sources/wanli-test.jsonl",
        "semif_external_sources/every-experiments.json",
        "semif_external_sources/every-source.zip",
        "semif_rebuilt_external/wanli256.jsonl",
        "semif_rebuilt_external/every_rows/inference204.jsonl",
        "semif_rebuilt_external/every_rows/gold154.jsonl",
        "semif_rebuilt_external/every_rows/firewall-actions.json",
        "tools/build_wanli.py",
        "tools/build_every.py",
    ]
    rebuilt_external = [
        "semif_rebuilt_external/wanli256.jsonl",
        "semif_rebuilt_external/every_rows/inference204.jsonl",
        "semif_rebuilt_external/every_rows/gold154.jsonl",
        "semif_rebuilt_external/every_rows/firewall-actions.json",
    ]
    files = []
    for name in official_files:
        files.append(artifact(root, name, format_name="jsonl" if name.endswith(".jsonl") else "file"))
    for name in semif_owned + semif_manifests + external + rebuilt_external:
        files.append(artifact(root, name, format_name="jsonl" if name.endswith(".jsonl") else "file"))

    repo_root = Path(__file__).resolve().parent.parent
    adapter_path = repo_root / "scripts/jevbench_adapter_v1.py"
    contract_path = repo_root / "research/jevbench_adapter_contract_v1.json"
    adapter_binding = {"status": "unavailable"}
    if adapter_path.is_file() and contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected = contract.get("bound_artifacts", {}).get("adapter", {}).get("sha256")
        actual = sha256(adapter_path)
        adapter_binding = {
            "status": "match" if expected == actual else "stale_contract_hash",
            "adapter_sha256": actual,
            "contract_expected_adapter_sha256": expected,
            "contract_path": "research/jevbench_adapter_contract_v1.json",
            "adapter_path": "scripts/jevbench_adapter_v1.py",
        }

    manifest = {
        "schema_version": "nanojev-offline-benchmark-bundle-v1",
        "status": "offline_bundle_provenance_separated",
        "generated_by": "scripts/build_offline_jevbench_bundle_v1.py",
        "adapter_binding": adapter_binding,
        "tracks": [
            {
                "id": "official_jevbench_v1.2.4_public",
                "kind": "official_public_benchmark",
                "source_repo": "https://github.com/fstandhartinger/jevbench",
                "pinned_revision": OFFICIAL_REVISION,
                "license": "MIT",
                "evaluation_only": True,
                "training_allowed": False,
                "calibration_fit_allowed": False,
                "rows": 231,
                "files": [x for x in files if x["path"].startswith("official_")],
            },
            {
                "id": "semif_owned",
                "kind": "supplemental_open_project_authored",
                "source_repo": "https://github.com/TheoLeeCJ/SemIf",
                "pinned_revision": SEMIF_REVISION,
                "license": "MIT (code); fixture provenance retained per source manifests",
                "evaluation_only": True,
                "training_allowed": False,
                "calibration_fit_allowed": False,
                "rows": {"authored144": 144, "perturbations108": 108, "shape777": 777},
                "files": [x for x in files if x["path"].startswith(("semif_owned/", "semif_manifests/"))],
            },
            {
                "id": "semif_external_sources",
                "kind": "supplemental_external_source_snapshots",
                "source_repo": "https://github.com/TheoLeeCJ/SemIf",
                "pinned_revision": SEMIF_REVISION,
                "license": "see source licenses and manifests",
                "evaluation_only": True,
                "training_allowed": False,
                "calibration_fit_allowed": False,
                "files": [x for x in files if x["path"].startswith("semif_external_sources/")],
            },
            {
                "id": "semif_rebuilt_external",
                "kind": "supplemental_rebuilt_evaluation_rows",
                "source_repo": "https://github.com/TheoLeeCJ/SemIf",
                "pinned_revision": SEMIF_REVISION,
                "license": "see source licenses and manifests",
                "evaluation_only": True,
                "training_allowed": False,
                "calibration_fit_allowed": False,
                "rows": {"wanli256": 256, "every_inference204": 204, "every_gold154": 154},
                "files": [x for x in files if x["path"].startswith("semif_rebuilt_external/")],
            },
            {
                "id": "typesafe_selected_102",
                "kind": "not_included_external_snapshot",
                "source": "SemIf source-selection.jsonl + build_typesafe.py",
                "rows": 102,
                "status": "blocked_missing_redistributable_snapshot",
                "reason": "The required TypeSafe case snapshots are not shipped by SemIf; do not reconstruct from provider outputs without the source snapshots.",
                "files": [],
            },
        ],
        "files": files,
        "provenance_rules": [
            "Official JevBench rows remain in the official track and are never mixed with SemIf rows.",
            "All tracks are evaluation-only; no track may enter training or calibration fitting.",
            "Official/provider outputs and probabilities are not bundled as labels.",
            "The TypeSafe selected-102 track is not complete until its source snapshots are independently obtained and hash-verified.",
        ],
    }
    out = args.output or (root / "offline_bundle_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "tracks": len(manifest["tracks"]), "files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
