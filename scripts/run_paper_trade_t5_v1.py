#!/usr/bin/env python3
"""Freeze derived protocols and run the cached-data T5 matrix; never fetch or overwrite."""
import argparse
import itertools
import json
import pathlib
import subprocess
import sys

from paper_trade_protocol_v1 import load_protocol, sha256_file
from paper_trade_report_v1 import ROOT, build_report, derived_protocol, load_study, write_new


def audit_inputs(base):
    census = {}
    for venue, manifest_path in base["input_manifests"].items():
        manifest = json.loads((ROOT / manifest_path).read_text())
        paths = set()
        for record in manifest["files"]:
            path = ROOT / record["local_path"]
            if path.resolve() in paths:
                raise ValueError("duplicate manifest file")
            paths.add(path.resolve())
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"cached input changed: {path}")
        # Binance loader globs every zip, so undeclared extras could change its input.
        if venue == "binance":
            used = set()
            root = ROOT / "data/binance_vision_v1"
            for symbol in base["symbols"]:
                for series in ("markPriceKlines", "klines", "indexPriceKlines", "fundingRate"):
                    used.update(p.resolve() for p in (root / series / symbol).glob("*.zip"))
        else:
            root = ROOT / "data/venue_perp_v1" / venue
            series = ["kline", "mark", "index", "funding"] + (["funding_info"] if venue == "aster" else [])
            used = {(root / f"{symbol}.{name}.json").resolve()
                    for symbol in base["symbols"] for name in series}
        if not used <= paths:
            raise ValueError(f"loader input missing from manifest: {used - paths}")
        census[venue] = {"manifest_path": manifest_path,
                         "manifest_sha256": sha256_file(ROOT / manifest_path),
                         "checked_files": len(paths), "all_file_hashes_match": True,
                         "loader_files_covered": len(used), "declared_venue": manifest.get("venue"),
                         "declared_base_url": manifest.get("base_url"),
                         "metadata_warning": "Aster manifest names Hyperliquid base_url; preserved, not verified" if venue == "aster" else None}
    return census


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=pathlib.Path, default=ROOT / "research/paper_trade_t5_sensitivity_v1.json")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--report-output", type=pathlib.Path)
    args = parser.parse_args()
    if args.report_output is not None and args.report_output.exists():
        parser.error("report output already exists; evidence is append-only")
    study = load_study(args.study)
    base, evidence = load_protocol(ROOT / study["base_protocol"])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/paper_trade_perp_v1.py", "scripts/paper_trade_attribution_v1.py",
               "scripts/paper_trade_report_v1.py", "scripts/run_paper_trade_t5_v1.py",
               "scripts/financial_simulator_v1.py", "scripts/financial_backtest_v1.py",
               "scripts/build_perp_pit_v1.py", "scripts/paper_trade_protocol_v1.py",
               "scripts/benchmark_nanojev_v2.py"]
    source_hashes = {p: sha256_file(ROOT / p) for p in sources}
    before = audit_inputs(base)
    study_hash = sha256_file(args.study)
    write_new(out / "frozen_inputs.json", {"study": study, "study_sha256": study_hash,
              "base_protocol_sha256": evidence["protocol_sha256"],
              "source_sha256": source_hashes, "input_census": before})
    protocols = {}
    for window, seed in itertools.product(study["windows"], study["seeds"]):
        path = out / "protocols" / f"{window}-{seed}.json"
        write_new(path, derived_protocol(base, window, study["windows"][window], seed))
        protocols[window, seed] = path
    runs, receipts = [], []
    try:
        for venue, window, seed in itertools.product(study["venues"], study["windows"], study["seeds"]):
            path = out / "receipts" / f"{venue}-{window}-{seed}.json"
            protocol = protocols[window, seed]
            command = [sys.executable, "scripts/paper_trade_perp_v1.py", "--source", venue,
                       "--protocol", str(protocol), "--protocol-sha256", sha256_file(protocol),
                       "--output", str(path)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            runs.append({"cell": [venue, window, seed], "returncode": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
            print(json.dumps({"cell": [venue, window, seed], "returncode": completed.returncode}), flush=True)
            if completed.returncode:
                raise ValueError(f"replay failed: {path}: {completed.stderr}")
            receipts.append(path)
        # Byte determinism of one repeated full-window cell, not robustness evidence.
        first = out / "receipts" / f"{study['venues'][0]}-full-{study['seeds'][0]}.json"
        repeated = out / "repeat_receipt.json"
        protocol = protocols["full", study["seeds"][0]]
        repeat = subprocess.run([sys.executable, "scripts/paper_trade_perp_v1.py", "--source", study["venues"][0],
                                 "--protocol", str(protocol), "--protocol-sha256", sha256_file(protocol),
                                 "--output", str(repeated)], cwd=ROOT, capture_output=True, text=True, timeout=120)
        if repeat.returncode or first.read_bytes() != repeated.read_bytes():
            raise ValueError(f"repeated cell differs: {repeat.stderr}")
        after = audit_inputs(base)
        if before != after or source_hashes != {p: sha256_file(ROOT / p) for p in sources}:
            raise ValueError("inputs or source changed during study")
        if study_hash != sha256_file(args.study) or evidence["protocol_sha256"] != sha256_file(ROOT / study["base_protocol"]):
            raise ValueError("protocol changed during study")
        report = build_report(receipts, study, base, evidence["protocol_sha256"])
        report["study_sha256"] = study_hash
        report["execution_verification"] = {"input_census": after, "source_sha256": source_hashes,
                    "inputs_and_source_unchanged": True, "repeat_cell_byte_identical": True,
                    "repeat_cell_sha256": sha256_file(repeated), "remote_data_calls": 0}
        write_new(args.report_output or out / "report.json", report)
    except Exception as error:
        write_new(out / "failure.json", {"error": str(error), "completed_runs": len(runs)})
        raise
    finally:
        write_new(out / "run_log.json", runs)


if __name__ == "__main__":
    main()
