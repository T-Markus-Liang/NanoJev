#!/usr/bin/env python3
"""Large synthetic leakage checks. No market returns, learned performance, or live orders."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes
from financial_pit_v1 import model_input, validate_record, walk_forward


MINUTE = 60_000_000_000
ORIGIN = 1_735_689_600_000_000_000
EVENT = {"name": "synthetic_binary_event", "horizon_ns": 5 * MINUTE,
         "outcome_rule": "minute index plus asset index is even; not market data"}


def records():
    definition = sha256_bytes(canonical_json(EVENT).encode())
    rows = []
    for minute in range(4000):
        for asset in range(3):
            decision = ORIGIN + minute * MINUTE
            row = {"schema_version": "nanojev-financial-pit-v1", "id": f"synthetic-{minute}-{asset}",
                   "asset_id": f"synthetic-asset-{asset}", "venue": "synthetic-only", "decision_ns": decision,
                   "universe_available_ns": ORIGIN - MINUTE,
                   "features": {"signal": {"value": (minute % 11 - 5) / 100, "event_ns": decision - 2_000_000_000,
                                            "available_ns": decision - 1_000_000_000, "fit_cutoff_ns": ORIGIN - MINUTE,
                                            "source_id": "synthetic-only-no-market-feed", "version": "v1"}},
                   "label": {"event": EVENT["name"], "definition_sha256": definition,
                             "end_ns": decision + 5*MINUTE,
                             "available_ns": decision + (10 if minute % 100 == 98 else 5)*MINUTE + 1_000_000_000,
                             "outcome": (minute + asset) % 2 == 0}}
            rows.append(row)
    return rows


def run():
    if not __debug__:
        raise ValueError("verification requires Python assertions enabled")
    started = time.perf_counter()
    rows = records()
    bounds = ((1000, 1400, 1800, 2200), (1400, 1800, 2200, 2600), (1800, 2200, 2600, 3000))
    folds = [{"train": [ORIGIN, ORIGIN+a*MINUTE], "dev": [ORIGIN+a*MINUTE, ORIGIN+b*MINUTE],
              "calibration": [ORIGIN+b*MINUTE, ORIGIN+c*MINUTE], "test": [ORIGIN+c*MINUTE, ORIGIN+d*MINUTE]}
             for a, b, c, d in bounds]
    embargo = 2 * MINUTE
    reports = walk_forward(rows, folds, embargo, ORIGIN + 4001*MINUTE)
    by_id = {row["id"]: row for row in rows}
    for report in reports:
        assert report["all_phases_nonempty"]
        for left, right in (("train", "dev"), ("dev", "calibration"), ("calibration", "test")):
            chosen = [by_id[key] for key in report["retained"][left]]
            assert max(row["label"]["end_ns"] + embargo for row in chosen) < report["windows"][right][0]
            assert max(row["label"]["available_ns"] for row in chosen) <= report["windows"][left][1]
    rejected = 0
    for index in range(1000):
        sample = deepcopy(rows[index])
        case = index % 4
        if case == 0:
            sample["features"]["signal"]["available_ns"] = sample["decision_ns"] + 1
        elif case == 1:
            sample["features"]["signal"]["fit_cutoff_ns"] = sample["decision_ns"] + 1
        elif case == 2:
            sample["universe_available_ns"] = sample["decision_ns"] + 1
        else:
            sample["label"]["available_ns"] = sample["decision_ns"] - 1
        try:
            validate_record(sample)
        except ValueError:
            rejected += 1
    assert rejected == 1000
    for original in rows[:1000]:
        changed = deepcopy(original)
        changed["label"]["outcome"] = not changed["label"]["outcome"]
        changed["label"]["end_ns"] += MINUTE
        changed["label"]["available_ns"] += MINUTE
        assert model_input(original) == model_input(changed)
    return {"schema_version": "nanojev-financial-pit-stress-v1", "status": "passed",
            "scope": "Synthetic declared-timestamp integrity only; no real financial data, training, backtest, transaction costs, profits or live trades.",
            "records": len(rows), "assets": 3, "event_definition": EVENT,
            "future_data_mutations_rejected": rejected, "label_isolation_mutations": 1000,
            "folds": [{"counts": report["counts"], "exclusion_counts": report["exclusion_counts"],
                       "selection_sha256": sha256_bytes(canonical_json(report).encode()),
                       "windows": report["windows"], "all_phases_nonempty": report["all_phases_nonempty"]}
                      for report in reports],
            "dataset_sha256": sha256_bytes(canonical_json(rows).encode()), "embargo_ns": embargo,
            "asof_ns": ORIGIN + 4001*MINUTE,
            "code": [file_identity(__file__), file_identity(Path(__file__).with_name("financial_pit_v1.py"))],
            "elapsed_ms": (time.perf_counter() - started) * 1000}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": report["records"], "status": report["status"]}))
