#!/usr/bin/env python3
"""Assert that no normalized-identical states carry opposite labels.

The check that caught the context_relevance_v1 correction/overlap_distractor
contradiction: states are normalized (digit runs in conversation content and
scenario_id replaced by a placeholder; pointers are left verbatim because
message indices carry positional meaning), then grouped. Any group containing
more than one argmax label is a label-contract violation.

Usage: python3 scripts/validate_valen_label_consistency_v1.py [data_dir]
       data_dir must contain train.jsonl and/or eval.jsonl (valen schema).
Exit code is 1 when contradictions exist, 0 otherwise.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "valen_nano_v2"
DIGITS = re.compile(r"\d+(\.\d+)?")


def normalized_state(state_text):
    """Normalize surface ids/numbers inside message content only."""
    try:
        state = json.loads(state_text)
    except (TypeError, json.JSONDecodeError):
        return DIGITS.sub("<N>", str(state_text))
    for message in state.get("conversation", []):
        if "content" in message:
            message["content"] = DIGITS.sub("<N>", message["content"])
    if "scenario_id" in state:
        state["scenario_id"] = DIGITS.sub("<N>", state["scenario_id"])
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def argmax_label(record):
    probs = record["targets"]["irrelevant"]["probabilities"]
    return max(probs.items(), key=lambda kv: kv[1])[0]


def check_dir(data_dir):
    data_dir = Path(data_dir)
    failures, report = [], {"records": 0, "normalized_states": 0, "violations": 0}
    groups = defaultdict(list)
    for name in ("train", "eval"):
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            groups[(name, normalized_state(record["request"]["state"]))].append(record)
            report["records"] += 1
    report["normalized_states"] = len(groups)
    for (name, _), records in sorted(groups.items()):
        labels = Counter(argmax_label(record) for record in records)
        if len(labels) > 1:
            report["violations"] += len(records)
            kinds = Counter((record["meta"]["source_dataset"], record["meta"]["candidate_kind"])
                            for record in records)
            failures.append(f"{name}: {len(records)} records share a normalized state with labels "
                            f"{dict(labels)}; kinds={dict(kinds)}")
    return failures, report


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    failures, report = check_dir(data_dir)
    print(json.dumps(report, sort_keys=True))
    for failure in failures:
        print("VIOLATION:", failure)
    if failures:
        sys.exit(1)
    print("OK: no normalized-identical states carry opposite labels")


if __name__ == "__main__":
    main()
