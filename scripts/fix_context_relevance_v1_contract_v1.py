#!/usr/bin/env python3
"""Repair the context_relevance_v1 label-contract contradiction into a v2 dataset.

Diagnosis: the original context_relevance_v1 generator emitted byte-identical
candidate templates (modulo scenario ids/values) for candidate_kind "correction"
(labelled keep, irrelevant=false) and "overlap_distractor" (labelled drop,
irrelevant=true). Labels followed the declared arm, not the state content, so
normalized-identical states carry opposite labels.

Contract applied here: a stale/superseded candidate fact is irrelevant=true
(drop) only when the current value it was superseded by is present elsewhere in
the state; when no current value exists, the stale fact is the sole evidence and
the label is irrelevant=false (keep). Per family:

  code         user request already states the new port        -> drop
  order        user request states current quantity/unit_price  -> drop
  risk         "current limit" is never stated                  -> keep
  support      corrected deadline exists only in the candidate  -> keep
  robotics     corrected speed is absent from the state         -> keep
  multilingual corrected time/place are absent from the state   -> keep

Both affected kinds receive the same label per family, so identical normalized
states can no longer disagree. All other records pass through unchanged. Frozen
splits (test/ood) are relabelled by the same contract; they remain frozen for
gate purposes and stay out of training builds.

Usage: python3 scripts/fix_context_relevance_v1_contract_v1.py [src_dir] [dst_dir]
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "context_relevance_v1"
DEFAULT_DST = ROOT / "data" / "context_relevance_v2_seed20260919"

SPLITS = ("train", "dev", "calibration", "test", "ood")
AFFECTED_KINDS = ("correction", "overlap_distractor")
# Whether the superseding current value is already present elsewhere in the
# state (the user request) for the shared correction/overlap_distractor
# template of each family.
CURRENT_VALUE_IN_STATE = {
    "code": True,
    "order": True,
    "risk": False,
    "support": False,
    "robotics": False,
    "multilingual": False,
}
SCHEMA = "nanojev-context-relevance-v2"
SEED = 20260919


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def contract_label(family, kind):
    """Intended irrelevant label for an affected kind under the v2 contract."""
    if kind not in AFFECTED_KINDS:
        return None
    if family not in CURRENT_VALUE_IN_STATE:
        raise ValueError(f"unknown scenario family for affected kind: {family}")
    return CURRENT_VALUE_IN_STATE[family]


def fix_row(row):
    """Return (row, changed). Relabels gold and gold_probs in place."""
    metadata = row["metadata"]
    family, kind = metadata["scenario_family"], metadata["candidate_kind"]
    intended = contract_label(family, kind)
    if intended is None:
        return row, False
    old = row["gold"]["irrelevant"]
    if old == intended:
        return row, False
    row["gold"]["irrelevant"] = intended
    row["gold_probs"]["irrelevant"] = {"false": float(not intended), "true": float(intended)}
    metadata["contract_fix"] = {
        "schema": "context-relevance-contract-v2",
        "previous_irrelevant": old,
        "reason": ("stale candidate superseded by current value already in the user request"
                   if intended else
                   "no current value elsewhere in state; stale candidate is the sole evidence"),
    }
    return row, True


def build(src=DEFAULT_SRC, dst=DEFAULT_DST):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "source": "self_authored_programmatic_relabelled",
        "license": "CC0-1.0",
        "limitations": [
            "Synthetic relevance labels; not production conversations.",
            "Test/OOD remain frozen after this relabel and must never select a checkpoint or threshold.",
            "Tool dependency is represented in canonical synthetic state, not live provider protocol.",
        ],
        "contract_fix": {
            "base": "data/context_relevance_v1",
            "affected_kinds": list(AFFECTED_KINDS),
            "rule": ("correction and overlap_distractor emitted identical normalized states with "
                     "opposite labels; relabelled so a stale candidate is drop(true) only when the "
                     "superseding current value is present elsewhere in the state, else keep(false)"),
            "current_value_in_state": CURRENT_VALUE_IN_STATE,
        },
        "splits": {},
    }
    totals = {}
    for split in SPLITS:
        rows, changed, families, kinds, groups = [], Counter(), Counter(), Counter(), set()
        for line in (src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row, was_changed = fix_row(json.loads(line))
            metadata = row["metadata"]
            if was_changed:
                changed[f"{metadata['scenario_family']}:{metadata['candidate_kind']}"] += 1
            families[metadata["scenario_family"]] += 1
            kinds[metadata["candidate_kind"]] += 1
            groups.add(metadata["source_group_id"])
            rows.append(row)
        text = "".join(serialized(row) + "\n" for row in rows)
        (dst / f"{split}.jsonl").write_text(text, encoding="utf-8")
        manifest["splits"][split] = {
            "records": len(rows),
            "questions": len(rows),
            "families": dict(families),
            "kinds": dict(kinds),
            "source_groups": len(groups),
            "relabelled": dict(changed),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        totals[split] = sum(changed.values())
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"relabelled": totals, "output": str(dst)}))
    return manifest


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST
    build(src, dst)
