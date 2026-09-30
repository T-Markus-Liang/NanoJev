#!/usr/bin/env python3
"""Assemble the v5 four-tier eval manifest -> data/v5_eval/eval_manifest.json.

Implements docs/V5_DATA_DESIGN_V1.md §7.2 (eval composition) and the tier
layout consumed by scripts/check_v5_gates_v1.py (G1-G6, §9).

READ-ONLY over all inputs: this script never copies eval payloads. It emits a
merged manifest of *pointers* (path + sha256 + record/label counts) so scoring
runs and the gate checker resolve the canonical files themselves.

Tiers
-----
T1  real-frozen     existing 590 labels = 525 main + 65 drop-supp; the fixed
                    holdout (training_allowed=false). Primary G1/G2/G6 input.
T2  real-v5-ext     mined-eval pool reserved by data/v5_mining/pool_split.json
                    (eval_pool_hashes, 18 transcripts). The adjudicated eval
                    file does not exist yet -> status "pending"; expected
                    layout documented so the assembler is forward-compatible.
T3  synthetic eval  existing data/valen_nano_v4/eval.jsonl (G4 leg A,
                    non-regression). The future context_relevance_v5 eval pool
                    (G4 leg B, saturation tripwire) is referenced as pending.
T4  transfer holdout placeholder spec only — an entire source family withheld
                    from v5 train; the concrete source is fixed once the v5
                    train manifest exists (§7.2 arm A/B convention recorded).

Usage:
    python3 scripts/assemble_v5_eval_v1.py
    python3 scripts/assemble_v5_eval_v1.py --out data/v5_eval/eval_manifest.json
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- paths ----

T1_MAIN = "data/real_context_eval_v1/eval.jsonl"
T1_SUPP = "data/real_context_eval_v1/drop_supp/eval.jsonl"
# results rows for the incumbent lora_v4 were scored on candidates.jsonl
# (superset of eval.jsonl); the gate checker joins index -> record_id ->
# eval label through this file.
T1_SCORE_INDEX = "data/real_context_eval_v1/candidates.jsonl"

T2_POOL_SPLIT = "data/v5_mining/pool_split.json"
T2_EXPECTED_EVAL = "data/real_context_eval_v5_ext/eval.jsonl"
T2_EXPECTED_EXCLUDED = "data/real_context_eval_v5_ext/excluded.jsonl"

T3_V4_EVAL = "data/valen_nano_v4/eval.jsonl"
T3_V5_SYNTH_EXPECTED = "data/valen_nano_v5/eval.jsonl"

T4_EXPECTED_EVAL = "data/v5_transfer_holdout/eval.jsonl"

INCUMBENT_RESULTS = {
    "lora_v4_frozen_main": "results/lora_real_candidates_v1.jsonl",
    "lora_v4_frozen_supp": "results/lora_drop_supp_v1.jsonl",
    "winnow_frozen_main": "results/winnow_real_candidates_full_v1.jsonl",
    "winnow_frozen_supp": "results/winnow_drop_supp_v1.jsonl",
    "kev_frozen_main": "results/kev_real_candidates_v1.jsonl",
    "kev_frozen_supp": "results/kev_drop_supp_v1.jsonl",
}

INSTRUCTIONS_SHA256 = (
    "4659848727be93d683a792b8566b97e82191f114b9aa1944a74c6df05f222b90")


# ------------------------------------------------------------- helpers ----

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_jsonl(path):
    """Return content-free stats for a valen-format eval.jsonl, or None."""
    path = Path(path)
    if not path.exists():
        return None
    records = 0
    groups = set()
    labels = {"true": 0, "false": 0, "null": 0}
    sources = {}
    families = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            records += 1
            rec = json.loads(line)
            groups.add(rec.get("group_id"))
            tgt = ((rec.get("targets") or {}).get("irrelevant") or {}).get(
                "probabilities") or {}
            t = tgt.get("true")
            if t is None:
                labels["null"] += 1
            elif t >= 0.5:
                labels["true"] += 1
            else:
                labels["false"] += 1
            meta = rec.get("meta") or {}
            src = meta.get("source_root") or meta.get("source_dataset")
            if src:
                sources[src] = sources.get(src, 0) + 1
            fam = meta.get("family")
            if fam:
                families[fam] = families.get(fam, 0) + 1
    return {
        "path": str(path),
        "records": records,
        "groups": len(groups),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "label_counts": labels,
        "per_source": sources,
        "per_family": families,
    }


def stat_pointer(path):
    """Hash/size for a file that must not be parsed (e.g. results jsonl)."""
    path = Path(path)
    if not path.exists():
        return None
    return {"path": str(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


# ------------------------------------------------------------ assemble ----

def build_manifest(root):
    root = Path(root)
    manifest = {
        "schema_version": "nanojev-v5-eval-assembly-v1",
        "builder": "scripts/assemble_v5_eval_v1.py",
        "design_doc": "docs/V5_DATA_DESIGN_V1.md §7.2/§9",
        "seed": 20261201,
        "content_free": True,
        "training_allowed": False,
        "evaluation_only": True,
        "question": {
            "qid": "irrelevant",
            "type": "noul",
            "semantics": "true == candidate segment certainly irrelevant == "
                         "drop; false == keep",
            "instructions_sha256": INSTRUCTIONS_SHA256,
        },
        "read_only_note": (
            "assembler copies no payloads; entries are pointers "
            "(path+sha256+counts). scoring joins results rows to labels via "
            "meta.record_id; see scripts/check_v5_gates_v1.py."),
        "tiers": {},
        "incumbent_baseline_files": {},
        "excluded": {
            "data/real_context_eval_v1/**": "frozen eval; never train",
            "data/real_context_eval_v5_ext/**": "v5-ext eval; never train",
            "data/v5_transfer_holdout/**": "transfer eval; never train",
            "data/jevbench_offline_bundle_v1": "evaluation_only bundle",
            "*_test.jsonl,*_ood.jsonl": "frozen splits reserved for gates",
            "results/**": "scorer receipts; never enter training",
        },
    }

    # -- T1: real-frozen --------------------------------------------------
    t1_files = []
    for rel in (T1_MAIN, T1_SUPP):
        st = stat_jsonl(root / rel)
        t1_files.append(st if st else {"path": rel, "missing": True})
    t1_ready = all(not f.get("missing") for f in t1_files)
    t1 = {
        "tier": "T1",
        "name": "real-frozen",
        "role": "fixed holdout, never trained on; primary acceptance "
                "instrument (G1, G2, G5, G6)",
        "status": "ready" if t1_ready else "missing",
        "training_allowed": False,
        "scored_order": "concatenation order for merged-590 metrics: "
                        "main eval.jsonl rows then drop_supp/eval.jsonl rows",
        "files": {
            "main": t1_files[0],
            "drop_supp": t1_files[1],
            "score_index": {
                "path": T1_SCORE_INDEX,
                "exists": (root / T1_SCORE_INDEX).exists(),
                "note": "incumbent-era results were scored on "
                        "candidates.jsonl (superset of eval.jsonl); join "
                        "row.i -> meta.record_id -> eval label",
            },
        },
        "expected_scored": 590,
        "labels_expected": {"keep": 534, "drop": 56},
        "incumbent_reference": {
            "scorer": "lora_v4 (nano_sft_text_v4_fp32/valen-head@Qwen3.5-0.8B)",
            "acc": 0.9453, "fp": 11, "fn": 21, "drop_recall": 0.625,
            "confident_fp_noul_ge_0_9": 7,
            "consensus_lora_and_winnow": {"fp": 0, "precision": 1.0,
                                          "recall": 0.1786},
            "source": "docs/REAL_CONTEXT_EVAL_RESULTS_V1.md",
        },
    }
    manifest["tiers"]["T1_real_frozen"] = t1

    # -- T2: mined-eval pool (real-v5-ext) --------------------------------
    pool_path = root / T2_POOL_SPLIT
    t2 = {
        "tier": "T2",
        "name": "real-v5-ext",
        "role": "mined eval extension: fresh-source check on mining + "
                "drop-positive coverage (informational for gates; reported "
                "alongside G1/G5 when present)",
        "training_allowed": False,
        "files": {
            "expected_eval": T2_EXPECTED_EVAL,
            "expected_excluded": T2_EXPECTED_EXCLUDED,
            "pool_split": T2_POOL_SPLIT,
        },
    }
    if pool_path.exists():
        pool = json.loads(pool_path.read_text())
        t2["pool_split"] = {
            "path": T2_POOL_SPLIT,
            "schema_version": pool.get("schema_version"),
            "rule": pool.get("rule"),
            "eval_pool_transcripts": len(pool.get("eval_pool_hashes") or []),
            "train_pool_transcripts": len(pool.get("train_pool_hashes") or []),
            "eval_v1_frozen": len(pool.get("eval_v1_frozen_hashes") or []),
            "eval_pool_hashes_sha256": hashlib.sha256(json.dumps(
                sorted(pool.get("eval_pool_hashes") or [])).encode()
            ).hexdigest(),
        }
        ext_eval = stat_jsonl(root / T2_EXPECTED_EVAL)
        if ext_eval:
            t2["status"] = "ready"
            t2["eval_file"] = ext_eval
        else:
            t2["status"] = "pending_mining_and_labeling"
            t2["pending_note"] = (
                "18 transcripts reserved in pool_split.json "
                "eval_pool_hashes are disjoint from the 60 eval_v1 "
                "transcripts and from train_pool. Expected artifact: "
                f"{T2_EXPECTED_EVAL} produced by the v5 miner + "
                "adjudication pass (target ~300-400 candidates per §7.2). "
                "Caveat §7.3.5: label_real_context_v1.py --finalize ignores "
                "--out-dir — always pass --eval-out explicitly.")
    else:
        t2["status"] = "pending_pool_split"
        t2["pending_note"] = (
            f"{T2_POOL_SPLIT} not found. Expected convention: the v5 mining "
            "pool split reserves ~18 transcripts (eval_pool_hashes) for "
            f"eval mining; the adjudicated eval lands at {T2_EXPECTED_EVAL}. "
            "Assembler tolerates the absence; rerun after the pool split "
            "is written.")
    manifest["tiers"]["T2_real_v5_ext"] = t2

    # -- T3: synthetic eval ------------------------------------------------
    v4_eval = stat_jsonl(root / T3_V4_EVAL)
    v5_synth = stat_jsonl(root / T3_V5_SYNTH_EXPECTED)
    t3 = {
        "tier": "T3",
        "name": "synthetic",
        "role": "family-wise discrimination + G4 non-regression / "
                "saturation tripwire",
        "training_allowed": False,
        "files": {
            "v4_eval": v4_eval if v4_eval else {"path": T3_V4_EVAL,
                                               "missing": True},
            "v5_synth_eval": v5_synth if v5_synth else {
                "path": T3_V5_SYNTH_EXPECTED, "missing": True,
                "note": "context_relevance_v5 eval/dev pools (gidx "
                        "disjoint from train, same mod-rule scheme as v4; "
                        "~1500 eval + ~200 dev per §7.2). Also acceptable: "
                        "data/context_relevance_v5_seed20261201/eval.jsonl "
                        "if the v5 emitters write a raw pool before "
                        "consolidation."},
        },
    }
    t3["status"] = ("ready" if v4_eval and v5_synth
                    else "partial" if v4_eval else "missing")
    t3["g4_legs"] = {
        "leg_a_nonregression": "acc on v4_eval >= 0.99 (lora_v4 = 1.0000)",
        "leg_b_tripwire": "acc on v5_synth_eval >= 0.99 AND at least one "
                          "borderline prediction; acc~1.0 with zero "
                          "borderline = memorization tripwire -> extend "
                          "the set, do not claim victory",
    }
    manifest["tiers"]["T3_synthetic"] = t3

    # -- T4: transfer holdout (placeholder spec) ---------------------------
    t4_eval = stat_jsonl(root / T4_EXPECTED_EVAL)
    t4 = {
        "tier": "T4",
        "name": "transfer-holdout",
        "role": "measures train->new-source gap (G3: seen-source acc - "
                "transfer acc <= 10pp; kev's 15-18pp gap is the anti-target)",
        "training_allowed": False,
        "files": {
            "expected_eval": T4_EXPECTED_EVAL,
            "eval_file": t4_eval if t4_eval else None,
        },
        "spec": {
            "rule": "an ENTIRE source family is withheld from v5 train and "
                    "mined for eval only (§7.2, §7.3.3: selected BEFORE "
                    "family emitters run; excluded from every "
                    "mining/generation input list)",
            "arm_convention": {
                "arm_A": "train on codex_sessions + SWE-Gym-family "
                         "trajectories; hold out claude_projects + "
                         "SWE-chat-family",
                "arm_B": "mirror of arm_A for a second measurement",
            },
            "withheld_source": None,
            "pending_decision": "concrete withheld family is fixed once the "
                                "v5 train manifest (data/valen_nano_v5/"
                                "manifest.json) exists; whichever mined/"
                                "public source family the train manifest "
                                "marks as absent becomes T4.",
            "target_size": "500-800 records (§7.2)",
        },
    }
    t4["status"] = "ready" if t4_eval else "pending_definition"
    manifest["tiers"]["T4_transfer_holdout"] = t4

    # -- incumbent baseline pointers (G1/G2/G6 read these) -----------------
    for name, rel in INCUMBENT_RESULTS.items():
        st = stat_pointer(root / rel)
        manifest["incumbent_baseline_files"][name] = (
            st if st else {"path": rel, "missing": True})

    # -- group disjointness assertion T1 vs T3 -----------------------------
    if v4_eval and t1_ready:
        t1_groups = set()
        for rel in (T1_MAIN, T1_SUPP):
            for line in open(root / rel):
                if line.strip():
                    t1_groups.add(json.loads(line).get("group_id"))
        t3_groups = set()
        for line in open(root / T3_V4_EVAL):
            if line.strip():
                t3_groups.add(json.loads(line).get("group_id"))
        manifest["group_disjointness"] = {
            "T1_vs_T3_v4_eval": not (t1_groups & t3_groups),
            "rule": "group_id never straddles tiers (§7.3.1); T2/T4 "
                    "disjointness is enforced at emit time because their "
                    "source pools are reserved before emitters run",
        }

    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/v5_eval/eval_manifest.json")
    args = ap.parse_args()

    manifest = build_manifest(args.repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {k: v["status"] for k, v in manifest["tiers"].items()}
    print(json.dumps({"event": "v5_eval_manifest_written",
                      "out": str(args.out), "tiers": summary}))


if __name__ == "__main__":
    main()
