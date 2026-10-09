#!/usr/bin/env python3
"""Score T9d-v3 two-arm run: corpus v3 test per-qtype accuracy + heldout_v1 metrics.

Reads only files; no model calls, no MPS. Emits results/t9d_v3_run_summary.json.
Amendment: research/nanojev_v2_t9d_v3_amendment_v1.json
Baselines: heldout acc 0.463768115942029 (results/heldout_v1_g4_baseline.json),
eng-test acc 0.44 atomic reference (W42/J-D5 reporting convention).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_BASELINE_ACC = 0.463768115942029
ENG_TEST_BASELINE_ACC = 0.44
ARMS = ("headonly", "full")
SEEDS = (17, 18, 19)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gold_key(qtype, gold):
    if qtype == "boolean":
        return "true" if gold else "false"
    if qtype == "score":
        return str(gold)
    return str(gold)


def score_test_predictions(run_dir):
    """Argmax accuracy overall and per question type from predictions_test.jsonl."""
    path = run_dir / "predictions_test.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_type = {}
    n_correct = 0
    for r in rows:
        probs, gold = r["student_probs"], r["gold_index"]
        pred = max(range(len(probs)), key=probs.__getitem__)
        correct = int(pred == gold)
        n_correct += correct
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += correct
    return {"questions": len(rows), "accuracy": n_correct / len(rows) if rows else None,
            "by_type": {t: {"n": b["n"], "accuracy": b["correct"] / b["n"]}
                        for t, b in sorted(by_type.items())}}


def score_heldout(receipt_path):
    """Accuracy / coverage@0.9 / mean confidence against engineering_heldout_v1 gold."""
    if not receipt_path.is_file():
        return None
    receipt = load(receipt_path)
    gold_by_state = {}
    for line in (ROOT / "research/engineering_heldout_v1/items.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold_by_state[row["id"]] = row["gold"]
    by_type, confs = {}, []
    n = n_correct = answered = confident_wrong = 0
    for state in receipt["states"]:
        gold_map = gold_by_state.get(state["id"], {})
        for qid, ans in state["answers"].items():
            probs = ans["probabilities"]
            pred = max(probs, key=probs.__getitem__)
            conf = probs[pred]
            correct = int(pred == gold_key(ans["type"], gold_map.get(qid)))
            n += 1
            n_correct += correct
            confs.append(conf)
            if conf >= 0.9:
                answered += 1
                confident_wrong += (1 - correct)
            b = by_type.setdefault(ans["type"], {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += correct
    return {"questions": n, "accuracy": n_correct / n if n else None,
            "coverage_at_0.9": answered / n if n else None, "answered_count": answered,
            "confident_wrong_count": confident_wrong,
            "mean_confidence": math.fsum(confs) / len(confs) if confs else None,
            "by_type": {t: {"n": b["n"], "accuracy": b["correct"] / b["n"]}
                        for t, b in sorted(by_type.items())}}


def main():
    per_run = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            run_dir = ROOT / f"checkpoints/domain_adaptation_v3_{arm}_seed{seed}"
            entry = {"run_dir": str(run_dir.relative_to(ROOT)), "status": "missing"}
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                s = load(summary_path)
                entry.update(status="complete", best_step=s["best_step"],
                             best_dev_target_ce=s["best_dev_target_ce"],
                             test_target_ce=(s["metrics_by_split"].get("test") or {}).get("target_ce"),
                             dev_target_ce=(s["metrics_by_split"].get("dev") or {}).get("target_ce"),
                             training_seconds=s.get("training_seconds"))
                entry["eng_test"] = score_test_predictions(run_dir)
            elif (run_dir / "config.json").exists():
                entry["status"] = "in_progress_or_incomplete"
            heldout = score_heldout(ROOT / f"results/heldout_v1_post_v3_{arm}_seed{seed}.json")
            if heldout is not None:
                entry["heldout_v1"] = heldout
                entry["heldout_acc_delta_vs_0.464"] = (
                    heldout["accuracy"] - HELDOUT_BASELINE_ACC if heldout["accuracy"] is not None else None)
            per_run[key] = entry

    arms = {}
    for arm in ARMS:
        runs = [per_run[f"{arm}_seed{s}"] for s in SEEDS]
        complete = [r for r in runs if r["status"] == "complete"]
        heldout_accs = [r["heldout_v1"]["accuracy"] for r in complete if r.get("heldout_v1")]
        eng_accs = [r["eng_test"]["accuracy"] for r in complete if r.get("eng_test") and r["eng_test"]["accuracy"] is not None]
        step0 = sum(1 for r in complete if r["best_step"] == 0)
        arms[arm] = {
            "completed_seeds": len(complete),
            "seeds_selecting_step0": step0,
            "heldout_acc_min": min(heldout_accs) if heldout_accs else None,
            "heldout_acc_mean": math.fsum(heldout_accs) / len(heldout_accs) if heldout_accs else None,
            "eng_test_acc_min": min(eng_accs) if eng_accs else None,
            "eng_test_acc_mean": math.fsum(eng_accs) / len(eng_accs) if eng_accs else None,
            "beats_heldout_baseline_min_rule": bool(heldout_accs) and min(heldout_accs) > HELDOUT_BASELINE_ACC,
            "beats_eng_test_baseline_mean_rule": bool(eng_accs) and (math.fsum(eng_accs) / len(eng_accs)) > ENG_TEST_BASELINE_ACC,
        }

    a, b = arms["headonly"], arms["full"]
    if a["heldout_acc_mean"] is None and b["heldout_acc_mean"] is None:
        verdict = "incomplete: no arm finished yet"
    else:
        winner = max(ARMS, key=lambda x: arms[x]["heldout_acc_mean"] if arms[x]["heldout_acc_mean"] is not None else -1)
        beats = [x for x in ARMS if arms[x]["beats_heldout_baseline_min_rule"]]
        verdict = (f"best arm by mean heldout acc: {winner}; "
                   f"arms beating 0.464 heldout baseline on min-across-seeds: {beats or 'none'}")

    receipt = {
        "schema_version": "nanojev-v2-t9d-v3-run-summary-v1",
        "amendment": "research/nanojev_v2_t9d_v3_amendment_v1.json",
        "amendment_sha256": "2f57ac65df4a24e3a2f9a74e5a44330664715ad7d3e0c554746e5a17c4e8f42a",
        "bound_protocol_sha256": "7d687325040159241d7c92df1103cbdf04f4e85f0e3654e8576c41f8aed02b1a",
        "baselines": {"heldout_v1_atomic_acc": HELDOUT_BASELINE_ACC,
                      "eng_test_atomic_acc_reference": ENG_TEST_BASELINE_ACC},
        "selection_rule": "dev target CE minimum, step 0 eligible, earliest-step tie-break (protocol v2 unchanged)",
        "per_run": per_run,
        "arms": arms,
        "verdict": verdict,
    }
    out = ROOT / "results/t9d_v3_run_summary.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out), "arms": arms, "verdict": verdict}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
