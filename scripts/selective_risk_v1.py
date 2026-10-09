#!/usr/bin/env python3
"""J-C selective-risk serving layer: coverage-risk measurement + operating points.

This is advisory measurement machinery, independent of the final model. It does
NOT change predictor defaults, does NOT fit temperatures, and does NOT claim
accuracy. Given a checkpoint and eval datasets it:

  1. runs (or loads cached) T=1.0 local predictions via DecisionPredictor,
  2. computes full coverage-risk curves over a confidence-threshold grid
     (coverage, selective accuracy/NLL/Brier/ECE-10 on the answered subset,
     confident errors at each threshold), per question type AND overall,
  3. counts protected errors (wrong argmax with confidence >= 0.9; the
     zero-tolerance metric from the abstention-gate protocol),
  4. fits per-type operating points (threshold achieving a target selective
     accuracy at maximum coverage) on the CALIBRATION split ONLY,
  5. measures those frozen thresholds read-once on heldout cohorts and reports
     transfer gaps honestly.

Split discipline (fail-closed):
  * the --calibration file is the ONLY fit source; every --eval file is
    measured, never fitted;
  * fit and each eval cohort must be disjoint in (row_id, qid) and in
    state_id — enforced by hash check before any fitting;
  * missing or empty cohort files abort the run (no partial report);
  * rows whose declared `split` field disagrees with the file's declared role
    abort the run.

Outputs: cached prediction payloads + a self-hashed JSON report under
results/selective_risk_v1/; optionally the declarative routing spec
research/selective_risk_layer_v1.json (--spec-out).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

REPORT_SCHEMA = "nanojev-selective-risk-report-v1"
SPEC_SCHEMA = "nanojev-selective-risk-layer-v1"
GRID_POINTS = 101                      # 0.00 .. 1.00 inclusive, step 0.01
TARGET_ACCURACIES = (0.70, 0.80, 0.90, 0.95)
PROTECTED_GATE = 0.9                   # zero-tolerance confident-error gate
MIN_QUESTIONS = 32                     # T9c minimums, reused per question type
MIN_STATES = 8
ECE_BINS = 10
NLL_FLOOR = 1e-12
QUESTION_TYPES = ("boolean", "choice", "score")


# ----------------------------------------------------------------- utilities

def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cohort file missing: {p}")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"cohort file is empty: {p}")
    return rows


def mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else None


def threshold_grid(points=GRID_POINTS):
    if type(points) is not int or points < 2:
        raise ValueError("grid must have >= 2 points")
    return [i / (points - 1) for i in range(points)]


# ------------------------------------------------------------ record building

def option_count(question, probs):
    """Recomputed from the visible row (same convention as the T9c machinery)."""
    if question["type"] == "boolean":
        return 2
    crit = question.get("criteria")
    if isinstance(crit, (dict, list)) and len(crit):
        return len(crit)
    return len(probs)


def answers_map(pred_payload):
    out = {}
    for state in pred_payload["states"]:
        for qid, answer in state["answers"].items():
            out[(state["id"], qid)] = answer
    return out


def build_records(rows, answers, cohort, split):
    """One record per (row, question). `answers`: {(row_id, qid): answer}."""
    records = []
    for row in rows:
        for qid, question in row["questions"].items():
            key = (row["id"], qid)
            if key not in answers:
                raise KeyError(f"missing prediction for {row['id']}:{qid}")
            probs = answers[key]["probabilities"]
            gold = row["gold_probs"][qid]
            pred = max(probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
            gold_arg = max(gold.items(), key=lambda kv: (kv[1], kv[0]))[0]
            records.append({
                "cohort": cohort,
                "split": split,
                "row_id": row["id"],
                "state_id": row.get("state_id") or row["id"],
                "source_group_id": row.get("metadata", {}).get("source_group_id"),
                "family_id": row.get("family_id"),
                "qid": qid,
                "question_type": question["type"],
                "group": f"{question['type']}:{option_count(question, probs)}",
                "probs": probs,
                "gold_probs": gold,
                "pred_argmax": pred,
                "gold_argmax": gold_arg,
                "confidence": max(probs.values()),
                "correct": pred == gold_arg,
            })
    return records


def records_identities(records):
    """Canonical per-question identities used by the isolation hash check."""
    return sorted(f"{r['row_id']}::{r['qid']}" for r in records)


def records_hash(records):
    h = hashlib.sha256()
    for ident in records_identities(records):
        h.update(ident.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_fit_eval_isolation(fit_records, eval_records, fit_name, eval_name):
    """Fail closed unless fit and eval cohorts are disjoint.

    Enforced by hash check on canonical (row_id, qid) identities and on
    state_id sets; any overlap raises before fitting/transfer is attempted.
    """
    fit_ids = set(records_identities(fit_records))
    eval_ids = set(records_identities(eval_records))
    shared = fit_ids & eval_ids
    if shared:
        raise ValueError(
            f"fit/eval isolation violated: {len(shared)} shared question "
            f"identities between {fit_name} and {eval_name} "
            f"(e.g. {sorted(shared)[:3]})")
    fit_states = {r["state_id"] for r in fit_records}
    eval_states = {r["state_id"] for r in eval_records}
    shared_states = fit_states & eval_states
    if shared_states:
        raise ValueError(
            f"fit/eval isolation violated: {len(shared_states)} shared "
            f"state_ids between {fit_name} and {eval_name} "
            f"(e.g. {sorted(shared_states)[:3]})")
    return {
        "fit_records_sha256": records_hash(fit_records),
        "eval_records_sha256": records_hash(eval_records),
        "fit_questions": len(fit_ids),
        "eval_questions": len(eval_ids),
        "shared_question_identities": 0,
        "shared_state_ids": 0,
    }


# ------------------------------------------------------------------ metrics

def subset_metrics(records):
    """Accuracy / NLL / Brier / ECE-10 on an answered subset."""
    n = len(records)
    if not n:
        return {"n_answered": 0, "selective_accuracy": None, "selective_risk": None,
                "nll": None, "brier": None, "ece_fixed_10": None}
    nll = brier = 0.0
    n_correct = 0
    conf_pairs = []
    for r in records:
        keys = sorted(set(r["probs"]) | set(r["gold_probs"]))
        pr = [r["probs"].get(k, 0.0) for k in keys]
        gp = [r["gold_probs"].get(k, 0.0) for k in keys]
        nll += -math.fsum(g * math.log(max(p, NLL_FLOOR)) for g, p in zip(gp, pr))
        brier += math.fsum((p - g) ** 2 for p, g in zip(pr, gp))
        n_correct += r["correct"]
        conf_pairs.append((r["confidence"], r["correct"]))
    ece = 0.0
    bins = [[] for _ in range(ECE_BINS)]
    for conf, ok in conf_pairs:
        bins[min(int(conf * ECE_BINS), ECE_BINS - 1)].append((conf, ok))
    for b in bins:
        if b:
            ece += (len(b) / n) * abs(mean(c for c, _ in b)
                                      - mean(1.0 if o else 0.0 for _, o in b))
    acc = n_correct / n
    return {"n_answered": n, "selective_accuracy": acc, "selective_risk": 1.0 - acc,
            "nll": nll / n, "brier": brier / n, "ece_fixed_10": ece}


def curve_point(records, threshold):
    answered = [r for r in records if r["confidence"] >= threshold]
    point = {"threshold": threshold, "n_total": len(records),
             "coverage": len(answered) / len(records) if records else None}
    point.update(subset_metrics(answered))
    point["confident_errors"] = sum(1 for r in answered if not r["correct"])
    return point


def coverage_risk_curve(records, grid=None):
    """Full curve over the grid; coverage is non-increasing in t by construction."""
    grid = threshold_grid() if grid is None else list(grid)
    if any(t < 0 or t > 1 for t in grid) or grid != sorted(grid):
        raise ValueError("threshold grid must be sorted values within [0, 1]")
    points = [curve_point(records, t) for t in grid]
    for prev, cur in zip(points, points[1:]):
        if cur["coverage"] > prev["coverage"] + 1e-12:
            raise AssertionError("coverage must be non-increasing in threshold")
    return points


def curves_by_type(records, grid=None):
    out = {"overall": coverage_risk_curve(records, grid)}
    for qtype in QUESTION_TYPES:
        subset = [r for r in records if r["question_type"] == qtype]
        out[qtype] = coverage_risk_curve(subset, grid)
    return out


def protected_errors(records, gate=PROTECTED_GATE):
    """Zero-tolerance metric: wrong argmax whose confidence clears the gate."""
    flagged = [r for r in records if not r["correct"] and r["confidence"] >= gate]
    by_type = {t: sum(1 for r in flagged if r["question_type"] == t)
               for t in QUESTION_TYPES}
    return {"gate": gate, "count": len(flagged), "by_type": by_type,
            "examples": sorted(f"{r['row_id']}:{r['qid']}"
                               f"(conf={r['confidence']:.3f})"
                               for r in flagged)[:20]}


# ------------------------------------------------------- operating points

def by_question_type(records):
    out = {t: [r for r in records if r["question_type"] == t]
           for t in QUESTION_TYPES}
    out["overall"] = list(records)
    return out


def fit_operating_point(records, target_accuracy, grid=None):
    """Threshold achieving >= target selective accuracy at maximum coverage.

    Chosen rule (declarative, no search hyperparameters): among grid points
    whose selective accuracy meets the target, take the one with maximum
    coverage; ties break to higher accuracy then to the HIGHER threshold —
    the tightest gate consistent with the fit-set coverage, which leaves the
    largest abstention margin for transfer to unseen data.
    If no grid point meets the target, report the best achievable point and
    mark the target unreachable — an honest negative, not a failure.
    """
    grid = threshold_grid() if grid is None else list(grid)
    points = coverage_risk_curve(records, grid)
    feasible = [p for p in points
                if p["selective_accuracy"] is not None
                and p["selective_accuracy"] >= target_accuracy]
    best_any = max(points, key=lambda p: (p["selective_accuracy"] or -1.0,
                                          p["coverage"]))
    if not feasible:
        return {"target_selective_accuracy": target_accuracy,
                "status": "unreachable_on_this_grid",
                "threshold": None, "coverage": None,
                "selective_accuracy": None,
                "best_achievable": {"threshold": best_any["threshold"],
                                    "coverage": best_any["coverage"],
                                    "selective_accuracy": best_any["selective_accuracy"]}}
    chosen = max(feasible, key=lambda p: (p["coverage"],
                                         p["selective_accuracy"],
                                         p["threshold"]))
    return {"target_selective_accuracy": target_accuracy,
            "status": "estimated",
            "threshold": chosen["threshold"],
            "coverage": chosen["coverage"],
            "selective_accuracy": chosen["selective_accuracy"],
            "n_answered": chosen["n_answered"],
            "confident_errors": chosen["confident_errors"]}


def fit_operating_points(records, targets=TARGET_ACCURACIES, grid=None,
                         min_questions=MIN_QUESTIONS, min_states=MIN_STATES):
    """Per-type operating points fitted on ONE records list (calibration only)."""
    out = {}
    for name, subset in sorted(by_question_type(records).items()):
        n_q = len(subset)
        n_s = len({r["state_id"] for r in subset})
        entry = {"fit_rows": n_q, "fit_states": n_s,
                 "minimum_questions": min_questions,
                 "minimum_states": min_states}
        if n_q >= min_questions and n_s >= min_states:
            entry["status"] = "estimated"
            entry["points"] = {
                f"{t:.2f}": fit_operating_point(subset, t, grid)
                for t in targets}
        else:
            entry["status"] = "unestimated_below_minimum"
            entry["points"] = {}
        out[name] = entry
    return out


def measure_transfer(fitted, eval_records, targets=TARGET_ACCURACIES, grid=None):
    """Read-once application of calibration-fitted thresholds to eval records.

    Returns per-type rows: fitted threshold + calibration coverage/accuracy
    beside measured coverage/accuracy on eval, and the honest transfer gap.
    """
    eval_by_type = by_question_type(eval_records)
    grid = threshold_grid() if grid is None else list(grid)
    table = {}
    for name, fit_entry in sorted(fitted.items()):
        subset = eval_by_type.get(name, [])
        row = {"eval_rows": len(subset),
               "eval_states": len({r["state_id"] for r in subset}),
               "fit_status": fit_entry["status"], "targets": {}}
        if not subset:
            row["status"] = "no_eval_rows"
        for key, point in fit_entry.get("points", {}).items():
            cell = {"fitted_threshold": point["threshold"],
                    "fit_coverage": point.get("coverage"),
                    "fit_selective_accuracy": point.get("selective_accuracy"),
                    "fit_status": point["status"],
                    "target": point["target_selective_accuracy"]}
            if point["status"] == "estimated" and subset:
                measured = curve_point(subset, point["threshold"])
                m_acc = measured["selective_accuracy"]
                cell.update(
                    measured_coverage=measured["coverage"],
                    measured_n_answered=measured["n_answered"],
                    measured_selective_accuracy=m_acc,
                    measured_confident_errors=measured["confident_errors"],
                    transfer_gap_accuracy=(m_acc - point["target_selective_accuracy"]
                                           if m_acc is not None else None),
                    target_met_on_eval=(m_acc is not None
                                        and m_acc >= point["target_selective_accuracy"]))
            row["targets"][key] = cell
        table[name] = row
    return table


# ------------------------------------------------------------- cohort loading

def load_cohort(path, expected_split=None):
    """Fail-closed cohort loader: file must exist, be non-empty, and every row's
    declared split must match expected_split when one is given."""
    rows = load_jsonl(path)
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "id" not in row or "state" not in row \
                or "questions" not in row or "gold_probs" not in row:
            raise ValueError(f"{path}: row {i} lacks id/state/questions/gold_probs")
        if expected_split is not None and row.get("split") != expected_split:
            raise ValueError(
                f"{path}: row {row.get('id')} declares split "
                f"{row.get('split')!r}, expected {expected_split!r}")
    return rows


def rows_to_payload(rows):
    return {"states": [{"id": r["id"], "state": r["state"],
                        "questions": r["questions"]} for r in rows]}


# ------------------------------------------------------------- prediction io

def predict_cohort(predictor, rows, batch_questions=8, shared_prefix=False,
                   states_per_call=32):
    """Chunked local prediction; returns a merged openjev-toy-inference-v1 payload."""
    from predict_toy_decisions import validate_request  # local import: torch-free path
    states = rows_to_payload(rows)["states"]
    merged = {"schema_version": "openjev-toy-inference-v1", "states": [],
              "execution": {"forward_passes": 0, "network_model_calls": 0,
                            "chunk_calls": 0}}
    for start in range(0, len(states), states_per_call):
        chunk = {"states": states[start:start + states_per_call]}
        validate_request(chunk)
        result = predictor.predict(chunk, batch_questions=batch_questions,
                                   shared_prefix=shared_prefix)
        merged["states"].extend(result["states"])
        ex = result["execution"]
        merged["execution"]["forward_passes"] += ex["forward_passes"]
        merged["execution"]["network_model_calls"] += ex["network_model_calls"]
        merged["execution"]["chunk_calls"] += 1
        merged["execution"].update(
            device=ex["device"], precision=ex["precision"],
            prefix_sharing=ex["prefix_sharing"],
            max_length=ex["max_length"])
        merged["checkpoint"] = result["checkpoint"]
    return merged


def pred_paths(pred_dir, arm, cohort):
    pred_dir = Path(pred_dir)
    return (pred_dir / f"{arm}__{cohort}_pred.json",
            pred_dir / f"{arm}__{cohort}_pred.meta.json")


def save_predictions(pred_dir, arm, cohort, payload, meta):
    pred_path, meta_path = pred_paths(pred_dir, arm, cohort)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return pred_path


def load_predictions(pred_dir, arm, cohort, dataset_path, checkpoint_dir):
    """Reuse cached predictions only when the meta receipt proves same data+model."""
    pred_path, meta_path = pred_paths(pred_dir, arm, cohort)
    if not pred_path.is_file() or not meta_path.is_file():
        return None
    meta = load_json(meta_path)
    expected = {"dataset_path": str(Path(dataset_path).resolve()),
                "dataset_sha256": sha256_file(dataset_path),
                "checkpoint_dir": str(Path(checkpoint_dir).resolve())}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"cached predictions {pred_path} do not match current "
                f"{key} ({meta.get(key)!r} != {value!r}); refusing reuse")
    payload = load_json(pred_path)
    if payload.get("execution", {}).get("network_model_calls") != 0:
        raise ValueError(f"{pred_path}: network_model_calls != 0")
    return payload


# ---------------------------------------------------------------- spec emit

def build_layer_spec(arm, checkpoint_dir, fit_cohort_name, fit_path,
                     fitted, fit_meta, transfer=None):
    """Declarative routing spec; advisory only, no serving change authorized."""
    transfer_status = {}
    for cohort, table in (transfer or {}).items():
        cells = [c for row in table.values() for c in row.get("targets", {}).values()
                 if c.get("fit_status") == "estimated"]
        met = sum(1 for c in cells if c.get("target_met_on_eval"))
        transfer_status[cohort] = {
            "targets_measured": len(cells),
            "targets_met": met,
            "verdict": ("thresholds_transfer" if cells and met == len(cells)
                        else "transfer_gaps_present" if cells
                        else "nothing_to_measure"),
            "receipt": "results/selective_risk_v1/ per-arm report JSON",
        }
    operating = {}
    for name, entry in sorted(fitted.items()):
        points = {}
        for key, p in entry.get("points", {}).items():
            if p["status"] == "estimated":
                points[key] = {
                    "status": "estimated",
                    "abstain_below": p["threshold"],
                    "expected_coverage_on_fit_split": p["coverage"],
                    "selective_accuracy_on_fit_split": p["selective_accuracy"],
                }
            else:
                points[key] = {
                    "status": "unreachable_on_fit_split",
                    "abstain_below": None,
                    "best_achievable_on_fit_split": p.get("best_achievable"),
                }
        operating[name] = {
            "status": entry["status"],
            "fit_rows": entry["fit_rows"], "fit_states": entry["fit_states"],
            "targets": points,
        }
    return {
        "schema_version": SPEC_SCHEMA,
        "layer_id": "selective_risk_v1",
        "status": "advisory_measurement_layer",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {"checkpoint_id": arm, "checkpoint_dir": checkpoint_dir},
        "confidence_definition": "max softmax probability over the question's candidate set at temperature 1.0",
        "threshold_source": {
            "fitted_on_split": fit_cohort_name,
            "fitted_on_file": str(fit_path),
            "fitted_on_sha256": sha256_file(fit_path),
            "fit_role": "calibration_fit_only",
            "never_fitted_on": ["heldout_v1", "heldout_v2", "test", "ood", "dev"],
            "isolation": "fit/eval disjointness on (row_id,qid) and state_id enforced by hash check at fit time",
        },
        "operating_points": operating,
        "abstention_semantics": {
            "gate_field": "abstain_below",
            "rule": "confidence < abstain_below -> answer.status='abstained', value=null, choice=null",
            "abstain_route": "caller_or_main_model",
            "never_silently_answered": True,
            "raw_probabilities_preserved": True,
            "authorizes_execution": False,
        },
        "protected_error_policy": {
            "definition": "wrong argmax with confidence >= 0.9",
            "limit": 0,
            "measured_counts_location": "results/selective_risk_v1/ reports",
        },
        "ood_detector": {
            "status": "placeholder_interface_not_implemented",
            "interface": {
                "input": "question record (state text, question type, candidate set)",
                "output": {"out_of_scope": "bool", "score": "float|null",
                           "reason": "string|null"},
                "routing": "out_of_scope=true routes to caller_or_main_model identically to confidence abstention",
            },
            "note": "existing scope guard (integrations/codex-skill assess_question_scope) is a lexical heuristic, not a learned OOD detector",
        },
        "non_goals": [
            "not a probability re-calibrator: probabilities are passed through unchanged at T=1.0",
            "not an accuracy claim: selective accuracies are measured quantities, not guarantees",
            "not a serving change: advisory/reporting layer like the existing abstention gate",
            "not fitted on heldout/test: thresholds transfer is measured, never tuned, on eval cohorts",
        ],
        "transfer_measured": transfer_status,
        "fit_meta": fit_meta,
        "network_model_calls": 0,
        "training_authorized": False,
        "deployment_authorized": False,
        "authorizes_execution": False,
    }


# -------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",
                        help="local checkpoint dir (required unless --skip-inference)")
    parser.add_argument("--arm", help="report arm name; default = checkpoint dir name")
    parser.add_argument("--calibration", required=True,
                        help="jsonl rows used as the ONLY fit source")
    parser.add_argument("--calibration-split-label", default="calibration",
                        help="expected row.split value in --calibration (default 'calibration'; '' disables)")
    parser.add_argument("--eval", action="append", default=[], metavar="NAME=PATH",
                        help="eval-only cohort, repeatable; never fitted on")
    parser.add_argument("--eval-split-label", default=None,
                        help="expected row.split for eval files (default: no check)")
    parser.add_argument("--pred-dir", default=str(ROOT / "results" / "selective_risk_v1" / "preds"))
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--spec-out", help="optional routing spec JSON path")
    parser.add_argument("--shared-prefix", action="store_true",
                        help="opt-in X3 packed forward path (off by default like the predictor)")
    parser.add_argument("--batch-questions", type=int, default=8)
    parser.add_argument("--states-per-call", type=int, default=32)
    parser.add_argument("--grid-points", type=int, default=GRID_POINTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--skip-inference", action="store_true",
                        help="reuse cached predictions only; fail if absent/mismatched")
    args = parser.parse_args(argv)

    t0 = time.time()
    arm = args.arm or Path(args.checkpoint_dir or "arm").name
    grid = threshold_grid(args.grid_points)

    # ---------------- load cohorts (fail-closed before any model work) -------
    cal_label = args.calibration_split_label or None
    cohorts = {"calibration": {
        "role": "fit", "path": Path(args.calibration),
        "rows": load_cohort(args.calibration, cal_label)}}
    for spec in args.eval:
        if "=" not in spec:
            raise ValueError(f"--eval expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or name in cohorts:
            raise ValueError(f"bad or duplicate eval cohort name {name!r}")
        cohorts[name] = {"role": "eval", "path": Path(path),
                         "rows": load_cohort(path, args.eval_split_label)}
    if not args.eval:
        raise ValueError("at least one --eval cohort is required")

    # ---------------- predictions (cached or fresh) --------------------------
    predictor = None
    if not args.skip_inference:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required without --skip-inference")
        from predict_toy_decisions import DecisionPredictor
        predictor = DecisionPredictor(
            args.checkpoint_dir, device_name=args.device, precision=args.precision)
    elif not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir still required to bind cached predictions")

    records = {}
    pred_meta = {}
    for name, cohort in cohorts.items():
        payload = None
        if args.skip_inference:
            payload = load_predictions(args.pred_dir, arm, name,
                                       cohort["path"], args.checkpoint_dir)
            if payload is None:
                raise FileNotFoundError(
                    f"no cached predictions for {arm}/{name}; run without "
                    f"--skip-inference first")
        else:
            payload = load_predictions(args.pred_dir, arm, name,
                                       cohort["path"], args.checkpoint_dir)
            if payload is None:
                payload = predict_cohort(
                    predictor, cohort["rows"],
                    batch_questions=args.batch_questions,
                    shared_prefix=args.shared_prefix,
                    states_per_call=args.states_per_call)
                meta = {"arm": arm, "cohort": name, "role": cohort["role"],
                        "dataset_path": str(cohort["path"].resolve()),
                        "dataset_sha256": sha256_file(cohort["path"]),
                        "dataset_rows": len(cohort["rows"]),
                        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                        "shared_prefix": bool(args.shared_prefix),
                        "batch_questions": args.batch_questions,
                        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "network_model_calls": 0}
                save_predictions(args.pred_dir, arm, name, payload, meta)
        amap = answers_map(payload)
        records[name] = build_records(cohort["rows"], amap, name,
                                      cohort["role"])
        pred_meta[name] = payload.get("execution", {})

    # ---------------- isolation guard (hash check before any fitting) --------
    isolation = {}
    for name, cohort in cohorts.items():
        if cohort["role"] != "eval":
            continue
        isolation[f"calibration|{name}"] = assert_fit_eval_isolation(
            records["calibration"], records[name], "calibration", name)

    # ---------------- curves + protected errors ------------------------------
    curves = {name: curves_by_type(recs, grid) for name, recs in records.items()}
    protected = {name: {
        "overall": protected_errors(recs),
        **{t: protected_errors([r for r in recs if r["question_type"] == t])
           for t in QUESTION_TYPES}}
        for name, recs in records.items()}

    # ---------------- fit on calibration ONLY, measure on eval ---------------
    fitted = fit_operating_points(records["calibration"], TARGET_ACCURACIES, grid)
    transfer = {name: measure_transfer(fitted, records[name], TARGET_ACCURACIES, grid)
                for name, cohort in cohorts.items() if cohort["role"] == "eval"}

    # ---------------- report --------------------------------------------------
    input_files = {"calibration": cohorts["calibration"]["path"],
                   **{f"eval_{n}": c["path"] for n, c in cohorts.items()
                      if c["role"] == "eval"},
                   "this_script": Path(__file__)}
    if args.checkpoint_dir:
        ckpt = Path(args.checkpoint_dir)
        input_files["checkpoint_weights"] = ckpt / "best.safetensors"
        input_files["checkpoint_config"] = ckpt / "config.json"

    report = {
        "schema_version": REPORT_SCHEMA,
        "task": "J-C selective-risk serving layer measurement",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "arm": arm,
        "checkpoint_dir": args.checkpoint_dir,
        "grid": {"points": args.grid_points, "min": 0.0, "max": 1.0},
        "targets": list(TARGET_ACCURACIES),
        "protected_gate": PROTECTED_GATE,
        "cohorts": {n: {"role": c["role"], "path": str(c["path"]),
                        "rows": len(c["rows"]),
                        "questions": len(records[n]),
                        "states": len({r["state_id"] for r in records[n]}),
                        "dataset_sha256": sha256_file(c["path"])}
                    for n, c in cohorts.items()},
        "isolation": isolation,
        "operating_points_fit_on_calibration": fitted,
        "transfer": transfer,
        "curves": curves,
        "protected_errors": protected,
        "prediction_execution": pred_meta,
        "input_hashes_sha256": {k: {"path": str(v), "sha256": sha256_file(v)}
                                for k, v in input_files.items() if Path(v).is_file()},
        "disclosures": [
            "thresholds fitted on the calibration cohort only; eval cohorts measured read-once",
            "transfer gaps are reported, not tuned away; an unreachable target is a finding",
            "curves use T=1.0 probabilities; no temperature or probability refit anywhere",
            "network_model_calls=0; all inference local",
        ],
        "network_model_calls": 0,
        "training_authorized": False,
        "deployment_authorized": False,
        "authorizes_execution": False,
    }
    blob = json.dumps(report, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(blob).hexdigest()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    spec_path = None
    if args.spec_out:
        fit_meta = {"fit_rows": len(records["calibration"]),
                    "fit_states": len({r["state_id"] for r in records["calibration"]}),
                    "isolation": isolation}
        spec = build_layer_spec(arm, args.checkpoint_dir, "calibration",
                                cohorts["calibration"]["path"], fitted, fit_meta,
                                transfer=transfer)
        spec_path = Path(args.spec_out)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    print(json.dumps({"report": str(out_path), "spec": str(spec_path) if spec_path else None,
                      "arm": arm, "receipt_sha256": report["receipt_sha256"],
                      "protected_errors_overall": {n: p["overall"]["count"]
                                                   for n, p in protected.items()},
                      "elapsed_s": report["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
