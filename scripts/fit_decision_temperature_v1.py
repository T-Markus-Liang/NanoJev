#!/usr/bin/env python3
"""Fit ONE scalar temperature on a CALIBRATION split only and report test/OOD metrics.

Why this exists
---------------
`predict_toy_decisions.DecisionPredictor.predict` already applies a temperature as
`softmax(candidate_scores / temperature)` and already accepts a `temperature` argument.
Nothing in the repository, however, *fits* that scalar for the local checkpoint
`checkpoints/local_atomic_seed17/variants/local_atomic_seed17` on the frozen
`dataset/games_v4/data/local_maze_v1` cohort, so the served `temperature.value` is stuck
at the harmless but uncalibrated default of 1.0.

This script fits that one scalar by minimising a proper loss (NLL, optionally Brier) on a
**calibration split only**, then reports accuracy / NLL / Brier / ECE / answer-rate on the
untouched test and OOD splits at temperature 1.0 versus the fitted value.

How a candidate temperature is applied without editing the service
------------------------------------------------------------------
Two independent routes are used and cross-checked:

1. Inference route (authoritative). Call the unmodified
   `DecisionPredictor.predict(payload, temperature=T)` and read its probabilities. The
   response's `temperature.value` field confirms the scalar that was applied. This is the
   real serving path; no service file is modified.
2. Arithmetic route (search). Because `softmax(z / T)` is invariant to additive shifts in
   `z`, the scores are recoverable from a temperature-1.0 probability vector as
   `z ~= log(p)` up to an unknown constant. Re-applying `softmax(log(p) / T)` is therefore
   exactly what the service computes. The grid search uses this route because it is free;
   every reported number is recomputed from route 1.

The two routes are compared explicitly (`service_path_verification` in the output JSON).

Hard guard: fitting never touches test or OOD
---------------------------------------------
Enforced in five layers, none of which is a comment:

  G1  `--fit-split`/`--also-fit` are rejected by name if they are `test` or `ood`,
      whichever splits happen to be evaluated.
  G2  Every split file's sha256 must equal the frozen `manifest.json` value, so
      `test.jsonl` cannot be renamed/copied into a calibration slot.
  G3  Every row used for fitting must carry `split == <fit split>`; a mismatched row aborts.
  G4  Fitting state_ids and source_group_ids must be disjoint from the test and OOD
      state_ids and source_group_ids (computed from the data, not trusted from prose).
  G5  The objective function re-asserts the split of each row it scores.

`--guard-self-test` proves G1 rejects `test` and `ood` and admits `calibration`.
`--verify-only` runs every guard and hash check without loading the model.

No network is used: `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set by the predictor, and
the script aborts if any call reports `network_model_calls != 0`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COHORT = REPO_ROOT / "dataset/games_v4/data/local_maze_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17"
DEFAULT_SURVEY = REPO_ROOT / "research/skill_abstention_survey_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/temperature_fit_v1.json"

ALL_SPLITS = ("train", "dev", "calibration", "test", "ood")
# Splits that may never be used to fit or select a temperature.
FIT_FORBIDDEN = frozenset({"test", "ood"})
# Splits that may legitimately carry the fit.
FIT_ALLOWED = frozenset({"train", "dev", "calibration"})
THRESHOLDS = (0.5, 0.7, 0.9)
NLL_FLOOR = 1e-12
# Floor used only when recovering scores from a temperature-1.0 probability vector.
LOGIT_RECOVERY_FLOOR = 1e-30
SHARED_GRID = {"min": 0.05, "max": 20.0, "points": 4000}
REFINE_STEPS = 200
ECE_BINS = 10


class FitSplitViolation(ValueError):
    """Raised when a fit is requested on a split reserved for evaluation."""


# --------------------------------------------------------------------------- basics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(row)
    return rows


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ------------------------------------------------------------------- the hard guard


def enforce_fit_split(name: str) -> str:
    """G1: reject a fit on any split reserved for evaluation."""
    if name in FIT_FORBIDDEN:
        raise FitSplitViolation(
            f"refusing to fit a temperature on split {name!r}: "
            f"{sorted(FIT_FORBIDDEN)} are evaluation-only and must stay untouched"
        )
    if name not in FIT_ALLOWED:
        raise FitSplitViolation(
            f"unknown fit split {name!r}; allowed fit splits are {sorted(FIT_ALLOWED)}"
        )
    return name


def guard_self_test() -> dict:
    """Prove that the name-level guard both rejects and admits, at runtime."""
    rejected, admitted = {}, {}
    for name in sorted(FIT_FORBIDDEN):
        try:
            enforce_fit_split(name)
        except FitSplitViolation:
            rejected[name] = "rejected"
        else:
            rejected[name] = "NOT_REJECTED"
    for name in sorted(FIT_ALLOWED):
        try:
            enforce_fit_split(name)
        except FitSplitViolation:
            admitted[name] = "NOT_ADMITTED"
        else:
            admitted[name] = "admitted"
    ok = all(v == "rejected" for v in rejected.values()) and all(
        v == "admitted" for v in admitted.values()
    )
    return {"forbidden_splits": rejected, "allowed_splits": admitted, "passed": ok}


# ------------------------------------------------------------------------ split book


class Cohort:
    """Loads the frozen cohort, pins every file hash, and computes real disjointness."""

    def __init__(self, cohort_dir: Path, manifest_name: str = "manifest.json"):
        self.directory = Path(cohort_dir).resolve()
        self.manifest_path = self.directory / manifest_name
        if not self.manifest_path.is_file():
            raise ValueError(f"missing manifest: {self.manifest_path}")
        self.manifest = read_json(self.manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.rows, self.files = {}, {}

        for split in ALL_SPLITS:
            path = self.directory / f"{split}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            actual = sha256_file(path)
            declared = (
                self.manifest.get("outputs", {}).get(split, {}).get("sha256")
            )
            self.rows[split] = rows
            self.files[split] = {
                "path": str(path.relative_to(REPO_ROOT))
                if path.is_relative_to(REPO_ROOT)
                else str(path),
                "sha256": actual,
                "sha256_declared_in_manifest": declared,
                "manifest_hash_match": declared == actual,
                "rows": len(rows),
                "questions": sum(len(r["questions"]) for r in rows),
                "source_groups": len(self.source_groups(split)),
                "state_ids": len(self.state_ids(split)),
                "true_prevalence": mean(
                    1.0 if r["gold"][q] else 0.0 for r in rows for q in r["questions"]
                ),
            }

    def state_ids(self, split):
        return {row["state_id"] for row in self.rows[split]}

    def source_groups(self, split):
        return {
            row.get("metadata", {}).get("source_group_id")
            for row in self.rows[split]
        } - {None}

    def require_pinned(self, splits):
        """G2: every split about to be read must match its frozen manifest hash."""
        bad = [s for s in splits if not self.files[s]["manifest_hash_match"]]
        if bad:
            raise ValueError(
                "refusing to proceed: split file hash does not match the frozen manifest for "
                + ", ".join(
                    f"{s} (file={self.files[s]['sha256'][:16]}..., "
                    f"manifest={str(self.files[s]['sha256_declared_in_manifest'])[:16]}...)"
                    for s in bad
                )
            )

    def disjointness(self, pairs):
        out = {}
        for left, right in pairs:
            state_overlap = self.state_ids(left) & self.state_ids(right)
            group_overlap = self.source_groups(left) & self.source_groups(right)
            out[f"{left}|{right}"] = {
                "state_id_overlap": len(state_overlap),
                "source_group_overlap": len(group_overlap),
            }
        return out


# ------------------------------------------------------------------ temperature math


def softmax(scores, temperature):
    if not (isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
            and math.isfinite(temperature) and temperature > 0):
        raise ValueError("temperature must be a finite positive number")
    top = max(scores)
    exps = [math.exp((value - top) / temperature) for value in scores]
    total = math.fsum(exps)
    if not total > 0 or not math.isfinite(total):
        raise ValueError("degenerate softmax")
    return [value / total for value in exps]


def recover_scores(probabilities):
    """Scores up to an additive constant; exact for temperature scaling."""
    return [math.log(max(p, LOGIT_RECOVERY_FLOOR)) for p in probabilities]


def rescale(probabilities, temperature, floor_hits=None):
    """Apply a temperature to a temperature-1.0 probability vector."""
    if floor_hits is not None:
        floor_hits[0] += sum(1 for p in probabilities if p < LOGIT_RECOVERY_FLOOR)
    return softmax(recover_scores(probabilities), temperature)


# ------------------------------------------------------------------------- inference


def build_payload(rows):
    return {
        "states": [
            {"id": row["id"], "state": row["state"], "questions": row["questions"]}
            for row in rows
        ]
    }


def load_engine(checkpoint_dir: Path, device: str, batch_questions: int, max_length=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from predict_toy_decisions import DecisionPredictor  # local import: torch is heavy

    return DecisionPredictor(
        str(checkpoint_dir),
        max_length=max_length,
        device_name=device,
        precision="fp32",
    )


def run_inference(engine, rows, temperature, batch_questions):
    result = engine.predict(
        build_payload(rows), batch_questions=batch_questions, temperature=temperature
    )
    execution = result["execution"]
    if execution.get("network_model_calls") != 0:
        raise RuntimeError("aborting: a model call reported network_model_calls != 0")
    answers = {}
    for state in result["states"]:
        for qid, answer in state["answers"].items():
            answers[(state["id"], qid)] = answer
    if len(answers) != sum(len(r["questions"]) for r in rows):
        raise RuntimeError("incomplete inference response")
    return answers, result


def binary_records(rows, answers):
    """Flatten (state, question) pairs into binary records with a fixed candidate order."""
    records = []
    for row in rows:
        for qid, question in row["questions"].items():
            if question.get("type") != "boolean":
                raise ValueError(
                    f"{row['id']}:{qid} is type {question.get('type')!r}; this cohort "
                    "analysis is Boolean-only by construction"
                )
            answer = answers[(row["id"], qid)]
            probs = answer["probabilities"]
            if set(probs) != {"false", "true"}:
                raise ValueError(f"{row['id']}:{qid} has unexpected candidates {sorted(probs)}")
            vector = [float(probs["false"]), float(probs["true"])]
            if abs(math.fsum(vector) - 1.0) > 1e-5:
                raise ValueError(f"{row['id']}:{qid} probabilities do not sum to one")
            records.append(
                {
                    "state_id": row["id"],
                    "qid": qid,
                    "split": row["split"],
                    "probs": vector,
                    "gold": 1 if row["gold"][qid] else 0,
                }
            )
    return records


# --------------------------------------------------------------------------- metrics


def binary_metrics(records):
    """Accuracy, NLL, Brier, ECE and threshold operating points for Boolean questions."""
    if not records:
        return {"questions": 0, "states": 0}
    p = [r["probs"][1] for r in records]
    y = [r["gold"] for r in records]
    n = len(records)

    correct = sum(1 for pi, yi in zip(p, y) if (pi >= 0.5) == bool(yi))
    nll = math.fsum(
        -math.log(max(pi if yi else 1.0 - pi, NLL_FLOOR)) for pi, yi in zip(p, y)
    ) / n
    brier_scalar = math.fsum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    ece_fixed, mce_fixed, bins_fixed = ece(p, y, ECE_BINS, adaptive=False)
    ece_adaptive, mce_adaptive, bins_adaptive = ece(p, y, ECE_BINS, adaptive=True)

    confidences = [max(pi, 1.0 - pi) for pi in p]
    operating = {}
    for threshold in THRESHOLDS:
        answered = [
            (pi, yi) for pi, yi, c in zip(p, y, confidences) if c >= threshold
        ]
        key = f"{threshold:.1f}"
        operating[key] = {
            "answer_rate": len(answered) / n,
            "answered": len(answered),
            "accuracy_on_answered": (
                mean(1.0 if (pi >= 0.5) == bool(yi) else 0.0 for pi, yi in answered)
                if answered
                else None
            ),
        }

    return {
        "questions": n,
        "states": len({r["state_id"] for r in records}),
        "accuracy": correct / n,
        "nll": nll,
        "nll_probability_floor": NLL_FLOOR,
        "brier_scalar_bernoulli": brier_scalar,
        "brier_two_class_sum": 2.0 * brier_scalar,
        "ece_fixed_width_10": ece_fixed,
        "ece_adaptive_width_10": ece_adaptive,
        "mce_fixed_width_10": mce_fixed,
        "ece_bins_fixed": bins_fixed,
        "confidence_min": min(confidences),
        "confidence_mean": mean(confidences),
        "confidence_median": percentile(confidences, 0.5),
        "confidence_p90": percentile(confidences, 0.9),
        "confidence_max": max(confidences),
        "operating_points": operating,
    }


def ece(p, y, bins, adaptive):
    """Expected/Maximum Calibration Error on the positive class p(true)."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    if adaptive:
        groups = [
            order[i * len(order) // bins:(i + 1) * len(order) // bins] for i in range(bins)
        ]
        edges = None
    else:
        groups = [
            [i for i in order if min(int(p[i] * bins), bins - 1) == b]
            for b in range(bins)
        ]
        edges = [[b / bins, (b + 1) / bins] for b in range(bins)]

    total = len(p)
    value = 0.0
    worst = 0.0
    detail = []
    for index, group in enumerate(groups):
        if not group:
            detail.append({"bin": index, "count": 0})
            continue
        confidence = mean(p[i] for i in group)
        frequency = mean(y[i] for i in group)
        gap = abs(confidence - frequency)
        weight = len(group) / total
        value += weight * gap
        worst = max(worst, gap)
        entry = {
            "bin": index,
            "count": len(group),
            "mean_confidence": confidence,
            "empirical_true_rate": frequency,
            "gap": gap,
        }
        if edges:
            entry["range"] = edges[index]
        detail.append(entry)
    return value, worst, detail


def metrics_at_temperature(records, temperature, floor_hits=None):
    rescaled = [
        {**record, "probs": rescale(record["probs"], temperature, floor_hits)}
        for record in records
    ]
    return binary_metrics(rescaled)


# ------------------------------------------------------------------------- the fit


def fit_temperature(records, objective="nll", fit_split="calibration"):
    """Fit one scalar on the supplied records only; G5 re-checks every row each evaluation."""
    if not records:
        raise FitSplitViolation("no fitting records supplied")
    scores = [recover_scores(r["probs"]) for r in records]
    golds = [r["gold"] for r in records]
    splits = sorted({r["split"] for r in records})
    if splits != [fit_split]:
        raise FitSplitViolation(
            f"fitting set must be exactly the {fit_split!r} split, found {splits}"
        )
    for record in records:  # G5: explicit, re-evaluated per row
        if record["split"] != fit_split:
            raise FitSplitViolation(f"row outside the fit split reached the objective: {record}")

    def loss(temperature):
        total = 0.0
        for score, gold in zip(scores, golds):
            probabilities = softmax(score, temperature)
            if objective == "nll":
                total += -math.log(max(probabilities[gold], NLL_FLOOR))
            elif objective == "brier":
                total += math.fsum(
                    (value - (1.0 if index == gold else 0.0)) ** 2
                    for index, value in enumerate(probabilities)
                )
            else:
                raise ValueError(f"unknown objective {objective!r}")
        return total / len(scores)

    low, high, points = SHARED_GRID["min"], SHARED_GRID["max"], SHARED_GRID["points"]
    log_low, log_high = math.log(low), math.log(high)
    grid = [
        math.exp(log_low + (log_high - log_low) * i / (points - 1)) for i in range(points)
    ]
    grid[0], grid[-1] = low, high
    if objective == "nll":
        # Guarantee T = 1.0 is on the grid even if the spacing misses it.
        grid[min(range(points), key=lambda i: abs(math.log(grid[i])))] = 1.0
    values = [loss(t) for t in grid]
    best = min(range(points), key=lambda i: (values[i], abs(math.log(grid[i]))))

    # Deterministic golden-section refinement in log space inside the bracketing interval.
    left = math.log(grid[max(best - 1, 0)])
    right = math.log(grid[min(best + 1, points - 1)])
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)
    fc, fd = loss(math.exp(c)), loss(math.exp(d))
    for _ in range(REFINE_STEPS):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - inverse_phi * (right - left)
            fc = loss(math.exp(c))
        else:
            left, c, fc = c, d, fd
            d = left + inverse_phi * (right - left)
            fd = loss(math.exp(d))
    refined = math.exp((left + right) / 2.0)

    at_boundary = best in (0, points - 1)
    return {
        "objective": objective,
        "fitted_temperature": refined,
        "grid_optimum": grid[best],
        "grid_loss_at_optimum": values[best],
        "refined_loss": loss(refined),
        "loss_at_temperature_1": loss(1.0),
        "at_grid_boundary": at_boundary,
        "grid": {"min": low, "max": high, "points": points, "spacing": "log"},
        "refinement": {"method": "golden_section_in_log_space", "steps": REFINE_STEPS},
        "questions_used_to_fit": len(scores),
        "states_used_to_fit": len({r["state_id"] for r in records}),
        "fit_split": fit_split,
        "direction": (
            "sharpen (raises every confidence)" if refined < 1.0
            else "soften (lowers every confidence)" if refined > 1.0
            else "identity"
        ),
    }


def metric_delta(before, after):
    keys = ["accuracy", "nll", "brier_scalar_bernoulli", "brier_two_class_sum",
            "ece_fixed_width_10", "ece_adaptive_width_10", "mce_fixed_width_10",
            "confidence_mean", "confidence_max"]
    out = {
        key: {"temperature_1.0": before.get(key), "temperature_fitted": after.get(key),
              "delta": (after[key] - before[key])
              if before.get(key) is not None and after.get(key) is not None else None}
        for key in keys
    }
    for threshold in THRESHOLDS:
        key = f"{threshold:.1f}"
        a, b = before["operating_points"][key], after["operating_points"][key]
        out[f"answer_rate_at_{key}"] = {
            "temperature_1.0": a["answer_rate"],
            "temperature_fitted": b["answer_rate"],
            "delta": b["answer_rate"] - a["answer_rate"],
        }
        out[f"accuracy_on_answered_at_{key}"] = {
            "temperature_1.0": a["accuracy_on_answered"],
            "temperature_fitted": b["accuracy_on_answered"],
            "delta": (b["accuracy_on_answered"] - a["accuracy_on_answered"])
            if a["accuracy_on_answered"] is not None
            and b["accuracy_on_answered"] is not None else None,
        }
    return out


def bootstrap_delta(records_at_one, records_at_fitted, samples, seed):
    """State-clustered bootstrap of (fitted - 1.0) metric deltas on a held-out split.

    Deterministic given `seed`. Clusters are `state_id`, matching the repository's existing
    evaluators, because the four questions of one maze share a state and are correlated.
    """
    if samples <= 0:
        return {"samples": 0, "status": "disabled"}
    clusters = {}
    for raw, fitted in zip(records_at_one, records_at_fitted):
        clusters.setdefault(raw["state_id"], []).append((raw, fitted))
    groups = list(clusters.values())
    if len(groups) < 2:
        return {"samples": samples, "status": "insufficient_clusters"}

    rng = random.Random(seed)
    collected = {}

    def record(key, value):
        if value is not None and math.isfinite(value):
            collected.setdefault(key, []).append(value)

    for _ in range(samples):
        draw = [pair for _ in groups for pair in rng.choice(groups)]
        base = binary_metrics([raw for raw, _ in draw])
        fit = binary_metrics([fitted for _, fitted in draw])
        for key in ("accuracy", "nll", "brier_scalar_bernoulli", "ece_fixed_width_10",
                    "ece_adaptive_width_10", "mce_fixed_width_10"):
            record(key, fit[key] - base[key])
        for threshold in THRESHOLDS:
            name = f"{threshold:.1f}"
            record(f"answer_rate_at_{name}",
                   fit["operating_points"][name]["answer_rate"]
                   - base["operating_points"][name]["answer_rate"])
            a = base["operating_points"][name]["accuracy_on_answered"]
            b = fit["operating_points"][name]["accuracy_on_answered"]
            record(f"accuracy_on_answered_at_{name}", b - a if a is not None and b is not None
                   else None)

    report = {}
    for key, values in collected.items():
        low, high = percentile(values, 0.025), percentile(values, 0.975)
        report[key] = {
            "mean_delta": mean(values),
            "ci95_low": low,
            "ci95_high": high,
            "excludes_zero": (low > 0) or (high < 0),
            "resamples": len(values),
        }
    return {"samples": samples, "seed": seed, "clusters": len(groups),
            "unit": "state_id", "deltas": report}


# ------------------------------------------------- engineering-survey post-hoc check


def survey_probe(engine, survey_path: Path, temperature, batch_questions):
    """Run the frozen 13-question engineering survey. Never used to fit anything."""
    survey = read_json(survey_path)
    payload = survey["request"]
    answers, result = run_inference(engine, [
        {"id": s["id"], "state": s["state"], "questions": s["questions"]}
        for s in payload["states"]
    ], temperature, batch_questions)
    expected = survey.get("expected_answers", {})
    rows = []
    for state in payload["states"]:
        for qid, question in state["questions"].items():
            answer = answers[(state["id"], qid)]
            probabilities = answer["probabilities"]
            chosen = max(probabilities, key=probabilities.__getitem__)
            confidence = max(probabilities.values())
            gold = expected.get(qid)
            rows.append(
                {
                    "state_id": state["id"],
                    "qid": qid,
                    "type": question["type"],
                    "confidence": confidence,
                    "chosen": chosen,
                    "expected": gold,
                    "correct": None if gold is None else (chosen == gold),
                    "temperature_to_reach": {
                        f"{threshold:.1f}": temperature_for_confidence(
                            list(probabilities.values()), threshold
                        )
                        for threshold in THRESHOLDS
                    },
                }
            )
    return rows, result


def temperature_for_confidence(probabilities, target):
    """The temperature at which max(probabilities) would exactly equal `target`.

    Because scalar temperature scaling is strictly monotone, this inverse is exact: a
    question clears a threshold `t` if and only if `T <= temperature_for_confidence(p, t)`.
    Returns None when no positive temperature can reach the target (e.g. a threshold below
    the uniform mass 1/k).
    """
    def at(temperature):
        return max(softmax(recover_scores(probabilities), temperature))

    low, high = 1e-4, 1e4
    if at(low) < target or at(high) > target:
        return None
    for _ in range(200):
        middle = math.sqrt(low * high)
        if at(middle) >= target:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def survey_metrics(rows):
    confidences = [row["confidence"] for row in rows]
    scored = [row for row in rows if row["correct"] is not None]
    operating = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.1f}"
        answered = [row for row in rows if row["confidence"] >= threshold]
        answered_scored = [row for row in answered if row["correct"] is not None]
        operating[key] = {
            "answered": len(answered),
            "questions": len(rows),
            "answer_rate": len(answered) / len(rows),
            "answered_fixed_answer_subset": len(answered_scored),
            "fixed_answer_subset": len(scored),
            "accuracy_on_answered_fixed_subset": (
                mean(1.0 if row["correct"] else 0.0 for row in answered_scored)
                if answered_scored else None
            ),
        }
    return {
        "questions": len(rows),
        "fixed_answer_questions": len(scored),
        "accuracy_on_fixed_subset_at_argmax": (
            mean(1.0 if row["correct"] else 0.0 for row in scored) if scored else None
        ),
        "confidence_min": min(confidences),
        "confidence_mean": mean(confidences),
        "confidence_median": percentile(confidences, 0.5),
        "confidence_max": max(confidences),
        "operating_points": operating,
        # Each question clears threshold t exactly when T <= its own inverse. So the largest
        # temperature that still answers every question is the smallest per-question inverse,
        # and the largest temperature that answers any question is the largest inverse.
        "temperature_ceiling_to_answer_all_questions": {
            f"{threshold:.1f}": (
                min(values) if (values := [
                    row["temperature_to_reach"][f"{threshold:.1f}"] for row in rows
                    if row["temperature_to_reach"][f"{threshold:.1f}"] is not None
                ]) else None
            )
            for threshold in THRESHOLDS
        },
        "temperature_ceiling_for_most_confident_question": {
            f"{threshold:.1f}": (
                max(values) if (values := [
                    row["temperature_to_reach"][f"{threshold:.1f}"] for row in rows
                    if row["temperature_to_reach"][f"{threshold:.1f}"] is not None
                ]) else None
            )
            for threshold in THRESHOLDS
        },
    }


# ---------------------------------------------------------------------------- main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT,
                        help="frozen local_maze_v1 cohort directory containing manifest.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT,
                        help="local checkpoint directory (no download, no network)")
    parser.add_argument("--fit-split", default="calibration",
                        help="split used to fit the single scalar (default: calibration); "
                             "test and ood are refused")
    parser.add_argument("--also-fit", default="dev",
                        help="comma-separated extra NON-test splits fitted for robustness "
                             "only (default: dev); test and ood are refused")
    parser.add_argument("--eval-splits", default="test,ood",
                        help="evaluation splits reported at temperature 1.0 vs fitted "
                             "(default: test,ood)")
    parser.add_argument("--fit-objective", choices=["nll", "brier"], default="nll",
                        help="proper loss minimised on the fit split (default: nll)")
    parser.add_argument("--batch-questions", type=int, default=64,
                        help="complete questions per forward pass; 0 = all in one forward")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda[:index]")
    parser.add_argument("--max-length", type=int, default=None,
                        help="default: the checkpoint's own training max_length")
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY,
                        help="frozen engineering-question survey probed post hoc with the "
                             "fitted temperature (never used for fitting)")
    parser.add_argument("--skip-survey", action="store_true",
                        help="skip the post-hoc engineering-survey probe")
    parser.add_argument("--bootstrap-samples", type=int, default=2000,
                        help="state-clustered bootstrap resamples for held-out metric "
                             "deltas; 0 disables (default: 2000)")
    parser.add_argument("--bootstrap-seed", type=int, default=17,
                        help="seed for the deterministic bootstrap (default: 17)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSON output path")
    parser.add_argument("--verify-only", action="store_true",
                        help="run guards, hash pins and disjointness checks without "
                             "loading the model")
    parser.add_argument("--guard-self-test", action="store_true",
                        help="assert the fit-split guard rejects test/ood and admits "
                             "calibration, then exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.guard_self_test:
        report = guard_self_test()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    # G1 first: refuse a forbidden fit before any file is even read.
    fit_split = enforce_fit_split(args.fit_split)
    also_fit = [enforce_fit_split(name.strip())
                for name in args.also_fit.split(",") if name.strip()]
    eval_splits = [name.strip() for name in args.eval_splits.split(",") if name.strip()]
    for name in eval_splits:
        if name not in ALL_SPLITS:
            raise ValueError(f"unknown evaluation split {name!r}")

    cohort = Cohort(args.cohort_dir)
    needed = sorted({fit_split, *also_fit, *eval_splits, "test", "ood"})
    missing = [name for name in needed if name not in cohort.rows]
    if missing:
        raise ValueError(f"cohort is missing required splits: {missing}")
    # G2: hash pins, including test/ood even when they are not evaluated.
    cohort.require_pinned(needed)

    fit_records = {}
    raw_records = {}
    inference = {}
    if not args.verify_only:
        engine = load_engine(args.checkpoint_dir, args.device, args.batch_questions,
                             args.max_length)
    for split in needed:
        if args.verify_only:
            continue
        answers, result = run_inference(engine, cohort.rows[split], 1.0,
                                        args.batch_questions)
        records = binary_records(cohort.rows[split], answers)
        raw_records[split] = records
        inference[split] = {
            "temperature": result["temperature"],
            "execution": result["execution"],
        }
        if split in (fit_split, *also_fit):
            fit_records[split] = records

    # G3 + G4: prove the fitting data is the fit split and is disjoint from test/ood.
    disjointness = cohort.disjointness([
        ("test", "calibration"), ("test", "dev"), ("test", "train"),
        ("ood", "calibration"), ("ood", "dev"), ("ood", "train"),
        ("test", "ood"), ("calibration", "dev"),
    ])
    for split in (fit_split, *also_fit):
        for evaluation in ("test", "ood"):
            overlap = disjointness[f"{evaluation}|{split}"] if f"{evaluation}|{split}" in disjointness \
                else disjointness[f"{split}|{evaluation}"]
            if overlap["state_id_overlap"] or overlap["source_group_overlap"]:
                raise FitSplitViolation(
                    f"refusing to fit: split {split!r} shares "
                    f"{overlap['state_id_overlap']} state_ids / "
                    f"{overlap['source_group_overlap']} source groups with {evaluation!r}"
                )
    if not args.verify_only:
        for split, records in fit_records.items():
            wrong = {r["split"] for r in records} - {split}
            if wrong:
                raise FitSplitViolation(f"fitting rows carry foreign splits: {sorted(wrong)}")

    if args.verify_only:
        report = {
            "schema_version": "nanojev-temperature-fit-v1",
            "mode": "verify-only",
            "guard_self_test": guard_self_test(),
            "fit_split": fit_split,
            "also_fit": also_fit,
            "eval_splits": eval_splits,
            "splits": cohort.files,
            "disjointness": disjointness,
            "manifest": {
                "path": str(cohort.manifest_path.relative_to(REPO_ROOT)),
                "sha256": cohort.manifest_sha256,
                "declared_checks": cohort.manifest.get("checks"),
            },
        }
        print(json.dumps(report, indent=2))
        return 0

    # ---------------------------------------------------------------- fit
    fits = {}
    for split, records in fit_records.items():
        fits[split] = {
            objective: fit_temperature(records, objective, split)
            for objective in ("nll", "brier")
        }
    primary = fits[fit_split][args.fit_objective]
    fitted_temperature = primary["fitted_temperature"]

    # Recompute every reported number from a real service-path call at the fitted value.
    metrics = {}
    verification = {}
    for split in needed:
        answers_fit, result_fit = run_inference(engine, cohort.rows[split],
                                                fitted_temperature, args.batch_questions)
        records_raw = raw_records[split]
        records_fit = binary_records(cohort.rows[split], answers_fit)
        floor_hits = [0]
        arithmetic = [
            {**record, "probs": rescale(record["probs"], fitted_temperature, floor_hits)}
            for record in records_raw
        ]
        worst = max(
            abs(a["probs"][i] - b["probs"][i])
            for a, b in zip(arithmetic, records_fit)
            for i in (0, 1)
        )
        verification[split] = {
            "routes_compared": ["arithmetic_rescale_of_temperature_1.0_run",
                                "direct_predict_temperature_fitted"],
            "temperature": fitted_temperature,
            "max_abs_probability_difference": worst,
            "agrees_within_probability_sum_tolerance": worst <= 1e-5,
        }
        metrics[split] = {
            "temperature_1.0": binary_metrics(records_raw),
            f"temperature_{fitted_temperature:.6f}": binary_metrics(records_fit),
            "delta_fitted_minus_1.0": metric_delta(
                binary_metrics(records_raw), binary_metrics(records_fit)
            ),
            "logit_recovery_floor_hits": floor_hits[0],
        }
        if split in eval_splits:
            metrics[split]["state_clustered_bootstrap_of_delta"] = bootstrap_delta(
                records_raw, records_fit, args.bootstrap_samples, args.bootstrap_seed
            )

    # ------------------------------------------------- engineering survey (post hoc)
    survey = None
    if not args.skip_survey and args.survey.is_file():
        rows_raw, result_raw = survey_probe(engine, args.survey, 1.0, args.batch_questions)
        rows_fit, result_fit = survey_probe(engine, args.survey, fitted_temperature,
                                            args.batch_questions)
        survey = {
            "path": str(args.survey.relative_to(REPO_ROOT)),
            "sha256": sha256_file(args.survey),
            "role": "post-hoc probe only; no temperature was fitted or selected on it",
            "temperature_ceiling_semantics": (
                "temperature_to_reach[t] is the largest T at which a question still clears "
                "threshold t, so a question is answered at t exactly when T <= that value. "
                "temperature_ceiling_to_answer_all_questions is the smallest such value: "
                "answering every question requires T at or below it."
            ),
            "temperature_1.0": survey_metrics(rows_raw),
            f"temperature_{fitted_temperature:.6f}": survey_metrics(rows_fit),
            "per_question_at_1.0": rows_raw,
            "per_question_at_fitted": rows_fit,
        }

    report = {
        "schema_version": "nanojev-temperature-fit-v1",
        "generated_by": "scripts/fit_decision_temperature_v1.py",
        "network_model_calls": 0,
        "checkpoint": {
            "directory": str(args.checkpoint_dir),
            "config": read_json(args.checkpoint_dir / "config.json"),
        },
        "method": {
            "what_is_fitted": "one shared scalar temperature applied as softmax(scores / T)",
            "not_fitted": [
                "model weights (no gradient step, no training)",
                "per-question or per-class temperatures",
                "any threshold or abstention rule",
            ],
            "fit_objective": args.fit_objective,
            "fit_split": fit_split,
            "also_fit_for_robustness": also_fit,
            "evaluation_splits": eval_splits,
            "how_temperature_is_applied": (
                "DecisionPredictor.predict(payload, temperature=T); the service response "
                "reports temperature.value = T with fitted_by_this_command = false. No "
                "service file was edited."
            ),
            "metrics_contract": {
                "nll": f"mean -log p of the true label, probability floored at {NLL_FLOOR}",
                "brier_scalar_bernoulli": "mean (p_true - y)^2",
                "brier_two_class_sum": "2 x the scalar Bernoulli Brier (repo continuity)",
                "ece_fixed_width_10": (
                    "10 fixed-width bins on p(true), confidence-weighted |mean_p - mean_y|"
                ),
                "ece_adaptive_width_10": "same, equal-count bins (bin-sensitivity check)",
                "answer_rate": (
                    "fraction with max(p_true, 1-p_true) >= threshold, matching the skill's "
                    "confidence() and abstain_below semantics"
                ),
                "accuracy_invariance": (
                    "temperature is strictly monotone, so argmax accuracy is invariant; only "
                    "calibration and the threshold operating point can move"
                ),
            },
        },
        "split_definition": {
            "cohort_directory": str(args.cohort_dir),
            "manifest": {
                "path": str(cohort.manifest_path.relative_to(REPO_ROOT)),
                "sha256": cohort.manifest_sha256,
                "declared_checks": cohort.manifest.get("checks"),
                "declared_inputs": cohort.manifest.get("inputs"),
            },
            "splits": cohort.files,
            "fit_data_is": (
                f"{fit_split}.jsonl only ({cohort.files[fit_split]['rows']} states, "
                f"{cohort.files[fit_split]['questions']} Boolean questions)"
            ),
            "why_this_is_not_test_data": [
                "the split is a separate frozen file with its own manifest sha256",
                "state_id overlap with every other split is 0 (computed, not asserted)",
                "source_group_id overlap with every other split is 0, so no maze group leaks",
                "the checkpoint config records train_questions = 576 = the whole of train.jsonl, "
                "so calibration.jsonl was never trained on",
                "the checkpoint config records selection = 'minimum dev target CE; held-out "
                "test first evaluated after training and checkpoint selection'",
            ],
            "disjointness": disjointness,
            "guard": {
                "self_test": guard_self_test(),
                "layers": [
                    "G1 name-level: --fit-split/--also-fit refuse test and ood",
                    "G2 hash-level: every split must match the frozen manifest sha256",
                    "G3 row-level: every fitting row must carry split == the fit split",
                    "G4 data-level: fitting state_ids/source_groups must be disjoint from "
                    "test/ood",
                    "G5 objective-level: each scored row's split is re-checked in the loop",
                ],
                "fitted_on_test_or_ood": False,
            },
        },
        "fit": {
            "primary": primary,
            "all_fits": fits,
            "fitted_temperature": fitted_temperature,
            "temperature_1.0_baseline": {
                "calibration_nll": fits[fit_split]["nll"]["loss_at_temperature_1"],
                "calibration_brier": fits[fit_split]["brier"]["loss_at_temperature_1"],
            },
        },
        "metrics": metrics,
        "service_path_verification": verification,
        "inference": inference,
        "engineering_survey_post_hoc": survey,
        "limits": [
            "One checkpoint, one cohort, one scalar. The fitted value is an in-domain "
            "operating point for local-maze Boolean geometry, not a general calibration.",
            "Temperature scaling cannot change argmax accuracy or the ranking of "
            "confidences; it only rescales them.",
            "ECE is bin-sensitive; two binnings are reported and both are finite-sample "
            "estimates on 176 (test) and 64 (OOD) questions.",
            "The engineering survey is applied post hoc to show direction, not to claim a "
            "calibrated engineering operating point.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "fitted_temperature": fitted_temperature,
        "fit_objective": args.fit_objective,
        "fit_split": fit_split,
        "guard_passed": report["split_definition"]["guard"]["self_test"]["passed"],
        "fitted_on_test_or_ood": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FitSplitViolation, ValueError) as error:
        print(f"fit_decision_temperature_v1: refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
