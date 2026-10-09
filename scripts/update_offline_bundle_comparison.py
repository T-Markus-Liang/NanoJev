#!/usr/bin/env python3
"""Update the offline-bundle comparison scorecard from normalized summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def receipt_stats(receipt_path: Path) -> dict:
    stats = {"receipt_errors": 0, "input_tokens": 0, "output_tokens": 0}
    if not receipt_path.exists():
        return stats
    for line in receipt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stats["receipt_errors"] += bool(row.get("error"))
        answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
        stats["input_tokens"] += row.get("input_tokens") or answer.get("input_tokens") or 0
        stats["output_tokens"] += row.get("output_tokens") or answer.get("output_tokens") or 0
    return stats


def source_tiers(predictions_path: Path, bundle_root: Path) -> dict:
    tier_by_id = {}
    official_dir = bundle_root / "official_jevbench_v1.2.4_public"
    for tier in ("original", "easy", "hard"):
        path = official_dir / f"{tier}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tier_by_id[json.loads(line)["id"]] = tier
    tiers = defaultdict(lambda: {"n": 0, "correct": 0})
    if not predictions_path.exists():
        return {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tier = tier_by_id.get(row.get("id"))
        if tier is None or row.get("gold_label") is None:
            continue
        tiers[tier]["n"] += 1
        tiers[tier]["correct"] += bool(row.get("correct"))
    return {
        tier: {
            "n": stats["n"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["n"] if stats["n"] else None,
        }
        for tier, stats in sorted(tiers.items())
    }


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(scorecard: dict) -> str:
    display_names = {
        "agentjev": "AgentJev-0.6B",
        "certo": "Certo",
        "decider_0_8b": "Decider-0.8B",
        "decider_2b": "Decider-2B",
        "gavel_base": "Gavel-base",
        "jfast_seed30": "J-FAST seed30",
        "kev_0_6b": "Kev-0.6B",
        "kev_4b": "Kev-4B",
        "kev_9b": "Kev-9B",
        "laya_base": "Laya base",
        "laya_typed_decisions": "Laya typed-decisions",
        "minicpm_seed20": "NanoJev MiniCPM seed20",
        "official_jev_direct": "Official Jev direct",
        "openalt_qwen35_4b": "OpenAlternative Qwen3.5-4B",
        "opendecision": "OpenDecision",
        "openjev_deberta_v3_large": "OpenJev DeBERTa-v3-large",
        "openjev_verdict_1_4": "OpenJev Verdict 151M",
        "openjev_zefan2b": "Open-Jev-2B",
        "reflex_stable": "Reflex stable",
        "semif_qwen35_4b_mlx4": "SemIf Qwen3.5-4B MLX4",
        "smalljev_v9": "SmallJev-v9",
        "this_that_model_1_0": "this-that-model-1.0",
        "winnow_12b_q8": "Winnow-12B Q8",
    }
    candidates = scorecard.get("candidates", {})
    official = candidates.get("official_jev_direct", {}).get("overall", {})
    official_acc = official.get("accuracy")
    official_wall = official.get("latency_s", {}).get("wall")
    rows = []
    for key, item in candidates.items():
        overall = item.get("overall", {})
        latency = overall.get("latency_s", {})
        every = item.get("every_retrieval") or {}
        perturb = item.get("semif_perturbation_stability") or {}
        acc = overall.get("accuracy")
        wall = latency.get("wall")
        delta = None if acc is None or official_acc is None else acc - official_acc
        speed = None if wall in (None, 0) or official_wall is None else official_wall / wall
        rows.append((acc if acc is not None else -1, key, item, delta, speed))
    rows.sort(reverse=True)

    lines = [
        "# NanoJev local decision benchmark",
        "",
        "**Frozen offline bundle:** 1,720 rows / 893 labeled rows across official JevBench public, SemIf authored/perturbation/shape fixtures, WANLI, and Every retrieval.",
        "**Protocol:** evaluation-only, pinned source/model revisions, no benchmark-derived training or calibration, and no generated answer tokens unless an upstream engine requires them. See [`docs/BENCHMARK_PROTOCOL_V1.md`](docs/BENCHMARK_PROTOCOL_V1.md).",
        "**Publication boundary:** this scorecard publishes aggregate metrics only; raw provider responses and per-item receipts remain local.",
        "",
        "Ranked by labeled accuracy. Local wall time is measured end-to-end on Apple Silicon; Official Jev direct uses remote response latency and is not strictly comparable for speed.",
        "",
        "| Rank | Candidate | Acc | Δ vs Jev | Speed× | BalAcc | NLL ↓ | Brier ↓ | Cov@0.9 | CW@0.9 | Perturb flip | Every R@1 | Every MRR | Wall s | p50 s | p95 s | Err | Tok in | Tok out |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, key, item, delta, speed) in enumerate(rows, 1):
        overall = item.get("overall", {})
        latency = overall.get("latency_s", {})
        every = item.get("every_retrieval") or {}
        perturb = item.get("semif_perturbation_stability") or {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                rank,
                display_names.get(key, key),
                fmt(overall.get("accuracy")),
                "—" if delta is None else f"{delta:+.4f}",
                "—" if speed is None else f"{speed:.2f}×",
                fmt(overall.get("balanced_accuracy")),
                fmt(overall.get("nll")),
                fmt(overall.get("brier")),
                fmt(overall.get("coverage_at_0.9")),
                overall.get("confident_wrong_at_0.9", "—"),
                fmt(perturb.get("argmax_flip_rate")),
                fmt(every.get("recall_at_1")),
                fmt(every.get("mrr")),
                fmt(latency.get("wall"), 1),
                fmt(latency.get("p50"), 3),
                fmt(latency.get("p95"), 3),
                item.get("receipt_errors", "—"),
                item.get("input_tokens", "—"),
                item.get("output_tokens", "—"),
            )
        )
    lines += [
        "",
        "Metric notes:",
        "- Acc / BalAcc: top-label accuracy over labeled rows; BalAcc averages class recall.",
        "- NLL / Brier: probability quality; lower is better.",
        "- Cov@0.9: fraction with max probability ≥0.9; CW@0.9 is the count of confident errors.",
        "- Perturb flip: argmax changes across the 108 SemIf perturbation pairs.",
        "- Every R@1/MRR: retrieval metrics on the 19-query Every subset.",
        "- Δ vs Jev: candidate accuracy minus Official Jev direct accuracy.",
        "- Speed×: Official Jev direct wall / candidate wall; >1 means faster on this bundle.",
        "- Err: receipt rows with transport/runtime/schema errors.",
        "",
        "Bundle: [`data/jevbench_offline_bundle_v1`](data/jevbench_offline_bundle_v1) "
        "(manifest SHA-256 `1fc3234acf806016adfd9906fade547ca597f87acf749aecac1ba706f6dd8ee5`).",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/offline_bundle_comparison_v1.json"))
    parser.add_argument("--bundle-root", type=Path,
                        default=Path("data/jevbench_offline_bundle_v1"))
    parser.add_argument("--candidate", action="append", required=True,
                        help="key=normalized-summary.json")
    parser.add_argument("--markdown", type=Path,
                        help="optional Markdown scorecard output")
    args = parser.parse_args()

    output = args.output.resolve()
    bundle_root = args.bundle_root.resolve()
    scorecard = load_json(output)
    candidates = dict(scorecard.get("candidates", {}))
    by_file = defaultdict(dict, {
        file_path: dict(metrics)
        for file_path, metrics in scorecard.get("by_file", {}).items()
    })
    public_tiers = dict(scorecard.get("official_public_source_tiers", {}))

    for spec in args.candidate:
        key, summary_path = spec.split("=", 1)
        summary_path = Path(summary_path).resolve()
        summary = load_json(summary_path)
        predictions_path = Path(summary["predictions_path"]).resolve()
        receipt_path = Path(summary["receipt"]).resolve()
        candidates[key] = {
            "overall": summary["overall"],
            "summary": str(summary_path),
            "receipt": str(receipt_path),
            **receipt_stats(receipt_path),
            "summary_sha256": sha256(summary_path),
            "predictions": str(predictions_path),
            "predictions_sha256": summary.get("predictions_sha256") or sha256(predictions_path),
            "semif_perturbation_stability": summary.get("semif_perturbation_stability"),
            "every_retrieval": summary.get("every_retrieval"),
        }
        for file_path, metrics in summary.get("files", {}).items():
            by_file[file_path][key] = metrics
        public_tiers[key] = source_tiers(predictions_path, bundle_root)

    scorecard["candidates"] = dict(sorted(candidates.items()))
    scorecard["by_file"] = {
        file_path: dict(sorted(metrics.items()))
        for file_path, metrics in sorted(by_file.items())
    }
    scorecard["official_public_source_tiers"] = dict(sorted(public_tiers.items()))

    headline = scorecard.setdefault("headline", {})
    if "official_jev_direct" in candidates and "minicpm_seed20" in candidates:
        headline["official_vs_minicpm_accuracy_delta"] = (
            candidates["official_jev_direct"]["overall"]["accuracy"]
            - candidates["minicpm_seed20"]["overall"]["accuracy"]
        )
    if "minicpm_seed20" in candidates and "jfast_seed30" in candidates:
        headline["minicpm_vs_jfast_accuracy_delta"] = (
            candidates["minicpm_seed20"]["overall"]["accuracy"]
            - candidates["jfast_seed30"]["overall"]["accuracy"]
        )
        headline["jfast_speedup_vs_minicpm"] = (
            candidates["minicpm_seed20"]["overall"]["latency_s"]["wall"]
            / candidates["jfast_seed30"]["overall"]["latency_s"]["wall"]
        )
    if "official_jev_direct" in candidates and "jfast_seed30" in candidates:
        headline["official_vs_jfast_accuracy_delta"] = (
            candidates["official_jev_direct"]["overall"]["accuracy"]
            - candidates["jfast_seed30"]["overall"]["accuracy"]
        )
    if "kev_0_6b" in candidates:
        kev = candidates["kev_0_6b"]
        headline["kev06_every_mrr"] = kev["every_retrieval"]["mrr"]
        if "minicpm_seed20" in candidates:
            mini = candidates["minicpm_seed20"]
            headline["kev06_vs_minicpm_accuracy_delta"] = (
                kev["overall"]["accuracy"] - mini["overall"]["accuracy"]
            )
            headline["kev06_speedup_vs_minicpm"] = (
                mini["overall"]["latency_s"]["wall"]
                / kev["overall"]["latency_s"]["wall"]
            )
        if "official_jev_direct" in candidates:
            headline["official_vs_kev06_accuracy_delta"] = (
                candidates["official_jev_direct"]["overall"]["accuracy"]
                - kev["overall"]["accuracy"]
            )

    for item in candidates.values():
        summary_path = Path(item["summary"])
        if "receipt" not in item and summary_path.exists():
            item["receipt"] = load_json(summary_path).get("receipt")
        if "receipt_errors" not in item and item.get("receipt"):
            item.update(receipt_stats(Path(item["receipt"])))

    output.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + "\n")
    result = {"output": str(output), "sha256": sha256(output),
              "candidates": sorted(candidates)}
    if args.markdown:
        markdown_path = args.markdown.resolve()
        markdown_path.write_text(render_markdown(scorecard), encoding="utf-8")
        result["markdown"] = str(markdown_path)
        result["markdown_sha256"] = sha256(markdown_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
