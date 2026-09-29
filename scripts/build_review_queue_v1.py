#!/usr/bin/env python3
"""Build model-proposal JSONs and the consolidated owner review queue for
the real-context eval V1.

Phases:

  proposals   Convert backend score files ({i, noul, ms} per line, index into
              candidates.jsonl) into proposal objects
              {record_id: true|false|"uncertain"} under
              data/real_context_eval_v1/proposals/.
              Mapping: 0.3 < noul < 0.7 -> "uncertain"; noul >= 0.7 -> true
              (drop); noul <= 0.3 -> false (keep). Missing noul (backend
              error, e.g. context overflow) -> "uncertain" so the record is
              always routed to human review.

  queue       Join the three label files (model proposals) over all
              candidates and emit review_queue.jsonl + review_queue_report.json.
              Queue membership = union of
                * any-model disagreement (proposal labels not all equal)
                * any "uncertain" vote (incl. missing scores)
                * unanimous drop (all three true)
                * 10% random sample of unanimous keep (seed 42)
              sorted by disagreement severity (scored-label flips, then
              uncertain count, then noul spread).

Local-only: real transcripts and raw backend probabilities never leave the
machine.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_context_eval_v1"
RESULTS = ROOT / "results"

BACKENDS = ("lora", "winnow", "kev")

UNC_LO, UNC_HI = 0.3, 0.7
KEEP_SAMPLE_FRAC = 0.10
SEED = 42


def load_candidates():
    return [json.loads(l) for l in
            (DATA_DIR / "candidates.jsonl").open() if l.strip()]


def load_scores(backend):
    """-> {record_id: noul_or_None}"""
    path = RESULTS / f"{backend}_real_candidates_v1.jsonl"
    rows = [json.loads(l) for l in path.open() if l.strip()]
    cands = load_candidates()
    by_i = {}
    for r in rows:
        by_i[r["i"]] = r
    out = {}
    for i, rec in enumerate(cands):
        rid = rec["meta"]["record_id"]
        row = by_i.get(i)
        out[rid] = row.get("noul") if row else None
    return out


def vote_of(noul):
    if noul is None:
        return "uncertain"
    if UNC_LO < noul < UNC_HI:
        return "uncertain"
    return True if noul >= 0.5 else False


def cmd_proposals(args):
    out_dir = DATA_DIR / "proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for b in args.backends:
        scores = load_scores(b)
        prop = {rid: vote_of(n) for rid, n in scores.items()}
        path = out_dir / f"{b}_proposals.json"
        path.write_text(json.dumps(prop, indent=0, sort_keys=True) + "\n",
                        encoding="utf-8")
        dist = Counter(prop.values())
        dist = {str(k): dist[k] for k in (True, False, "uncertain")}
        missing = sum(1 for n in scores.values() if n is None)
        summary[b] = {"path": str(path), "n": len(prop),
                      "distribution": dist, "missing_noul": missing}
        print(b, summary[b])
    return summary


def load_label_file(path):
    out = {}
    for l in Path(path).open():
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["record_id"]] = r["label"]
    return out


def scored_view(label):
    return "drop" if label is True else "keep"


def cmd_queue(args):
    cands = load_candidates()
    labels = {}
    for b in BACKENDS:
        lf = args.labels_dir / f"labels_{b}.jsonl"
        labels[b] = load_label_file(lf)
    scores = {b: load_scores(b) for b in BACKENDS}

    rows = []
    keep_unanimous = []
    for rec in cands:
        rid = rec["meta"]["record_id"]
        votes = {b: labels[b].get(rid) for b in BACKENDS}
        nouls = {b: scores[b][rid] for b in BACKENDS}
        v = list(votes.values())
        n_unc = sum(1 for x in v if x == "uncertain")
        n_drop = sum(1 for x in v if x is True)
        n_keep = sum(1 for x in v if x is False)
        disagreement = len(set(v)) > 1
        scored = [scored_view(x) for x in v]
        flips = sum(1 for i in range(3) for j in range(i + 1, 3)
                    if scored[i] != scored[j])
        avail = [n for n in nouls.values() if n is not None]
        spread = (max(avail) - min(avail)) if len(avail) > 1 else 0.0
        entry = {
            "record_id": rid,
            "group_id": rec.get("group_id"),
            "candidate_pointer": rec["meta"].get("candidate_pointer"),
            "noul": nouls,
            "votes": votes,
            "n_drop": n_drop, "n_keep": n_keep, "n_uncertain": n_unc,
            "scored_label_flips": flips,
            "noul_spread": round(spread, 6),
            "reasons": [],
        }
        if disagreement:
            entry["reasons"].append("model_disagreement")
        if n_unc:
            entry["reasons"].append("uncertain_vote")
        if n_drop == 3:
            entry["reasons"].append("unanimous_drop")
        if n_keep == 3:
            keep_unanimous.append(entry)
        else:
            if entry["reasons"]:
                rows.append(entry)
    rng = random.Random(SEED)
    sample_n = int(len(keep_unanimous) * KEEP_SAMPLE_FRAC + 0.5)
    sampled = rng.sample(keep_unanimous, sample_n)
    for e in sampled:
        e["reasons"].append("unanimous_keep_sample")
        rows.append(e)

    rows.sort(key=lambda e: (-e["scored_label_flips"], -e["n_uncertain"],
                             -e["noul_spread"], e["record_id"]))
    qpath = DATA_DIR / "review_queue.jsonl"
    with qpath.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    comp = Counter()
    for e in rows:
        for r in e["reasons"]:
            comp[r] += 1
    report = {
        "schema": "nanojev-real-context-eval-v1-review-queue",
        "queue_path": str(qpath),
        "n_candidates": len(cands),
        "queue_size": len(rows),
        "composition": dict(comp),
        "unanimous_keep_total": len(keep_unanimous),
        "unanimous_keep_sampled": len(sampled),
        "sample_seed": SEED,
        "sample_fraction": KEEP_SAMPLE_FRAC,
        "uncertain_band": [UNC_LO, UNC_HI],
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proposals")
    p.add_argument("--backends", nargs="+", default=list(BACKENDS),
                   choices=list(BACKENDS))
    q = sub.add_parser("queue")
    q.add_argument("--labels-dir", type=Path,
                   default=DATA_DIR / "labels")
    q.add_argument("--report-out", type=Path,
                   default=DATA_DIR / "review_queue_report.json")
    args = ap.parse_args()
    if args.cmd == "proposals":
        cmd_proposals(args)
    else:
        report = cmd_queue(args)
        # merge in per-pair kappa from adjudication reports if present
        adj = DATA_DIR / "adjudication"
        kappas = {}
        for f in sorted(adj.glob("adjudication_report_*.json")) \
                if adj.exists() else []:
            r = json.loads(f.read_text())
            kappas[f.stem.replace("adjudication_report_", "")] = {
                "n": r.get("double_labeled"),
                "raw_exact_agreement": r.get("raw_exact_agreement"),
                "scored_agreement": r.get("scored_agreement"),
                "cohens_kappa_binary": r.get("cohens_kappa_binary"),
                "disagreements": r.get("disagreements"),
                "scored_label_flips": r.get("scored_label_flips"),
            }
        report["pairwise"] = kappas
        args.report_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
