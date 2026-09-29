#!/usr/bin/env python3
"""Manual labeling workflow for real-context eval V1 candidates.

Implements the labeling half of docs/REAL_CONTEXT_EVAL_V1.md §4 and the rules
in docs/REAL_CONTEXT_LABELING_GUIDE_V1.md:

  --first-pass    Label candidates. Input is data/real_context_eval_v1/
                  candidates.jsonl (targets: null). Label sources:
                    --interactive      prompt per candidate (rendered
                                       conversation + marked candidate)
                    --subagent-json F  pre-decided labels: a JSON object
                                       {record_id: true|false|"uncertain"}
                                       or {record_id: {"label": ..., "evidence": "..."}}
                    --print [F]        render candidates for external labeling
                                       (stdout, or a file if a path is given)
                  Every label row records labeler identity (--labeler) and
                  labeler_type (--labeler-type human|model). Model proposals
                  are queue input only: they are NEVER scored labels and are
                  excluded at --finalize unless a human labeled the record.

  --adjudicate    Compare two label files (--labels-a, --labels-b) over the
                  overlapping record_ids. Writes disagreements to
                  adjudication_queue.jsonl plus an adjudication_report.json
                  (n, raw agreement, binary Cohen's kappa with
                  uncertain counted as keep).

  --finalize      Merge label files (--labels F [F ...]) into eval.jsonl with
                  targets filled, excluded.jsonl for rows that cannot be
                  scored, and update manifest.json (hashes/counts only —
                  content-free, per the eval design).

Label semantics: true  = candidate certainly irrelevant = DROP
                 false = keep (required evidence, constraint, correction,
                        tool dependency, safety restriction, interpretation
                        dependency — or uncertain)
                 "uncertain" = recorded, excluded from the scored set.

No provider calls, no scorer calls, no auto-derived ground truth.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "real_context_eval_v1"

LABEL_KEEP = False
LABEL_DROP = True
LABEL_UNCERTAIN = "uncertain"
LABEL_VALUES = (True, False, "uncertain")


# --- io -----------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: bad JSON: {e}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path):
    p = Path(path)
    return {"sha256": sha256_file(p), "bytes": p.stat().st_size}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_state(record):
    try:
        state = json.loads(record["request"]["state"])
    except (KeyError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"record {record.get('meta', {}).get('record_id', '?')}: "
            f"cannot parse request.state: {e}")
    return state


# --- label normalization --------------------------------------------------------

def normalize_label(value, where=""):
    """Map a raw label value to True (drop), False (keep), or 'uncertain'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "drop", "irrelevant", "1"):
            return True
        if v in ("false", "keep", "relevant", "0"):
            return False
        if v in ("uncertain", "unsure", "u", "skip", "?"):
            return LABEL_UNCERTAIN
    raise SystemExit(f"{where}: invalid label {value!r}; "
                     f"expected true|false|'uncertain' (or keep/drop)")


def load_label_file(path):
    """Read a label JSONL file -> list of label entries.

    Entry schema: {"record_id": str, "label": bool|"uncertain",
                   "labeler": str, "labeler_type": "human"|"model",
                   "labeled_at": iso str, "evidence": str?,
                   "confidence": "high"|"low"?}
    """
    entries = []
    for i, row in enumerate(load_jsonl(path), 1):
        rid = row.get("record_id")
        if not rid:
            raise SystemExit(f"{path}:{i}: missing record_id")
        if "label" not in row:
            raise SystemExit(f"{path}:{i}: {rid}: missing label")
        entries.append({
            "record_id": rid,
            "label": normalize_label(row["label"], f"{path}:{i}:{rid}"),
            "labeler": row.get("labeler", "unknown"),
            "labeler_type": row.get("labeler_type", "human"),
            "labeled_at": row.get("labeled_at"),
            "evidence": row.get("evidence"),
            "confidence": row.get("confidence"),
            "group_id": row.get("group_id"),
            "candidate_pointer": row.get("candidate_pointer"),
        })
    return entries


def load_label_objects(paths):
    """Load one or more label files -> {record_id: [entries]}."""
    by_id = defaultdict(list)
    for p in paths:
        for e in load_label_file(p):
            e["source_file"] = str(p)
            by_id[e["record_id"]].append(e)
    return by_id


# --- rendering -------------------------------------------------------------------

def render_record(record, out):
    """Render the full conversation with the candidate marked. No truncation —
    the labeler reads the full transcript (guide §1)."""
    meta = record.get("meta", {})
    state = parse_state(record)
    cand = state.get("candidate_pointer")
    conv = state.get("conversation", [])
    w = out.write
    w("=" * 78 + "\n")
    w(f"record_id:        {meta.get('record_id')}\n")
    w(f"group_id:         {record.get('group_id')}\n")
    w(f"candidate_pointer:{cand}   kind: {meta.get('candidate_kind')}\n")
    w(f"segments:         {len(conv)}\n")
    instructions = (record.get("request", {}).get("questions", {})
                    .get("irrelevant", {}).get("instructions"))
    if instructions:
        w("question:         " + instructions + "\n")
    w("-" * 78 + "\n")
    found = False
    for i, seg in enumerate(conv):
        is_cand = seg.get("pointer") == cand
        if is_cand:
            found = True
            w(">>> CANDIDATE " + "=" * 63 + "\n")
        w(f"[{i:03d}] role={seg.get('role')} pointer={seg.get('pointer')}\n")
        w(str(seg.get("content", "")) + "\n")
        if is_cand:
            w(">>> END CANDIDATE " + "=" * 59 + "\n")
        w("\n")
    if not found:
        w(f"!! candidate_pointer {cand} not present in conversation\n\n")
    return found


# --- first pass --------------------------------------------------------------------

def cmd_first_pass(args):
    candidates = load_jsonl(args.candidates)
    if args.record_ids:
        wanted = set(args.record_ids.split(","))
        candidates = [r for r in candidates
                      if r["meta"]["record_id"] in wanted]
    if args.shuffle:
        rng = random.Random(args.seed)
        candidates = list(candidates)
        rng.shuffle(candidates)
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print("no candidates selected", file=sys.stderr)
        return 0

    if args.print_mode is not None:
        # render-only path: dump for external labeling, write nothing else
        if args.print_mode == "-":
            for r in candidates:
                render_record(r, sys.stdout)
        else:
            with open(args.print_mode, "w", encoding="utf-8") as f:
                for r in candidates:
                    render_record(r, f)
            print(f"rendered {len(candidates)} candidates -> {args.print_mode}",
                  file=sys.stderr)
        return 0

    if not args.labeler:
        raise SystemExit("--first-pass (labeling) requires --labeler NAME")
    labeler_type = args.labeler_type
    if args.subagent_json and labeler_type == "human":
        print("note: --subagent-json labels are recorded with "
              f"--labeler-type {labeler_type}; pass --labeler-type model "
              "when the file contains model proposals", file=sys.stderr)

    labels_out = Path(args.labels_out or
                      Path(args.out_dir) / "labels"
                      / f"labels_{args.labeler}.jsonl")
    done = {}
    if labels_out.exists():
        for e in load_label_file(labels_out):
            done[e["record_id"]] = e
        print(f"resuming: {len(done)} existing labels in {labels_out}",
              file=sys.stderr)

    decided = {}
    if args.subagent_json:
        raw = json.loads(Path(args.subagent_json).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit("--subagent-json must be a JSON object "
                             "{record_id: label}")
        known = {r["meta"]["record_id"] for r in candidates}
        for rid, value in raw.items():
            if rid not in known:
                print(f"warn: {rid} not in selected candidates; skipped",
                      file=sys.stderr)
                continue
            if isinstance(value, dict):
                decided[rid] = {
                    "label": normalize_label(value.get("label"), f"subagent:{rid}"),
                    "evidence": value.get("evidence"),
                    "confidence": value.get("confidence"),
                }
            else:
                decided[rid] = {"label": normalize_label(value, f"subagent:{rid}"),
                                "evidence": None, "confidence": None}
    elif not args.interactive:
        raise SystemExit("--first-pass needs a label source: "
                         "--interactive, --subagent-json FILE, or --print [FILE]")

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(labels_out, "a", encoding="utf-8") as lf:
        for rec in candidates:
            rid = rec["meta"]["record_id"]
            if rid in done:
                continue
            if args.interactive:
                render_record(rec, sys.stdout)
                while True:
                    ans = input("label [k=keep/d=drop/u=uncertain/"
                                "s=skip/q=quit]: ").strip().lower()
                    if ans in ("q", "quit"):
                        print(f"stopped; {n_written} labels written "
                              f"this run -> {labels_out}")
                        return 0
                    if ans in ("s", "skip", ""):
                        label = None
                        break
                    if ans in ("k", "keep"):
                        label = LABEL_KEEP
                        break
                    if ans in ("d", "drop"):
                        label = LABEL_DROP
                        break
                    if ans in ("u", "uncertain"):
                        label = LABEL_UNCERTAIN
                        break
                    print("  ? use k/d/u/s/q")
                if label is None:
                    continue
                basis = input("evidence basis (one line, optional): ").strip()
                entry = {"label": label, "evidence": basis or None,
                         "confidence": None}
            else:
                if rid not in decided:
                    print(f"warn: no subagent label for {rid}; skipped",
                          file=sys.stderr)
                    continue
                entry = decided[rid]
            row = {
                "record_id": rid,
                "label": entry["label"],
                "labeler": args.labeler,
                "labeler_type": labeler_type,
                "labeled_at": now_iso(),
                "group_id": rec.get("group_id"),
                "candidate_pointer": rec["meta"].get("candidate_pointer"),
            }
            if entry.get("evidence"):
                row["evidence"] = entry["evidence"]
            if entry.get("confidence") in ("high", "low"):
                row["confidence"] = entry["confidence"]
            lf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} labels -> {labels_out} "
          f"(total now {len(done) + n_written})", file=sys.stderr)
    return 0


# --- adjudication -----------------------------------------------------------------

def cohens_kappa(labels_a, labels_b):
    """Binary Cohen's kappa. Returns None when undefined (single category)."""
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else None
    return (po - pe) / (1.0 - pe)


def scored_view(label):
    """Binary keep/drop view for agreement stats: uncertain counts as keep."""
    return "drop" if label is True else "keep"


def cmd_adjudicate(args):
    a = {e["record_id"]: e for e in load_label_file(args.labels_a)}
    b = {e["record_id"]: e for e in load_label_file(args.labels_b)}
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if not shared:
        raise SystemExit("label files share no record_ids")

    queue, agrees, raw_agrees = [], 0, 0
    for rid in shared:
        ea, eb = a[rid], b[rid]
        exact = ea["label"] == eb["label"]
        same_scored = scored_view(ea["label"]) == scored_view(eb["label"])
        if exact:
            raw_agrees += 1
        if same_scored:
            agrees += 1
        if not exact:
            queue.append({
                "record_id": rid,
                "group_id": ea.get("group_id") or eb.get("group_id"),
                "candidate_pointer": (ea.get("candidate_pointer")
                                      or eb.get("candidate_pointer")),
                "label_a": ea["label"], "labeler_a": ea["labeler"],
                "label_b": eb["label"], "labeler_b": eb["labeler"],
                "flips_scored_label": not same_scored,
                "evidence_a": ea.get("evidence"),
                "evidence_b": eb.get("evidence"),
            })
    kappa = cohens_kappa([scored_view(a[r]["label"]) for r in shared],
                         [scored_view(b[r]["label"]) for r in shared])
    report = {
        "schema": "nanojev-real-context-eval-v1-adjudication",
        "created_at": now_iso(),
        "labels_a": {"path": str(args.labels_a), **file_stat(args.labels_a),
                     "n": len(a)},
        "labels_b": {"path": str(args.labels_b), **file_stat(args.labels_b),
                     "n": len(b)},
        "double_labeled": len(shared),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "raw_exact_agreement": raw_agrees / len(shared),
        "scored_agreement": agrees / len(shared),
        "cohens_kappa_binary": kappa,
        "kappa_note": "uncertain counted as keep; target kappa >= 0.75",
        "disagreements": len(queue),
        "scored_label_flips": sum(q["flips_scored_label"] for q in queue),
    }
    write_jsonl(args.queue_out, queue)
    Path(args.report_out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"disagreement queue ({len(queue)}) -> {args.queue_out}",
          file=sys.stderr)
    return 0


# --- finalize ------------------------------------------------------------------------

def resolve_record(rid, entries):
    """Decide the final disposition of one candidate from its label entries.

    Returns (action, detail) where action is 'score' or 'exclude'.
    Human definite labels conflict -> SystemExit (adjudicate first).
    Model-labeled records are never scored (design §3.4: provider/scorer/
    model outputs are never labels); they may only queue a human decision.
    """
    human = [e for e in entries if e["labeler_type"] != "model"]
    model_only = bool(entries) and not human
    if model_only:
        return "exclude", {"reason": "model_proposal_not_label",
                           "label": LABEL_UNCERTAIN, "entry": entries[-1]}
    definite = [e for e in human if e["label"] in (True, False)]
    # an adjudicated entry is the discussion outcome and resolves conflicts
    adjudicated = [e for e in definite if e["labeler"] == "adjudicated"]
    vals = {e["label"] for e in definite}
    if len(vals) > 1 and not adjudicated:
        raise SystemExit(
            f"{rid}: conflicting human labels "
            f"{[(e['label'], e['labeler'], e.get('source_file')) for e in definite]} "
            f"-- resolve via --adjudicate and an adjudicated label file")
    if definite:
        winner = adjudicated[-1] if adjudicated else definite[-1]
        uncertain_along = any(e["label"] == LABEL_UNCERTAIN for e in human)
        # low-confidence routes to excluded.jsonl (design §4.4): an explicit
        # confidence:"low" on the winning entry, or a human 'uncertain' label
        # alongside the definite one. An adjudicated definite label is a
        # resolved disagreement and is high-confidence unless marked low.
        low = winner.get("confidence") == "low" or uncertain_along
        return "score", {"label": winner["label"], "entry": winner,
                         "confidence": "low" if low else "high"}
    # humans only produced uncertain labels
    return "exclude", {"reason": "uncertain",
                       "label": LABEL_UNCERTAIN, "entry": human[-1]}


def cmd_finalize(args):
    candidates = load_jsonl(args.candidates)
    labels = load_label_objects(args.labels)
    eval_rows, excluded_rows = [], []
    counts = Counter()
    for rec in candidates:
        rid = rec["meta"]["record_id"]
        entries = labels.get(rid, [])
        rec = dict(rec)
        rec["meta"] = dict(rec["meta"])
        if not entries:
            action, det = "exclude", {"reason": "no_label",
                                      "label": LABEL_UNCERTAIN, "entry": None}
        else:
            action, det = resolve_record(rid, entries)
        # design §4.4: low-confidence rows are excluded from the scored set
        if action == "score" and det.get("confidence") == "low":
            action, det["reason"] = "exclude", "low_confidence"
        label = det["label"]
        if label is LABEL_UNCERTAIN:
            # conservative keep semantics on excluded rows (design §4:
            # uncertain -> false); not part of the scored set either way
            prob = {"true": 0.0, "false": 1.0} if det["reason"] == "uncertain" \
                else None
        else:
            prob = {"true": 1.0, "false": 0.0} if label is True \
                else {"true": 0.0, "false": 1.0}
        rec["targets"] = ({"irrelevant": {"probabilities": prob}}
                          if prob is not None else None)
        entry = det.get("entry")
        if entry:
            rec["meta"]["labeler"] = entry["labeler"]
            rec["meta"]["labeler_type"] = entry["labeler_type"]
            rec["meta"]["labeled_at"] = entry.get("labeled_at")
            if entry.get("evidence"):
                rec["meta"]["evidence_basis"] = entry["evidence"]
        rec["meta"]["label_confidence"] = det.get("confidence", "low"
                                                  if action == "exclude"
                                                  else "high")
        if action == "score":
            rec["meta"]["label_origin"] = "human_label"
            eval_rows.append(rec)
            counts["keep" if label is False else "drop"] += 1
        else:
            rec["meta"]["exclusion_reason"] = det["reason"]
            rec["meta"]["label_origin"] = (
                "model_proposal_only" if det["reason"] == "model_proposal_not_label"
                else "unlabeled" if det["reason"] == "no_label" else "human_label")
            excluded_rows.append(rec)
            counts[f"excluded_{det['reason']}"] += 1

    write_jsonl(args.eval_out, eval_rows)
    write_jsonl(args.excluded_out, excluded_rows)

    # content-free manifest update (hashes + counts only; raw text stays
    # inside the gitignored data root)
    mpath = Path(args.manifest)
    manifest = json.loads(mpath.read_text(encoding="utf-8")) \
        if mpath.exists() else {"schema_version": "nanojev-real-context-eval-v1"}
    label_files = []
    for p in args.labels:
        ets = load_label_file(p)
        label_files.append({
            "path": str(p), **file_stat(p), "n": len(ets),
            "labelers": sorted({e["labeler"] for e in ets}),
            "labeler_types": sorted({e["labeler_type"] for e in ets}),
        })
    report_path = Path(args.out_dir) / "adjudication_report.json"
    agreement = None
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        agreement = {"double_labeled": rep.get("double_labeled"),
                     "cohens_kappa_binary": rep.get("cohens_kappa_binary"),
                     "scored_agreement": rep.get("scored_agreement"),
                     "report_sha256": sha256_file(report_path)}
    manifest["labeling"] = {
        "phase": "labeled",
        "finalized_at": now_iso(),
        "guide": "docs/REAL_CONTEXT_LABELING_GUIDE_V1.md",
        "label_files": label_files,
        "agreement": agreement,
        "rule": "uncertain -> excluded (conservative keep); model proposals "
                "are never scored labels",
    }
    manifest.setdefault("records", {})["candidates"] = len(candidates)
    manifest["records"]["eval"] = len(eval_rows)
    manifest["records"]["excluded"] = len(excluded_rows)
    manifest["label_counts"] = {
        "unlabeled_remaining": counts.get("excluded_no_label", 0),
        "keep": counts.get("keep", 0),
        "drop": counts.get("drop", 0),
        "excluded": {k[len("excluded_"):]: v for k, v in counts.items()
                     if k.startswith("excluded_")},
    }
    manifest.setdefault("files", {})
    for name, p in (("eval.jsonl", args.eval_out),
                    ("excluded.jsonl", args.excluded_out)):
        manifest["files"][name] = file_stat(p)
    q = Path(args.out_dir) / "adjudication_queue.jsonl"
    if q.exists():
        manifest["files"]["adjudication_queue.jsonl"] = file_stat(q)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(f"eval.jsonl:     {len(eval_rows)} scored rows "
          f"(keep={counts.get('keep', 0)} drop={counts.get('drop', 0)})")
    print(f"excluded.jsonl: {len(excluded_rows)} rows "
          f"({dict((k, v) for k, v in counts.items() if k.startswith('excluded'))})")
    print(f"manifest updated -> {mpath}")
    if counts.get("excluded_no_label"):
        print(f"WARNING: {counts['excluded_no_label']} candidates had no label "
              f"and were excluded; rerun --first-pass before finalizing "
              f"a complete set", file=sys.stderr)
    return 0


# --- cli -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--first-pass", action="store_true",
                      help="label candidates")
    mode.add_argument("--adjudicate", action="store_true",
                      help="compare two label files")
    mode.add_argument("--finalize", action="store_true",
                      help="merge labels into eval.jsonl + excluded.jsonl")
    p.add_argument("--candidates",
                   default=str(DEFAULT_DIR / "candidates.jsonl"))
    p.add_argument("--out-dir", default=str(DEFAULT_DIR))
    # first pass
    p.add_argument("--print", dest="print_mode", nargs="?", const="-",
                   default=None, metavar="FILE",
                   help="render candidates for external labeling "
                        "(stdout if no FILE)")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--subagent-json", metavar="FILE",
                   help="JSON {record_id: true|false|'uncertain'}")
    p.add_argument("--labeler", help="labeler identity -> meta.labeler")
    p.add_argument("--labeler-type", choices=("human", "model"),
                   default="human",
                   help="'model' marks proposals; never scored as labels")
    p.add_argument("--labels-out", metavar="FILE",
                   help="label JSONL output (default <out-dir>/labels/"
                        "labels_<labeler>.jsonl)")
    p.add_argument("--limit", type=int)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--record-ids", help="comma-separated record_id subset")
    # adjudicate
    p.add_argument("--labels-a", metavar="FILE")
    p.add_argument("--labels-b", metavar="FILE")
    p.add_argument("--queue-out",
                   default=str(DEFAULT_DIR / "adjudication_queue.jsonl"))
    p.add_argument("--report-out",
                   default=str(DEFAULT_DIR / "adjudication_report.json"))
    # finalize
    p.add_argument("--labels", nargs="+", metavar="FILE",
                   help="label JSONL files to merge (order = precedence for "
                        "non-conflicting entries)")
    p.add_argument("--eval-out", default=str(DEFAULT_DIR / "eval.jsonl"))
    p.add_argument("--excluded-out",
                   default=str(DEFAULT_DIR / "excluded.jsonl"))
    p.add_argument("--manifest", default=str(DEFAULT_DIR / "manifest.json"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.adjudicate:
        if not args.labels_a or not args.labels_b:
            raise SystemExit("--adjudicate requires --labels-a and --labels-b")
        return cmd_adjudicate(args)
    if args.finalize:
        if not args.labels:
            raise SystemExit("--finalize requires --labels FILE [FILE ...]")
        return cmd_finalize(args)
    return cmd_first_pass(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
