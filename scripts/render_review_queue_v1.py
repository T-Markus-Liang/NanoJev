#!/usr/bin/env python3
"""Render the real-context eval V1 owner review bundle.

Joins data/real_context_eval_v1/review_queue.jsonl (severity-ordered, output
of build_review_queue_v1.py) against candidates.jsonl and emits:

  * owner_review.md            — top --top items fully rendered (compact
                                 context: candidate marked '>>>', WINDOW
                                 segments each side, plus the final user
                                 request as the judging anchor), remaining
                                 items as a one-line summary table.
  * owner_labels_template.csv  — record_id,label,note for every queue row in
                                 severity order; the owner fills `label` with
                                 yes (=drop/certainly irrelevant), no
                                 (=keep), or uncertain, then converts with
                                 scripts/csv_to_labels_json.py.

Label semantics (docs/REAL_CONTEXT_LABELING_GUIDE_V1.md):
  true/yes   = candidate certainly irrelevant -> DROP
  false/no   = keep (default when not certain)
  uncertain  = recorded, excluded from the scored set

Local-only: real transcripts never leave the machine; this file is not for
publication.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_context_eval_v1"

WINDOW = 2              # segments shown each side of the candidate
SEG_MAX_CHARS = 360     # truncation for non-candidate segments
SEG_MAX_LINES = 4
CAND_MAX_CHARS = 1600   # the candidate itself gets more room
CAND_MAX_LINES = 14
ANCHOR_MAX_CHARS = 500  # final user request anchor
SNIPPET_CHARS = 90      # summary-table snippet

BACKENDS = ("lora", "winnow", "kev")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_state(record):
    """request.state is a JSON string -> {"conversation": [...], ...}.

    If a future record stores state as a dict/list already, handle it too.
    """
    st = record.get("request", {}).get("state")
    if isinstance(st, str):
        return json.loads(st)
    if isinstance(st, dict):
        return st
    if isinstance(st, list):
        return {"conversation": st}
    return {"conversation": []}


def clip(text, max_chars, max_lines):
    """Truncate a segment's text to max_lines lines / max_chars chars."""
    lines = str(text).splitlines() or [""]
    clipped = lines[:max_lines]
    out = []
    used = 0
    for ln in clipped:
        if used + len(ln) > max_chars:
            ln = ln[: max(0, max_chars - used)]
        out.append(ln)
        used += len(ln)
        if used >= max_chars:
            break
    truncated = len(lines) > len(clipped) or sum(len(l) for l in lines) > used
    return out, truncated


def vote_mark(v):
    return {True: "DROP", False: "keep", "uncertain": "?", None: "-"}.get(v, "?")


def votes_str(votes):
    return " ".join(f"{b}={vote_mark(votes.get(b))}" for b in BACKENDS)


def snippet(text, n=SNIPPET_CHARS):
    s = " ".join(str(text).split())
    return s[:n] + ("…" if len(s) > n else "")


def find_candidate_index(conv, pointer):
    for i, seg in enumerate(conv):
        if seg.get("pointer") == pointer:
            return i
    return None


def final_user_message(conv):
    for seg in reversed(conv):
        if seg.get("role") == "user":
            return seg.get("content", "")
    return None


def render_item(rank, qrow, record, out):
    meta = record.get("meta", {}) if record else {}
    state = parse_state(record) if record else {}
    conv = state.get("conversation", [])
    cand_ptr = qrow.get("candidate_pointer") or state.get("candidate_pointer")
    ci = find_candidate_index(conv, cand_ptr)

    w = out.append
    noul = qrow.get("noul", {})
    noul_str = " ".join(
        f"{b}={noul.get(b):.3f}" if isinstance(noul.get(b), (int, float))
        else f"{b}=-" for b in BACKENDS)
    w(f"### #{rank} `{qrow['record_id']}` — "
      f"{meta.get('candidate_kind', '?')} — votes {votes_str(qrow.get('votes', {}))}")
    w("")
    w(f"- **record_id**: `{qrow['record_id']}`  "
      f"**pointer**: `{cand_ptr}`  "
      f"**kind**: {meta.get('candidate_kind', '?')}  "
      f"**len_band**: {meta.get('length_band', '?')}  "
      f"**lang**: {meta.get('language_bucket', '?')}  "
      f"**segments**: {len(conv)}")
    w(f"- **noul**: {noul_str}  |  **votes**: {votes_str(qrow.get('votes', {}))}  |  "
      f"drop/keep/unc = {qrow.get('n_drop')}/{qrow.get('n_keep')}/{qrow.get('n_uncertain')}  |  "
      f"flips={qrow.get('scored_label_flips')}  spread={qrow.get('noul_spread')}")
    w(f"- **reasons**: {', '.join(qrow.get('reasons', []))}")
    anchor = final_user_message(conv)
    if anchor is not None:
        w(f"- **final user request (anchor)**: {snippet(anchor, ANCHOR_MAX_CHARS)}")
    w("")
    if ci is None:
        w(f"> !! candidate_pointer `{cand_ptr}` not found in conversation "
          f"({len(conv)} segments) — inspect with --print.")
        w("")
        return
    lo = max(0, ci - WINDOW)
    hi = min(len(conv), ci + WINDOW + 1)
    w(f"```")
    if lo > 0:
        w(f"… ({lo} earlier segments elided)")
    for i in range(lo, hi):
        seg = conv[i]
        is_cand = i == ci
        if is_cand:
            w(">>> CANDIDATE " + "=" * 50)
        w(f"[{i:03d}] {seg.get('role')} ({seg.get('pointer')})")
        lines, trunc = clip(seg.get("content", ""),
                            CAND_MAX_CHARS if is_cand else SEG_MAX_CHARS,
                            CAND_MAX_LINES if is_cand else SEG_MAX_LINES)
        for ln in lines:
            w(("  > " if is_cand else "    ") + ln)
        if trunc:
            w("    …[truncated]")
        if is_cand:
            w(">>> END CANDIDATE " + "=" * 46)
    if hi < len(conv):
        w(f"… ({len(conv) - hi} later segments elided)")
    w("```")
    w("")


def summary_row(rank, qrow, record):
    meta = record.get("meta", {}) if record else {}
    state = parse_state(record) if record else {}
    conv = state.get("conversation", [])
    cand_ptr = qrow.get("candidate_pointer") or state.get("candidate_pointer")
    ci = find_candidate_index(conv, cand_ptr)
    snip = snippet(conv[ci].get("content", "")) if ci is not None else "!!pointer-miss"
    snip = snip.replace("|", "\\|")
    return (f"| {rank} | `{qrow['record_id']}` | {meta.get('candidate_kind','?')} "
            f"| {votes_str(qrow.get('votes', {}))} "
            f"| {qrow.get('n_drop')}/{qrow.get('n_keep')}/{qrow.get('n_uncertain')} "
            f"| {qrow.get('scored_label_flips')} | {qrow.get('noul_spread')} "
            f"| {', '.join(qrow.get('reasons', []))} | {snip} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=DATA_DIR / "review_queue.jsonl")
    ap.add_argument("--candidates", type=Path, default=DATA_DIR / "candidates.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "owner_review.md")
    ap.add_argument("--csv", type=Path,
                    default=DATA_DIR / "owner_labels_template.csv")
    ap.add_argument("--top", type=int, default=60,
                    help="number of queue items fully rendered")
    args = ap.parse_args()

    queue = load_jsonl(args.queue)
    cands = {}
    for rec in load_jsonl(args.candidates):
        cands[rec["meta"]["record_id"]] = rec

    comp = Counter()
    for q in queue:
        for r in q.get("reasons", []):
            comp[r] += 1

    lines = []
    a = lines.append
    a("# Real-Context Eval V1 — Owner Review Bundle")
    a("")
    a("Rendered by `scripts/render_review_queue_v1.py` from "
      f"`{args.queue.name}` ({len(queue)} rows, severity order). "
      "Local-only: contains real transcript text — do not publish.")
    a("")
    a("## How to label")
    a("")
    a("1. Read the label semantics in `docs/REAL_CONTEXT_LABELING_GUIDE_V1.md`: "
      "**yes = certainly irrelevant → drop**, **no = keep** (default when not "
      "certain), **uncertain = excluded from the scored set**.")
    a(f"2. Open `owner_labels_template.csv` ({len(queue)} rows, same severity "
      "order as below) in a spreadsheet editor and fill the `label` column "
      "with `yes` / `no` / `uncertain` (synonyms drop/keep/true/false/u also "
      "accepted). Optionally add a one-line `note` (evidence basis per "
      "guide R8 — short pointer references only, no transcript text).")
    a("3. Fill CSV then run:")
    a("")
    a("```bash")
    a("python3 scripts/csv_to_labels_json.py "
      "data/real_context_eval_v1/owner_labels_template.csv "
      "-o data/real_context_eval_v1/owner_labels.json --with-notes")
    a("python3 scripts/label_real_context_v1.py --first-pass "
      "--subagent-json data/real_context_eval_v1/owner_labels.json "
      "--labeler markus --labeler-type human")
    a("```")
    a("")
    a("(Rows left blank are skipped by the converter; re-run it after filling "
      "more rows — the labeler resumes over already-labeled record_ids.)")
    a("")
    a("## Queue composition")
    a("")
    a(f"- queue size: **{len(queue)}** of {len(cands)} candidates")
    for r, n in comp.most_common():
        a(f"- `{r}`: {n}")
    a(f"- scored-label flips: "
      + ", ".join(f"{k} flips → {v} rows" for k, v in
                  sorted(Counter(q["scored_label_flips"] for q in queue).items())))
    a("")
    n_full = min(args.top, len(queue))
    a(f"## Top {n_full} items (fully rendered)")
    a("")
    a("Candidate marked `>>> CANDIDATE`. Context window: candidate ± "
      f"{WINDOW} segments; the **final user request** line is the judging "
      "anchor (guide §1). If the window is not enough to decide, pull the "
      "full transcript with:")
    a("")
    a("```bash")
    a("python3 scripts/label_real_context_v1.py --first-pass "
      "--record-ids <record_id> --print -")
    a("```")
    a("")

    for rank, qrow in enumerate(queue[:args.top], 1):
        render_item(rank, qrow, cands.get(qrow["record_id"]), lines)

    rest = queue[args.top:]
    if rest:
        a(f"## Remaining {len(rest)} items (summary table)")
        a("")
        a("| rank | record_id | kind | votes | d/k/u | flips | spread | "
          "reasons | candidate snippet |")
        a("|---|---|---|---|---|---|---|---|---|")
        for rank, qrow in enumerate(rest, args.top + 1):
            a(summary_row(rank, qrow, cands.get(qrow["record_id"])))
        a("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["record_id", "label", "note"])
        for qrow in queue:
            wcsv.writerow([qrow["record_id"], "", ""])

    print(f"wrote {args.out} ({n_full} rendered + {len(rest)} summarized)",
          file=sys.stderr)
    print(f"wrote {args.csv} ({len(queue)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
