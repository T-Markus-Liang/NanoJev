#!/usr/bin/env python3
"""Convert a filled owner-label CSV into the JSON object accepted by
scripts/label_real_context_v1.py --subagent-json:

    {record_id: true | false | "uncertain"}

Input CSV columns: record_id,label,note (as emitted by
render_review_queue_v1.py -> owner_labels_template.csv).

Label cell values (case-insensitive):
    drop side : yes, y, d, drop, true, 1, irrelevant   -> true
    keep side : no, n, k, keep, false, 0, relevant     -> false
    abstain   : uncertain, u, unsure, ?, skip          -> "uncertain"
    empty     : row skipped (warning to stderr)

With --with-notes, a non-empty `note` column turns the value into
{"label": ..., "evidence": note} — also accepted by --subagent-json
(guide R8 evidence basis).

Usage:
    python3 scripts/csv_to_labels_json.py owner_labels_template.csv \
        -o owner_labels.json --with-notes
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DROP_WORDS = {"yes", "y", "d", "drop", "true", "1", "irrelevant"}
KEEP_WORDS = {"no", "n", "k", "keep", "false", "0", "relevant"}
UNC_WORDS = {"uncertain", "u", "unsure", "?", "skip"}


def map_label(raw, where):
    v = raw.strip().lower()
    if v in DROP_WORDS:
        return True
    if v in KEEP_WORDS:
        return False
    if v in UNC_WORDS:
        return "uncertain"
    raise SystemExit(f"{where}: unrecognized label {raw!r}; expected "
                     "yes/no/uncertain (or drop/keep/true/false/u)")


def find_col(fieldnames, *prefixes):
    for name in fieldnames or []:
        low = name.strip().lower()
        if any(low.startswith(p) for p in prefixes):
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path, help="filled owner-label CSV")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output JSON path (default: stdout)")
    ap.add_argument("--with-notes", action="store_true",
                    help="emit {record_id: {label, evidence}} when the "
                         "note column is non-empty")
    ap.add_argument("--require-all", action="store_true",
                    help="error out if any row has an empty label")
    args = ap.parse_args()

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rid_col = find_col(reader.fieldnames, "record_id")
        lbl_col = find_col(reader.fieldnames, "label")
        note_col = find_col(reader.fieldnames, "note", "evidence")
        if not rid_col or not lbl_col:
            raise SystemExit(f"{args.csv_path}: need columns "
                             "record_id,label[,note]; "
                             f"found {reader.fieldnames}")
        labels = {}
        n_empty = 0
        for i, row in enumerate(reader, 2):
            rid = (row.get(rid_col) or "").strip()
            if not rid:
                continue
            raw = (row.get(lbl_col) or "").strip()
            note = (row.get(note_col) or "").strip() if note_col else ""
            if not raw:
                n_empty += 1
                if args.require_all:
                    raise SystemExit(
                        f"{args.csv_path}:{i}: {rid}: empty label "
                        "(--require-all)")
                continue
            if rid in labels:
                raise SystemExit(f"{args.csv_path}:{i}: duplicate "
                                 f"record_id {rid}")
            lab = map_label(raw, f"{args.csv_path}:{i}:{rid}")
            if args.with_notes and note:
                labels[rid] = {"label": lab, "evidence": note}
            else:
                labels[rid] = lab

    payload = json.dumps(labels, ensure_ascii=False, indent=1,
                         sort_keys=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}: {len(labels)} labels "
              f"({n_empty} rows with empty label skipped)", file=sys.stderr)
    else:
        sys.stdout.write(payload)
        print(f"{len(labels)} labels ({n_empty} empty-label rows skipped)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
