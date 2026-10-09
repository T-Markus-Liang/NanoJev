#!/usr/bin/env python3
"""Pure-python validation for the v5 F1/F3 corpus outputs
(data/v5_corpus/f1_antishortcut.jsonl, f3_boilerplate.jsonl,
build_report.json).

No model calls, no network — stdlib + build_real_context_eval_v1 helpers
(credential scan / hashing / state normalization), matching the
test_mine_v5_v1.py conventions. Covers:

  1. schema            — valen record shape {group_id, request, targets,
                         meta}; group_id under context_relevance_v5:,
                         record_id prefix v5f1:/v5f3:, binary targets,
                         meta carries pair_id/family/sub_family/
                         length_band/candidate_pointer
  2. instructions hash — questions.irrelevant.type == "noul" and the
                         instructions string is byte-identical to the
                         canonical valen string (sha256 pinned)
  3. pointer/anchor    — candidate_pointer resolves inside the emitted
                         conversation, candidate non-empty <=32KB, last
                         user conv segment == user_messages_in_order[-1]
  4. credential +      — component-level SECRET_PATTERNS scan + no
     placeholder         unfilled <SLOT> markers in emitted content
  5. pair completeness — every pair_id has >=2 members with opposite
                         argmax labels and one consistent pair_kind;
                         kind-specific structure (same_surface -> identical
                         candidate text; anchor_flip -> identical candidate
                         text+pointer; envelope_payload_flip -> same group,
                         2 pointers)
  6. label balance     — F1 keep:drop ~2:1 overall and per-sub-family
                         bounds; F3 exactly 1:1; >=5 domains each
  7. dedup             — request_sha256 + normalized-state hash vs
                         valen_nano_v4 (all splits) and real_context_eval_v1
                         candidates (the v1 builder's dedup approach),
                         plus intra-set uniqueness
  8. report            — build_report.json counts/hashes match the files

Usage: python3 scripts/test_v5_corpus_v1.py  (exit 0 = all checks pass)
"""

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_real_context_eval_v1", ROOT / "scripts" / "build_real_context_eval_v1.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

OUT_DIR = ROOT / "data" / "v5_corpus"
FILES = {
    "f1_antishortcut": OUT_DIR / "f1_antishortcut.jsonl",
    "f3_boilerplate": OUT_DIR / "f3_boilerplate.jsonl",
}
REPORT = OUT_DIR / "build_report.json"
VALEN_V4 = ROOT / "data" / "valen_nano_v4"
EVAL_V1 = [ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl",
           ROOT / "data" / "real_context_eval_v1" /
           "candidates_drop_supp_v1.jsonl"]
CANONICAL_INSTR_SHA = "4659848727be93d683a792b8566b97e82191f114b9aa1944a74c6df05f222b90"
MAX_CANDIDATE_BYTES = 32 * 1024
SLOT_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")

failures = []
checks = Counter()


def check(ok, label, detail=""):
    checks[label] += 1
    if not ok:
        failures.append(f"[{label}] {detail}")


def argmax_label(record):
    p = record["targets"]["irrelevant"]["probabilities"]
    return "true" if p["true"] >= p["false"] else "false"


def load_jsonl(path):
    return [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def reference_hashes():
    """v1-builder dedup approach: request_sha256 + normalized-state hash of
    every prior-corpus record (valen_nano_v4 all splits + real-context eval
    candidates)."""
    req, st = set(), set()
    files = [VALEN_V4 / n for n in ("train.jsonl", "eval.jsonl", "dev.jsonl")]
    files += [p for p in EVAL_V1 if p.is_file()]
    for path in files:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "request" not in rec:
                continue
            req.add(builder.sha256_bytes(
                builder.serialized(rec["request"]).encode("utf-8")))
            sh = builder.normalized_state_hash(rec)
            if sh:
                st.add(sh)
    return req, st


def check_family(family, records, ref_req, ref_st):
    pair_members = defaultdict(list)
    seen_ids, seen_req, seen_st = set(), set(), set()
    per_sub, per_domain = Counter(), Counter()
    label_counts = Counter()
    sub_labels = Counter()

    for r in records:
        meta = r.get("meta", {})
        rid = meta.get("record_id", "?")

        # --- schema -------------------------------------------------------
        check(set(r.keys()) >= {"group_id", "request", "targets", "meta"},
              "schema", f"{rid}: missing top-level keys")
        check(isinstance(r.get("group_id"), str)
              and r["group_id"].startswith("context_relevance_v5:"),
              "schema", f"{rid}: bad group_id")
        check(rid.startswith({"f1_antishortcut": "v5f1:",
                              "f3_boilerplate": "v5f3:"}[family]),
              "schema", f"{rid}: bad record_id prefix for {family}")
        check(rid not in seen_ids, "schema", f"{rid}: duplicate record_id")
        seen_ids.add(rid)
        for field in ("pair_id", "family", "sub_family", "length_band",
                      "candidate_pointer", "candidate_kind", "domain"):
            check(field in meta, "schema", f"{rid}: missing meta.{field}")
        check(meta.get("family") == family, "schema",
              f"{rid}: meta.family={meta.get('family')}")
        check(meta.get("length_band") in ("s", "m", "l", "xl"), "schema",
              f"{rid}: bad length_band")
        probs = (r.get("targets") or {}).get("irrelevant", {}) \
            .get("probabilities", {})
        check(set(probs) == {"true", "false"}
              and all(v in (0.0, 1.0) for v in probs.values())
              and abs(probs["true"] + probs["false"] - 1.0) < 1e-9,
              "schema", f"{rid}: targets not binary")
        label = argmax_label(r)
        label_counts[label] += 1
        sub_labels[(meta.get("sub_family"), label)] += 1
        per_sub[meta.get("sub_family")] += 1
        per_domain[meta.get("domain")] += 1

        q = (r.get("request", {}).get("questions") or {}) \
            .get("irrelevant") or {}
        check(q.get("type") == "noul", "instructions",
              f"{rid}: q type {q.get('type')}")
        check(hashlib.sha256(q.get("instructions", "").encode()).hexdigest()
              == CANONICAL_INSTR_SHA, "instructions",
              f"{rid}: instructions not verbatim canonical")

        try:
            state = json.loads(r["request"]["state"])
        except (ValueError, KeyError, TypeError):
            check(False, "schema", f"{rid}: unparseable state")
            continue
        conv = state.get("conversation") or []
        users = state.get("user_messages_in_order") or []
        ptr = state.get("candidate_pointer")

        # --- dedup ----------------------------------------------------------
        rhash = builder.sha256_bytes(
            builder.serialized(r["request"]).encode("utf-8"))
        shash = builder.normalized_state_hash(r)
        check(rhash not in ref_req and shash not in ref_st, "dedup_v4",
              f"{rid}: duplicate of a valen_nano_v4/real-context record")
        check(rhash not in seen_req and shash not in seen_st, "dedup_intra",
              f"{rid}: intra-set duplicate")
        seen_req.add(rhash)
        seen_st.add(shash)

        # --- pointer/anchor -------------------------------------------------
        ptrs = [s.get("pointer") for s in conv]
        check(ptr in ptrs, "pointer", f"{rid}: {ptr} not in conversation")
        check(len(ptrs) == len(set(ptrs)), "pointer",
              f"{rid}: duplicate conv pointers")
        cand = next((s for s in conv if s.get("pointer") == ptr), None)
        if cand is not None:
            check(bool(cand.get("content", "").strip()), "pointer",
                  f"{rid}: empty candidate")
            check(len(cand["content"].encode()) <= MAX_CANDIDATE_BYTES,
                  "pointer", f"{rid}: candidate over 32KB")
            check(cand.get("role") in ("user", "assistant", "tool"),
                  "pointer", f"{rid}: candidate is control/system")
        check(ptr == meta.get("candidate_pointer"), "pointer",
              f"{rid}: state pointer != meta")
        check(bool(users), "anchor", f"{rid}: empty user_messages_in_order")
        user_segs = [s for s in conv if s.get("role") == "user"]
        check(bool(user_segs), "anchor", f"{rid}: no user seg in conv")
        if user_segs and users:
            check(user_segs[-1].get("content") == users[-1], "anchor",
                  f"{rid}: last user seg != user_messages[-1]")
            check(cand.get("pointer") != user_segs[-1].get("pointer"),
                  "anchor", f"{rid}: candidate is the final anchor")

        # --- credential + placeholder scan -----------------------------------
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            check(hit is None, "credential",
                  f"{rid}: {hit} in conv {s.get('pointer')}")
            check(not SLOT_RE.search(s.get("content", "")), "placeholder",
                  f"{rid}: unfilled slot in {s.get('pointer')}")
        for u in users:
            check(builder.credential_scan(u) is None, "credential",
                  f"{rid}: credential in user_messages_in_order")
            check(not SLOT_RE.search(u), "placeholder",
                  f"{rid}: unfilled slot in user message")

        if meta.get("pair_id"):
            pair_members[meta["pair_id"]].append(r)

    # --- pair completeness ----------------------------------------------------
    for pid, members in sorted(pair_members.items()):
        labels = {argmax_label(m) for m in members}
        check(labels == {"true", "false"}, "pairs",
              f"{pid}: labels {sorted(labels)} not opposite")
        kinds = {m["meta"].get("pair_kind") for m in members}
        check(len(kinds) == 1, "pairs", f"{pid}: mixed pair_kind {kinds}")
        if kinds == {"same_surface"} or kinds == {"anchor_flip"}:
            texts = set()
            for m in members:
                st = json.loads(m["request"]["state"])
                cp = st["candidate_pointer"]
                c = next((s for s in st["conversation"]
                          if s["pointer"] == cp), None)
                texts.add(c["content"] if c else None)
            check(len(texts) == 1, "pairs",
                  f"{pid}: {next(iter(kinds))} pair with different "
                  "candidate text")
        if kinds == {"anchor_flip"}:
            check(len({m["meta"]["candidate_pointer"] for m in members})
                  == 1, "pairs",
                  f"{pid}: anchor_flip different candidate_pointer")
        if kinds == {"envelope_payload_flip"}:
            check(len({m["group_id"] for m in members}) == 1, "pairs",
                  f"{pid}: payload_flip must share one group")
            check(len({m["meta"]["candidate_pointer"] for m in members})
                  == 2, "pairs",
                  f"{pid}: payload_flip needs 2 candidate_pointers")
        if kinds == {"anchor_flip"} and family == "f3_boilerplate":
            check(len({m["group_id"] for m in members}) == 2, "pairs",
                  f"{pid}: f3 anchor_flip must span 2 groups")

    # --- label balance ----------------------------------------------------------
    keep, drop = label_counts["false"], label_counts["true"]
    if family == "f1_antishortcut":
        ratio = keep / max(1, drop)
        check(1.5 <= ratio <= 2.5, "balance",
              f"f1 keep:drop {keep}:{drop} = {ratio:.2f}, want ~2:1")
        for sub in ("empty_result_is_answer", "correction_keep",
                    "short_ack_with_constraint"):
            k = sub_labels.get((sub, "false"), 0)
            d = sub_labels.get((sub, "true"), 0)
            check(per_sub.get(sub, 0) > 0, "balance", f"{sub}: missing")
            check(k > d > 0 or (sub == "correction_keep" and k > 0),
                  "balance",
                  f"{sub}: keep={k} drop={d} (expect keep-heavy with drops)")
        check(sub_labels.get(("empty_result_is_answer", "true"), 0) > 0,
              "balance", "empty_result_is_answer needs drop members")
        check(sub_labels.get(("short_ack_with_constraint", "true"), 0) > 0,
              "balance", "short_ack needs constraint-free drop members")
    if family == "f3_boilerplate":
        check(keep == drop, "balance",
              f"f3 keep {keep} != drop {drop} (want 1:1 pairs)")
        check(len(pair_members) >= 600, "balance",
              f"f3 pairs {len(pair_members)} < 600")
        check(all(len(m) == 2 for m in pair_members.values()), "pairs",
              "f3: every pair must have exactly 2 members")
        dialects = {m.get("envelope_dialect") for m in
                    (r["meta"] for r in records)}
        check("claude" in dialects and "codex" in dialects, "balance",
              f"f3 missing a harness dialect: {dialects}")
    check(len(per_domain) >= 5, "balance",
          f"{family}: only {len(per_domain)} domains (<5)")

    return {"labels": dict(label_counts), "sub_families": dict(per_sub),
            "domains": dict(per_domain), "pairs": len(pair_members)}


def main():
    ref_req, ref_st = reference_hashes()
    summaries = {}
    for family, path in FILES.items():
        check(path.is_file(), "files", f"{path} missing")
        if not path.is_file():
            continue
        records = load_jsonl(path)
        check(1300 <= len(records) <= 1700, "files",
              f"{family}: {len(records)} records, want ~1500")
        summaries[family] = check_family(family, records, ref_req, ref_st)

    # --- report consistency -----------------------------------------------------
    check(REPORT.is_file(), "report", "build_report.json missing")
    if REPORT.is_file():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        check(report.get("seed") == 20261201, "report", "seed mismatch")
        check(report.get("instructions_sha256") == CANONICAL_INSTR_SHA,
              "report", "report instructions hash mismatch")
        fams = report.get("families", {})
        for family, path in FILES.items():
            if not path.is_file() or family not in fams:
                check(family in fams, "report",
                      f"{family} missing from report")
                continue
            sec = fams[family]
            check(sec.get("records") == len(load_jsonl(path)),
                  "report", f"{family}: report records mismatch")
            check(sec.get("sha256") == builder.sha256_file(path),
                  "report", f"{family}: report sha256 mismatch")
            check(sec.get("self_check", {}).get("status") == "pass",
                  "report", f"{family}: builder self_check not pass")
            check(sec.get("labels") == summaries[family]["labels"],
                  "report", f"{family}: report labels mismatch")

    total = sum(checks.values())
    print(f"{total} checks, {len(failures)} failures")
    for family, summ in summaries.items():
        print(f"  {family}: {summ}")
    for f in failures[:30]:
        print("  FAIL", f)
    if failures:
        print(f"... {len(failures)} total failures")
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
