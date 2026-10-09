#!/usr/bin/env python3
"""V8 round-2 mined-candidate pool for agent adjudication
(docs/DROP_LABEL_DILEMMA_V1.md Path A: "mine liberally, adjudicate each
with the main model").

This is a LOOSER-EMISSION re-run of scripts/mine_v5_hard_negatives_v1.py
(f2). The v5 miner has no CLI knobs — its emission caps are module
constants — so this script imports it, relaxes ONLY the emission caps, and
reuses every strategy, guard, windowing rule, and self-check unchanged:

  * Pool-exclusion integrity is pinned to the RECORDED ledger
    data/v5_mining/pool_split.json (written from f2_report.json):
      - the 60 eval_v1_frozen transcript hashes are NEVER mined;
      - the 18 v5_eval_reserved transcript hashes are NEVER mined.
    We do NOT re-run split_pools(): re-running top-K selection over the
    grown file set could silently rotate the reserve. The ledger is the
    contract.

  * Transcript growth since the f2 run (cutoff = mtime of the f2 output):
      - unchanged file whose hash is in the recorded train pool -> re-mined
        with the relaxed caps;
      - file created after the cutoff (birthtime > cutoff) -> genuinely new
        transcript -> mined under the same rules;
      - file that EXISTED at the cutoff but changed since (birthtime <=
        cutoff < mtime) and whose current hash is not in ANY recorded pool
        -> excluded as `grown_unknown_pool`: a grown eval/reserved
        transcript would have a new hash and be unrecognizable, so
        membership is unprovable and we stay conservative. (One such file
        this run; costs at most one transcript of yield, prevents any
        eval/reserve leak.)

  * f2 non-overlap (the adjudication queue must not see the same row
    twice): proposals whose (transcript_hash, candidate_seg, anchor_seg)
    already appears in f2 are skipped before emit; f2 record_ids, request
    hashes and normalized-state hashes join the dedup reference set (f4
    outcome-positive candidates are added too — same transcripts, same
    question shape). Record ids use the `v8r2:` prefix and pair ids
    `v8r2pair:` so collisions with `v5f2:*`/`v5f2pair:*` are impossible by
    construction; self_check still verifies all three overlap counters are
    zero.

  * UNCHANGED (never weakened): credential-scan patterns and enforcement
    points (flagged segment never a candidate / never in a window; flagged
    user message disqualifies anchors at/after it; component-level scan of
    every emitted state + self_check re-scan), <=32KB candidate cap,
    STATE_CHAR_BUDGET/HARD windowing budgets, extraction thresholds,
    target-null proposal-only semantics, seed determinism.

Output: data/v8_mining_round2/candidates.jsonl
        data/v8_mining_round2/report.json

Usage: python3 scripts/mine_v8_candidates_v1.py
"""

import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m5 = _load_module("mine_v5_hard_negatives_v1", "mine_v5_hard_negatives_v1.py")
builder = m5.builder

OUT_DIR = ROOT / "data" / "v8_mining_round2"
OUT_JSONL = OUT_DIR / "candidates.jsonl"
OUT_REPORT = OUT_DIR / "report.json"
F2_JSONL = ROOT / "data" / "v5_mining" / "f2_hard_negative_candidates.jsonl"
F4_JSONL = ROOT / "data" / "v5_mining" / "f4_outcome_positive_candidates.jsonl"
POOL_SPLIT_JSON = ROOT / "data" / "v5_mining" / "pool_split.json"

RID_PREFIX = "v8r2"
PAIR_PREFIX = "v8r2pair"
SEED = m5.SEED  # 20261201 — unchanged; determinism preserved
GROUP_LINEAGE = m5.GROUP_LINEAGE

Y, N, U = m5.Y, m5.N, m5.U
MAX_CANDIDATE_BYTES = m5.MAX_CANDIDATE_BYTES
STATE_CHAR_HARD = m5.STATE_CHAR_HARD

# --- relaxed emission caps (the ONLY thing loosened) --------------------------
# f2 run: per_transcript_cap alone discarded 2,646 generated proposals, and
# the cap ordering kept keep-side first, so the discarded tail was mostly
# drop-side. These relaxations recover that tail plus a wider per-boundary
# spread. All are emission limits, not evidence thresholds — guards and
# label rules are untouched.
RELAXED_CAPS = {
    "MAX_PER_BOUNDARY": 16,        # was 8
    "MAX_DEEP_PER_BOUNDARY": 6,    # was 3
    "MAX_DEEP_REANCHOR": 20,       # was 10
    "MAX_STALE_PER_TRANSCRIPT": 15,  # was 6
    "MAX_DUP_GROUPS": 8,           # was 4
    "MAX_DUP_PER_GROUP": 4,        # was 3
    "MAX_FOREIGN_TAIL": 16,        # was 8
    "MAX_KEEP_SIDE": 25,           # was 12
    "MAX_PER_TRANSCRIPT": 500,     # was 50 — the dominant limiter in f2
}
for _k, _v in RELAXED_CAPS.items():
    setattr(m5, _k, _v)


def _ptr_idx(pointer):
    # "/messages/{idx}/content" -> int idx
    return int(pointer.split("/")[2])


def load_f2_overlap():
    """Everything needed to guarantee zero overlap with the f2 file:
    record_ids, (transcript_hash, seg_idx, anchor_seg_idx) proposal keys,
    request sha256s, normalized state hashes."""
    ids, keys, req_hashes, state_hashes = set(), set(), set(), set()
    for line in F2_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec["meta"]
        ids.add(meta["record_id"])
        keys.add((meta["transcript_hash"],
                  _ptr_idx(meta["candidate_pointer"]),
                  _ptr_idx(meta["anchor_pointer"])))
        req_hashes.add(builder.sha256_bytes(
            builder.serialized(rec["request"]).encode("utf-8")))
        sh = builder.normalized_state_hash(rec)
        if sh:
            state_hashes.add(sh)
    return ids, keys, req_hashes, state_hashes


def load_prior_hashes(path):
    req_hashes, state_hashes = set(), set()
    if not path.is_file():
        return req_hashes, state_hashes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "request" not in rec:
            continue
        req_hashes.add(builder.sha256_bytes(
            builder.serialized(rec["request"]).encode("utf-8")))
        sh = builder.normalized_state_hash(rec)
        if sh:
            state_hashes.add(sh)
    return req_hashes, state_hashes


def classify_pools(usable, ledger, cutoff):
    """Pin exclusion to the recorded pool_split ledger + file birthtime.

    Returns (mineable, ledger_counts). Each mineable item gets
    item["v8_pool_role"] in {remine_train, new_transcript}.
    """
    frozen = set(ledger["eval_v1_frozen_hashes"])
    reserved = set(ledger["eval_pool_hashes"])
    train = set(ledger["train_pool_hashes"])
    counts = Counter()
    mineable = []
    for item in usable:
        th = item["transcript_hash"]
        if th in frozen:
            counts["eval_v1_frozen_seen_excluded"] += 1
            continue
        if th in reserved:
            counts["v5_eval_reserved_seen_excluded"] += 1
            continue
        if th in train:
            item["v8_pool_role"] = "remine_train"
            counts["train_pool_remined"] += 1
            mineable.append(item)
            continue
        # unknown hash: new file, grown file, or previously-unusable file
        st = os.stat(item["path"])
        birth = getattr(st, "st_birthtime", st.st_mtime)
        if birth > cutoff:
            item["v8_pool_role"] = "new_transcript"
            counts["new_transcripts_mined"] += 1
            mineable.append(item)
        elif st.st_mtime > cutoff:
            counts["grown_unknown_pool_excluded"] += 1
        else:
            # unchanged bytes but no recorded membership: should be
            # impossible for a usable transcript (it would have been
            # assigned a pool last run) — exclude conservatively anyway.
            counts["unchanged_unknown_pool_excluded"] += 1
    counts["train_pool_recorded"] = len(train)
    counts["train_pool_missing_files"] = len(
        train - {u["transcript_hash"] for u in usable})
    counts["frozen_recorded"] = len(frozen)
    counts["reserved_recorded"] = len(reserved)
    return mineable, counts


def emit_v8(item, prop, instructions, used_ids, f2, skips):
    """emit() identical to m5.emit except: f2 proposal-key pre-filter and
    the v8r2 record_id prefix. Every guard is byte-for-byte the same."""
    th, t, src = item["transcript_hash"], item["transcript"], item["source"]
    segs = t.segments
    idx, a_seg, a_uid = prop["seg"], prop["anchor_seg"], prop["anchor_uidx"]
    if (th, idx, a_seg) in f2["keys"]:
        skips["already_in_f2"] += 1
        return None
    if idx == a_seg or idx >= prop["cut"]:
        skips["candidate_out_of_scope"] += 1
        return None
    seg = segs[idx]
    if not m5.eligible_mine_kinds(seg):
        skips["bad_kind"] += 1
        return None
    if idx in item["flagged"]:
        skips["cred_candidate_blocked"] += 1
        return None
    if len(seg["text"].encode("utf-8", errors="ignore")) > MAX_CANDIDATE_BYTES:
        skips["candidate_over_32kb"] += 1
        return None
    ptr = f"/messages/{idx}/content"
    anchor_ptr = f"/messages/{a_seg}/content"
    prefix = segs[:prop["cut"]]
    user_list = t.users[:a_uid + 1]
    conv, windowed = m5.windowed_conversation(
        prefix, idx, a_seg, user_list, blocked=item["flagged"])
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": user_list}
    state_str = builder.serialized(state)
    if len(state_str.encode("utf-8")) > STATE_CHAR_HARD:
        skips["exceeds_budget"] += 1
        return None
    # emit-level credential gate — identical to f2: scan components, not
    # the serialized blob.
    cred_parts = [s["content"] for s in conv] + list(user_list)
    if any(builder.credential_scan(part) for part in cred_parts):
        skips["cred_state_blocked"] += 1
        return None
    rid = f"{RID_PREFIX}:{th[:12]}:seg{idx:04d}:a{a_seg:04d}"
    if rid in used_ids:
        rid = f"{rid}-{prop['strategy'][:4]}"
    used_ids.add(rid)
    meta = {
        "record_id": rid,
        "domain": "real",
        "modality": "text",
        "language_bucket": builder.language_bucket(t),
        "source": "real_transcript",
        "source_dataset": GROUP_LINEAGE,
        "source_root": src,
        "transcript_hash": th,
        "candidate_kind": seg["kind"],
        "candidate_pointer": ptr,
        "length_band": builder.length_band(len(segs)),
        "selection_method": f"mining:{prop['strategy']}:{prop['side']}",
        "supp_source": "mining",
        "mine_mode": prop["strategy"],
        "strategy": prop["strategy"],
        "mining_side": prop["side"],
        "anchor_user_index": a_uid,
        "anchor_pointer": anchor_ptr,
        "windowed": windowed,
        "state_chars": len(state_str.encode("utf-8")),
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "evidence_basis": prop["note"],
        "pair_id": None,
        "pool": "train",
    }
    if prop.get("evidence"):
        meta.update(prop["evidence"])
    meta.update(prop.get("extra", {}))
    return {
        "group_id": f"{GROUP_LINEAGE}:{th}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {"type": "noul",
                                               "instructions": instructions}}},
        "targets": None,
        "meta": meta,
    }


def assign_pairs_v8(emitted, th):
    """m5.assign_pairs with the v8r2pair: prefix (logic unchanged)."""
    by_seg = defaultdict(list)
    for rec, prop in emitted:
        by_seg[prop["seg"]].append((rec, prop))
    for seg, members in sorted(by_seg.items()):
        anchors = {p["anchor_seg"] for _, p in members}
        labels = {p["label"] for _, p in members}
        if len(anchors) >= 2 and Y in labels and N in labels:
            pid = f"{PAIR_PREFIX}:{th[:12]}:flip:seg{seg:04d}"
            for rec, p in members:
                if p["label"] in (Y, N):
                    rec["meta"]["pair_id"] = pid
                    rec["meta"]["pair_kind"] = "anchor_flip"
    by_anchor = defaultdict(list)
    for rec, prop in emitted:
        by_anchor[prop["anchor_seg"]].append((rec, prop))
    for a_seg, members in sorted(by_anchor.items()):
        free = [(rec, p) for rec, p in members
                if not rec["meta"]["pair_id"] and p["label"] in (Y, N)]
        if len(free) >= 2 and {p["label"] for _, p in free} == {Y, N}:
            pid = f"{PAIR_PREFIX}:{th[:12]}:same:a{a_seg:04d}"
            for rec, p in free:
                rec["meta"]["pair_id"] = pid
                rec["meta"]["pair_kind"] = "same_anchor"


def self_check_v8(records, frozen, reserved, f2):
    """m5.self_check + round-2 overlap/pool assertions. Raises on any error."""
    errors = []
    seen_ids = set()
    for r in records:
        rid = r["meta"]["record_id"]
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate record_id")
        seen_ids.add(rid)
        try:
            state = json.loads(r["request"]["state"])
        except ValueError:
            errors.append(f"{rid}: unparseable state")
            continue
        ptr = state.get("candidate_pointer")
        conv = state.get("conversation", [])
        ptrs = {s.get("pointer") for s in conv}
        if ptr not in ptrs:
            errors.append(f"{rid}: candidate_pointer {ptr} not in conv")
        cand = next((s for s in conv if s.get("pointer") == ptr), None)
        if cand is not None and not cand.get("content", "").strip():
            errors.append(f"{rid}: empty candidate content")
        if cand is not None and \
                len(cand["content"].encode("utf-8")) > MAX_CANDIDATE_BYTES:
            errors.append(f"{rid}: candidate over 32KB")
        users = state.get("user_messages_in_order") or []
        if not users:
            errors.append(f"{rid}: empty user_messages_in_order")
            continue
        user_segs = [s for s in conv if s.get("role") == "user"]
        if not user_segs:
            errors.append(f"{rid}: no user seg in conv")
            continue
        last_user = user_segs[-1]
        if last_user.get("pointer") != r["meta"].get("anchor_pointer"):
            errors.append(f"{rid}: last user conv seg "
                          f"{last_user.get('pointer')} != anchor "
                          f"{r['meta'].get('anchor_pointer')}")
        if last_user.get("content") != users[-1]:
            errors.append(f"{rid}: last user seg content != "
                          f"user_messages_in_order[-1]")
        # component-level scan — identical to f2
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in conv "
                              f"{s.get('pointer')}")
                break
        for u in users:
            hit = builder.credential_scan(u)
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in "
                              f"user_messages_in_order")
                break
        th = r["meta"]["transcript_hash"]
        if th in frozen:
            errors.append(f"{rid}: transcript in eval_v1 pool")
        if th in reserved:
            errors.append(f"{rid}: transcript in v5_eval_reserved pool")
        if not rid.startswith(f"{RID_PREFIX}:"):
            errors.append(f"{rid}: bad record_id prefix")
        if rid in f2["ids"]:
            errors.append(f"{rid}: record_id collides with f2")
        key = (th, _ptr_idx(ptr), _ptr_idx(r["meta"]["anchor_pointer"]))
        if key in f2["keys"]:
            errors.append(f"{rid}: (transcript, seg, anchor) already in f2")
        if r["targets"] is not None:
            errors.append(f"{rid}: targets must stay null (proposals only)")
    if errors:
        raise AssertionError(f"self_check failed ({len(errors)}): "
                             + "; ".join(errors[:10]))
    return {"records_checked": len(records), "status": "pass"}


def main():
    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(ROOT / "data" / "valen_nano_v3")
    ledger = json.loads(POOL_SPLIT_JSON.read_text(encoding="utf-8"))
    frozen = set(ledger["eval_v1_frozen_hashes"])
    reserved = set(ledger["eval_pool_hashes"])
    cutoff = os.path.getmtime(F2_JSONL)

    f2_ids, f2_keys, f2_req, f2_state = load_f2_overlap()
    f2 = {"ids": f2_ids, "keys": f2_keys}
    ref_requests, ref_states = m5.load_reference_hashes()
    f4_req, f4_state = load_prior_hashes(F4_JSONL)
    ref_requests |= f2_req | f4_req
    ref_states |= f2_state | f4_state

    usable, drops, scanned = m5.scan_transcripts()
    mineable, pool_counts = classify_pools(usable, ledger, cutoff)

    records, report_rows, skips = [], [], Counter()
    used_ids = set()
    seen_requests, seen_states = set(), set()
    dedup = Counter()
    for item in mineable:
        props = m5.mine_transcript(item, skips)
        emitted = []
        for prop in props:
            rec = emit_v8(item, prop, instructions, used_ids, f2, skips)
            if rec is None:
                continue
            rhash = builder.sha256_bytes(
                builder.serialized(rec["request"]).encode("utf-8"))
            shash = builder.normalized_state_hash(rec)
            if rhash in ref_requests or shash in ref_states:
                dedup["vs_prior_corpora"] += 1
                continue
            if rhash in seen_requests or shash in seen_states:
                dedup["intra_set"] += 1
                continue
            seen_requests.add(rhash)
            seen_states.add(shash)
            emitted.append((rec, prop))
        assign_pairs_v8(emitted, item["transcript_hash"])
        for rec, prop in emitted:
            records.append(rec)
            report_rows.append({
                "record_id": rec["meta"]["record_id"],
                "transcript": item["transcript_hash"][:12],
                "source": item["source"],
                "pool_role": item["v8_pool_role"],
                "strategy": prop["strategy"],
                "side": prop["side"],
                "seg": prop["seg"],
                "anchor_seg": prop["anchor_seg"],
                "kind": rec["meta"]["candidate_kind"],
                "proposed_label": prop["label"],
                "pair_id": rec["meta"]["pair_id"],
                "note": prop["note"],
                "candidate_snippet": m5.norm(
                    item["transcript"].segments[prop["seg"]]["text"])[:140],
                "anchor_snippet": m5.norm(
                    item["transcript"].users[prop["anchor_uidx"]])[:140],
            })

    records.sort(key=lambda r: (r["meta"]["transcript_hash"],
                                r["meta"]["record_id"]))

    check = self_check_v8(records, frozen, reserved, f2)

    # independent overlap verification vs f2 (all three must be zero)
    our_ids = {r["meta"]["record_id"] for r in records}
    our_keys = {(r["meta"]["transcript_hash"],
                 _ptr_idx(r["meta"]["candidate_pointer"]),
                 _ptr_idx(r["meta"]["anchor_pointer"])) for r in records}
    our_req = {builder.sha256_bytes(
        builder.serialized(r["request"]).encode("utf-8")) for r in records}
    overlap = {
        "record_id_overlap": len(our_ids & f2_ids),
        "proposal_key_overlap": len(our_keys & f2_keys),
        "request_hash_overlap": len(our_req & f2_req),
    }
    if any(overlap.values()):
        raise AssertionError(f"f2 overlap detected: {overlap}")

    label_counts = Counter(r["meta"]["proposed_label"] for r in records)
    strat_counts = Counter(r["meta"]["strategy"] for r in records)
    strat_side = Counter((r["meta"]["strategy"], r["meta"]["mining_side"])
                         for r in records)
    strat_label = Counter((r["meta"]["strategy"], r["meta"]["proposed_label"])
                          for r in records)
    side_counts = Counter(r["meta"]["mining_side"] for r in records)
    drop_side = side_counts.get("drop", 0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(builder.serialized(r) + "\n")

    sample_idx = sorted(rng.sample(range(len(report_rows)),
                                   min(20, len(report_rows))))
    top20 = [report_rows[i] for i in sample_idx]

    report = {
        "schema_version": "nanojev-v8-mining-round2-v1",
        "builder": "scripts/mine_v8_candidates_v1.py",
        "base_miner": "scripts/mine_v5_hard_negatives_v1.py",
        "seed": SEED,
        "purpose": ("DROP_LABEL_DILEMMA_V1 Path A: larger mined-candidate "
                    "pool for the agent adjudication queue. PROPOSALS ONLY: "
                    "targets stay null; proposed_label is a deterministic "
                    "suggestion pending adjudication."),
        "emission_caps": {"v8r2_relaxed": dict(RELAXED_CAPS),
                          "f2_reference": {
            "MAX_PER_BOUNDARY": 8, "MAX_DEEP_PER_BOUNDARY": 3,
            "MAX_DEEP_REANCHOR": 10, "MAX_STALE_PER_TRANSCRIPT": 6,
            "MAX_DUP_GROUPS": 4, "MAX_DUP_PER_GROUP": 3,
            "MAX_FOREIGN_TAIL": 8, "MAX_KEEP_SIDE": 12,
            "MAX_PER_TRANSCRIPT": 50}},
        "pool_integrity": {
            "rule": ("exclusions pinned to data/v5_mining/pool_split.json "
                     "ledger (eval_v1_frozen + v5_eval_reserved NEVER "
                     "mined); split NOT recomputed over the grown file "
                     "set; files changed since the f2 cutoff whose current "
                     "hash is in no recorded pool are excluded as "
                     "grown_unknown_pool"),
            "cutoff": cutoff,
            "frozen_hashes_mined": sum(
                1 for r in records
                if r["meta"]["transcript_hash"] in frozen),
            "reserved_hashes_mined": sum(
                1 for r in records
                if r["meta"]["transcript_hash"] in reserved),
            **dict(sorted(pool_counts.items())),
        },
        "overlap_vs_f2": overlap,
        "dedup": {
            "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3",
                               "data/valen_nano_v4",
                               "data/real_context_eval_v1/candidates*.jsonl",
                               "data/v5_mining/f2_hard_negative_candidates.jsonl",
                               "data/v5_mining/f4_outcome_positive_candidates.jsonl"],
            **dict(sorted(dedup.items())),
        },
        "self_check": check,
        "counts": {
            "files_scanned": scanned,
            "usable_transcripts": len(usable),
            "transcripts_mined": len(mineable),
            "unusable_drops": dict(sorted(drops.items())),
            "records": len(records),
            "proposed_labels": dict(sorted(label_counts.items())),
            "by_mining_side": dict(sorted(side_counts.items())),
            "drop_side_records": drop_side,
            "by_strategy": dict(sorted(strat_counts.items())),
            "by_strategy_side": {f"{k[0]}:{k[1]}": v
                                 for k, v in sorted(strat_side.items())},
            "by_strategy_label": {f"{k[0]}:{k[1]}": v
                                  for k, v in sorted(strat_label.items())},
            "by_kind": dict(Counter(r["meta"]["candidate_kind"]
                                    for r in records)),
            "by_source": dict(Counter(r["meta"]["source_root"]
                                      for r in records)),
            "by_pool_role": dict(Counter(
                row["pool_role"] for row in report_rows)),
            "windowed": sum(1 for r in records if r["meta"]["windowed"]),
            "paired_records": sum(1 for r in records
                                  if r["meta"]["pair_id"]),
            "pairs": len({r["meta"]["pair_id"] for r in records
                          if r["meta"]["pair_id"]}),
            "skips": dict(sorted(skips.items())),
        },
        "spot_review_sample_20": top20,
        "rows": report_rows,
        "caveats": [
            ("all guards identical to f2: credential scan per segment/user "
             "message + component scan of every emitted state + self_check "
             "re-scan; <=32KB candidate cap; STATE_CHAR_BUDGET/HARD "
             "windowing unchanged. ONLY emission caps were relaxed."),
            ("f2 non-overlap enforced at proposal key "
             "(transcript_hash, seg, anchor_seg), record_id and request-"
             "hash level; self_check asserts all three overlap counters "
             "are 0. Record ids are namespaced v8r2:."),
            ("pool exclusions are pinned to the recorded ledger rather "
             "than recomputed: re-running top-K reserve selection over "
             "grown files could rotate the reserve and leak an eval "
             "transcript into train."),
            ("grown_unknown_pool files existed at the f2 cutoff and "
             "changed since, with no recorded pool membership: possibly "
             "appended sessions whose earlier state was frozen/reserved. "
             "Excluded conservatively — cost is bounded yield, never a "
             "leak."),
            ("proposals remain deterministic heuristics for the "
             "adjudication queue, not labels."),
        ],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"scanned {scanned} files -> {len(usable)} usable transcripts")
    print("pool:", dict(sorted(pool_counts.items())))
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print(f"drop-side: {drop_side} | labels:",
          dict(sorted(label_counts.items())),
          "| strategies:", dict(sorted(strat_counts.items())))
    print("overlap vs f2:", overlap)
    print("skips:", dict(sorted(skips.items())),
          "| drops:", dict(sorted(drops.items())),
          "| dedup:", dict(sorted(dedup.items())))
    print(f"report -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
