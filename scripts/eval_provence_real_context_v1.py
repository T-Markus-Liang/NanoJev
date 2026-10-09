#!/usr/bin/env python3
"""Run the PUBLISHED OpenProvence reranker (inference only, released weights)
on the real-context eval as an external-candidate baseline.

Model: ``hotchpotch/open-provence-reranker-v1-gte-modernbert-base``
  - MIT license, gte-modernbert-base encoder (149M, English-only).
  - Provence-style dual head: per-sentence keep probability + a sigmoid
    reranking score for (question, context).

Mapping (per docs/PROVENCE_REPRO_PREP_V1.md §4):
  - question  := last user message (``user_messages_in_order[-1]``)
  - context   := candidate segment text (the conversation entry whose
                 ``pointer`` == ``candidate_pointer``)
  - provence_score := max per-sentence keep-probability of the candidate
                 (a segment counts as relevant if any sentence survives)
  - mapped_noul    := 1 - provence_score  (P(irrelevant))

Variant B (``*_ctx`` fields) feeds ``[prev_msg, candidate, next_msg]`` as
pre-split pseudo-sentences so the candidate is judged with ±1 neighbour
context; the score is the max keep-prob over chunks owned by the candidate.

transformers 5.4.0 compatibility: the released ``modeling_open_provence_
standalone.py`` targets transformers 4.57; two tokenizer methods removed in
v5 are shimmed (``build_inputs_with_special_tokens``,
``create_token_type_ids_from_sequences``) and ``all_tied_weights_keys`` is
computed lazily because the remote class never calls ``post_init()``.
Shims reproduce the v4 behaviour exactly: ``[CLS] q [SEP] c [SEP]`` input
layout (verified against ``tokenizer(q, c)`` pair-encoding) and all-zero
token_type_ids (ModernBERT has no token-type embeddings).

Deps: needs ``nltk`` (+punkt_tab data) which is NOT in the valen venv —
install to a target dir and pass PYTHONPATH, e.g.
    python3 -m pip install --target /tmp/provence_pydeps nltk
    NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('punkt_tab')"

Incremental output: rows are flushed per batch, so a crash preserves
completed work. ``--variant a`` scores candidate-alone (sentence split);
``--variant b`` adds the ±1-neighbour scores onto an existing output file.

Usage:
  PYTHONPATH=/tmp/provence_pydeps external/valen/.venv/bin/python \
      scripts/eval_provence_real_context_v1.py --variant a \
      --data data/real_context_eval_v1/eval.jsonl \
          data/real_context_eval_v1/drop_supp/eval.jsonl \
      --output results/provence_real_eval_v1.jsonl
  PYTHONPATH=/tmp/provence_pydeps external/valen/.venv/bin/python \
      scripts/eval_provence_real_context_v1.py --variant b \
      --data ... (same) --output results/provence_real_eval_v1.jsonl
"""
import argparse
import gc
import json
import time
from pathlib import Path

MODEL_ID = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
QUERY_CHAR_CAP = 1200      # keep question well under the 512-token block
NEIGHBOR_CHAR_CAP = 1500   # per neighbour pseudo-sentence (variant B)
CAND_CHUNK_CHARS = 1500    # candidate pseudo-sentence chunk size (variant B)


def install_transformers5_shims():
    """Backfill tokenizer/model methods removed in transformers 5.x."""
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    def _build_inputs_with_special_tokens(self, ids0, ids1=None):
        # gte-modernbert pair layout, verified against encode(q, c):
        # [CLS] q [SEP] c [SEP]
        out = [self.cls_token_id] + [int(t) for t in ids0] + [self.sep_token_id]
        if ids1 is not None:
            out += [int(t) for t in ids1] + [self.sep_token_id]
        return out

    def _create_token_type_ids_from_sequences(self, ids0, ids1=None):
        # ModernBERT has no token-type embeddings; all-zero ids also keep the
        # caller from forwarding token_type_ids to the encoder.
        return [0] * (len(ids0) + (2 if ids1 is None else len(ids1) + 3))

    if not hasattr(TokenizersBackend, "build_inputs_with_special_tokens"):
        TokenizersBackend.build_inputs_with_special_tokens = (
            _build_inputs_with_special_tokens)
    if not hasattr(TokenizersBackend, "create_token_type_ids_from_sequences"):
        TokenizersBackend.create_token_type_ids_from_sequences = (
            _create_token_type_ids_from_sequences)

    # transformers>=5 populates `all_tied_weights_keys` inside post_init(),
    # which the remote OpenProvence class never calls (it was written for
    # 4.x).  Compute it lazily so weight loading does not crash.
    orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = (
                self.get_expanded_tied_weights_keys(all_submodels=True))
        return orig(self, loading_info)

    if not getattr(PreTrainedModel.mark_tied_weights_as_initialized,
                   "_provence_patched", False):
        _patched._provence_patched = True
        PreTrainedModel.mark_tied_weights_as_initialized = _patched


def entry_text(entry):
    if not entry:
        return ""
    c = entry.get("content")
    return c if c is not None else (entry.get("text") or "")


def chunk_text(text, size):
    """Split text into <=size-char pieces, preferring newline boundaries."""
    text = text or ""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start + 1, end)
            if nl > start + size // 4:
                end = nl + 1
        chunks.append(text[start:end])
        start = end
    return chunks or [""]


def extract(path):
    """Parse eval.jsonl rows -> (row-meta, model inputs)."""
    rows = []
    for line in Path(path).open():
        if not line.strip():
            continue
        rec = json.loads(line)
        state = json.loads(rec["request"]["state"])
        conv = state.get("conversation", [])
        cp = state.get("candidate_pointer")
        idx = next((i for i, s in enumerate(conv) if s.get("pointer") == cp),
                   -1)
        users = state.get("user_messages_in_order") or []
        query = users[-1] if users else state.get("current_user_request", "")
        tgt = ((rec.get("targets") or {}).get("irrelevant") or {}).get(
            "probabilities") or {}
        meta = rec.get("meta") or {}
        rows.append({
            "record_id": meta.get("record_id") or rec.get("id") or
                f"{path}:{len(rows)}",
            "label_drop": bool(tgt.get("true", 0) > 0.5),
            "target_true": tgt.get("true"),
            "source_file": str(path),
            "language_bucket": meta.get("language_bucket"),
            "candidate_kind": meta.get("candidate_kind"),
            "_query": query[:QUERY_CHAR_CAP],
            "query_truncated": len(query) > QUERY_CHAR_CAP,
            "_cand": entry_text(conv[idx]) if idx >= 0 else "",
            "_prev": entry_text(conv[idx - 1]) if idx > 0 else "",
            "_next": entry_text(conv[idx + 1])
                if 0 <= idx < len(conv) - 1 else "",
        })
    return rows


def strip_private(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def mps_cleanup():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--infer-batch", type=int, default=8,
                    help="process() internal inference batch (blocks/fwd); "
                         "kept small because MPS memory grows unboundedly "
                         "with this checkpoint (remote-code leak).")
    ap.add_argument("--device", default=None,
                    help="cpu|mps|auto (default auto -> MPS on Apple "
                         "Silicon). Use cpu for large context lists: MPS "
                         "allocator hit 180GB on variant B.")
    ap.add_argument("--variant", choices=["a", "b"], required=True)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    install_transformers5_shims()
    from transformers import AutoModel  # noqa: E402  (after shims)

    t0 = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      device=args.device)
    print(json.dumps({"event": "model_loaded", "model": args.model,
                      "device": str(getattr(model, "_runtime_device", "?")),
                      "max_length": getattr(model, "max_length", None),
                      "seconds": round(time.time() - t0, 1)}), flush=True)

    rows = []
    for path in args.data:
        rows.extend(extract(path))

    done_ids = set()
    existing = []
    if args.output.exists():
        for line in args.output.open():
            if line.strip():
                old = json.loads(line)
                existing.append(old)
                if args.variant == "a" and "provence_score" in old:
                    done_ids.add(old["record_id"])
                if args.variant == "b" and "provence_ctx_score" in old:
                    done_ids.add(old["record_id"])
    by_id = {r["record_id"]: r for r in rows}
    if args.variant == "b" and existing:
        # merge stored rows back so private fields are available for scoring
        for old in existing:
            if old["record_id"] in by_id:
                by_id[old["record_id"]].update(
                    {k: v for k, v in old.items() if not k.startswith("_")})

    out_mode = "a" if args.output.exists() and done_ids else "w"
    # For variant B we always rewrite the file at the end to keep one row per
    # record; variant A appends per batch.
    if args.variant == "b":
        results = {r["record_id"]: strip_private(r) for r in rows}
        for old in existing:
            results.setdefault(old["record_id"], {}).update(old)
        pending = [r for r in rows if r["record_id"] not in done_ids]
        for start in range(0, len(pending), args.batch):
            chunk = pending[start:start + args.batch]
            questions, contexts, spans = [], [], []
            for r in chunk:
                elems = []
                if r["_prev"]:
                    elems.append(r["_prev"][-NEIGHBOR_CHAR_CAP:])
                lo = len(elems)
                elems.extend(chunk_text(r["_cand"], CAND_CHUNK_CHARS))
                hi = len(elems)
                if r["_next"]:
                    elems.append(r["_next"][:NEIGHBOR_CHAR_CAP])
                questions.append(r["_query"])
                contexts.append([elems])
                spans.append((lo, hi))
            res = model.process(
                question=questions, context=contexts,
                threshold=0.1, title=None, show_progress=False,
                batch_size=args.infer_batch, preprocess_workers=0,
                return_sentence_metrics=True)
            for r, sp, (lo, hi) in zip(chunk,
                                       res["sentence_probabilities"],
                                       spans):
                doc = sp[0] if sp and isinstance(sp[0], (list, tuple)) \
                    else (sp or [])
                doc = [float(p) for p in doc]
                cp = doc[lo:hi]
                upd = {
                    "provence_ctx_score": max(cp) if cp else 0.0,
                    "mapped_noul_ctx": 1.0 - (max(cp) if cp else 0.0),
                    "provence_ctx_n_elems": len(doc),
                }
                r.update(upd)
                results[r["record_id"]].update(upd)
            mps_cleanup()
            print(json.dumps({"event": "variantB",
                              "done": start + len(chunk),
                              "pending": len(pending)}), flush=True)
            # incremental flush every batch
            with args.output.open("w") as f:
                for r in rows:
                    f.write(json.dumps(results[r["record_id"]],
                                       ensure_ascii=False) + "\n")
    else:
        f = args.output.open(out_mode)
        pending = [r for r in rows if r["record_id"] not in done_ids]
        for start in range(0, len(pending), args.batch):
            chunk = pending[start:start + args.batch]
            res = model.process(
                question=[r["_query"] for r in chunk],
                context=[r["_cand"] for r in chunk],
                threshold=0.1, title=None, show_progress=False,
                batch_size=args.infer_batch, preprocess_workers=0,
                return_sentence_metrics=True)
            for r, sp, rk in zip(chunk, res["sentence_probabilities"],
                                 res["reranking_score"]):
                sp = [float(p) for p in (sp or [])]
                r["provence_score"] = max(sp) if sp else 0.0
                r["mapped_noul"] = 1.0 - r["provence_score"]
                r["provence_mean_prob"] = (sum(sp) / len(sp)) if sp else 0.0
                r["provence_n_sent"] = len(sp)
                r["provence_kept_frac_t01"] = (
                    sum(1 for p in sp if p > 0.1) / len(sp)) if sp else 0.0
                r["provence_rerank"] = float(rk) if rk is not None else None
                f.write(json.dumps(strip_private(r), ensure_ascii=False)
                        + "\n")
            f.flush()
            mps_cleanup()
            print(json.dumps({"event": "variantA",
                              "done": start + len(chunk),
                              "pending": len(pending)}), flush=True)
        f.close()

    print(json.dumps({"event": "wrote", "output": str(args.output),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
