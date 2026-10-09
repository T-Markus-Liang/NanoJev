#!/usr/bin/env python3
"""Fetch a bounded open-corpus conversation sample for real-traffic eval.

Sources (public HF datasets, sampled via the datasets-server rows API —
no full-dataset download, no auth):

- HuggingFaceH4/ultrachat_200k (default/test_sft): multi-turn dialogues.
- NousResearch/hermes-function-calling-v1 (func_calling/train):
  conversations carrying tool calls and results.

Conversion: → openai_chat wire body + auto-generated eligibility sidecar
(assistant messages older than the trailing exchange are the candidates —
mirrors real usage where old context is the filterable part). Output is a
fixture-manifest-shaped file so `run_e2e_local_upstream_v1.py` consumes it
unchanged. Local-only output under data/ (gitignored).
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "open_corpus_v1"
ROWS = "https://datasets-server.huggingface.co/rows"
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
            "user": "user", "assistant": "assistant", "tool": "tool",
            "function": "tool"}

SOURCES = {
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "config": "default", "split": "test_sft",
        "field": "messages", "role_key": "role", "text_key": "content",
        "license": "MIT (dataset card)",
        "url": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
    },
    "hermes_fc": {
        "dataset": "NousResearch/hermes-function-calling-v1",
        "config": "func_calling", "split": "train",
        "field": "conversations", "role_key": "from", "text_key": "value",
        "license": "public dataset (Nous Research)",
        "url": "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1",
    },
    "toolace": {
        "dataset": "Team-ACE/ToolACE",
        "config": "default", "split": "train",
        "field": "conversations", "role_key": "from", "text_key": "value",
        "system_column": "system",
        "license": "CC-BY-NC 4.0 (dataset card)",
        "url": "https://huggingface.co/datasets/Team-ACE/ToolACE",
    },
}


def fetch_rows(source, offset, length):
    query = urllib.parse.urlencode({
        "dataset": source["dataset"], "config": source["config"],
        "split": source["split"], "offset": offset, "length": length})
    with urllib.request.urlopen(f"{ROWS}?{query}", timeout=60) as response:
        return json.loads(response.read())


def convert(source_name, source, row, index):
    raw_messages = row.get(source["field"]) or []
    messages = []
    system_col = source.get("system_column")
    if system_col and isinstance(row.get(system_col), str) \
            and row[system_col].strip():
        messages.append({"role": "system", "content": row[system_col]})
    for item in raw_messages:
        role = ROLE_MAP.get(item.get(source["role_key"], ""))
        text = item.get(source["text_key"])
        if role is None or not isinstance(text, str) or not text.strip():
            continue
        messages.append({"role": role, "content": text})
    # Dataset convention: conversation ends with the assistant's reference
    # answer. Request = everything before it (ends with the final user turn);
    # the dropped tail becomes the pseudo-reference for answer comparison.
    reference = None
    if len(messages) >= 2 and messages[-1]["role"] == "assistant":
        reference = messages.pop()["content"]
    # Need a final user/tool turn plus filterable history (tool-final is the
    # normal function-calling flow: model answers after the tool result).
    if len(messages) < 5 or messages[-1]["role"] not in ("user", "tool"):
        return None
    if not any(m["role"] in ("assistant", "tool") for m in messages[:-3]):
        return None
    sidecar = {"segments": {}}
    # Eligible = assistant/tool history except the immediate pre-question
    # exchange (last two history items stay pinned as current context).
    for i, message in enumerate(messages[:-3]):
        if message["role"] in ("assistant", "tool"):
            sidecar["segments"][f"/messages/{i}/content"] = {"eligible": True}
    if not sidecar["segments"]:
        return None
    return {
        "case_id": f"{source_name}_{index}",
        "kind": "open_corpus",
        "wire_format": "openai_chat",
        "description": f"real open-corpus conversation ({source_name})",
        "body": {"model": "open-corpus", "messages": messages},
        "reference_answer": reference,
        "sidecar": sidecar,
        "source": {"dataset": source["dataset"], "config": source["config"],
                   "split": source["split"], "row_offset": index,
                   "license": source["license"], "url": source["url"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=int, default=50)
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], []
    for name, source in SOURCES.items():
        picked = 0
        for page in range(args.pages):
            if picked >= args.per_source:
                break
            offset = page * 100
            try:
                data = fetch_rows(source, offset, 100)
            except Exception as exc:
                print(f"{name}: page {page} fetch failed: {exc}")
                break
            for item in data.get("rows") or []:
                if picked >= args.per_source:
                    break
                case = convert(name, source, item["row"], offset + item["row_idx"])
                if case:
                    cases.append(case)
                    picked += 1
            time.sleep(0.5)
        print(f"{name}: {picked} conversations converted")
        provenance.append({"source": name, "picked": picked,
                           "license": source["license"], "url": source["url"]})
    manifest = {
        "schema_version": "nanojev-open-corpus-manifest-v1",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "public HF dataset samples for local-only evaluation; "
                "no labels — quality measured via answer-agreement proxy",
        "provenance": provenance,
        "cases": cases,
    }
    out = OUT_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"total {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
