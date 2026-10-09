#!/usr/bin/env python3
"""Build a stitched-noise corpus from real open-corpus conversations.

Real agent sessions accumulate unrelated contexts: a long conversation about
topic A, then the user switches to topic B — the A-history is real text but
genuinely irrelevant. This script produces that shape by splicing the middle
turns of one real conversation in front of another's final exchange.

Every message is real open-corpus text (no synthesized content); only the
conversation boundaries are artificial. The spliced history segments are
marked eligible in the sidecar — exactly how an agent runtime would tag
stale session context.

Output: data/open_corpus_v1/stitched_manifest.json (manifest-shaped).
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "open_corpus_v1" / "manifest.json"
OUT = ROOT / "data" / "open_corpus_v1" / "stitched_manifest.json"


def main():
    manifest = json.loads(IN.read_text(encoding="utf-8"))
    sources = [c for c in manifest["cases"]]
    cases = []
    for index, target in enumerate(sources):
        donor = sources[(index * 37 + 11) % len(sources)]  # fixed spread
        if donor["case_id"] == target["case_id"]:
            continue
        donor_msgs = donor["body"]["messages"]
        target_msgs = target["body"]["messages"]
        # Donor's middle turns (skip its system + final exchange).
        filler = [m for m in donor_msgs[1:-2]
                  if m["role"] in ("user", "assistant", "tool")]
        if len(filler) < 3:
            continue
        system = ([m for m in target_msgs if m["role"] == "system"] or
                  [{"role": "system", "content":
                    "Continue the current task using available context."}])
        tail = [m for m in target_msgs if m["role"] != "system"]
        messages = system[:1] + filler + tail
        if len(messages) < 7 or messages[-1]["role"] not in ("user", "tool"):
            continue
        sidecar = {"segments": {}}
        for i, message in enumerate(messages):
            # eligible = donor filler only (system and target tail pinned)
            if 1 <= i <= len(filler) and \
                    message["role"] in ("assistant", "tool"):
                sidecar["segments"][f"/messages/{i}/content"] = \
                    {"eligible": True}
        if not sidecar["segments"]:
            continue
        cases.append({
            "case_id": f"stitched_{target['case_id']}",
            "kind": "stitched_real_noise",
            "wire_format": "openai_chat",
            "description": "real donor history spliced before real target "
                           "exchange (topic-switch noise)",
            "body": {"model": "open-corpus-stitched", "messages": messages},
            "sidecar": sidecar,
            "source": {"dataset": "stitched",
                       "donor": donor["case_id"], "target": target["case_id"],
                       "provenance": [target.get("source"),
                                      donor.get("source")]},
        })
    out = {"schema_version": "nanojev-stitched-corpus-v1",
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "real open-corpus text; boundaries artificial; donor "
                   "history is genuinely irrelevant to the target exchange",
           "cases": cases}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(cases)} stitched cases -> {OUT}")


if __name__ == "__main__":
    main()
