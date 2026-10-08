#!/usr/bin/env python3
"""filter_v8_maxlen_v1.py — drop records whose compiled prompt exceeds 8192
tokens (the valen qwen compiler raises ValueError; we catch and filter)."""
import json, sys
sys.path.insert(0, "external/valen")
from multiprocessing import Pool

C = None

def init():
    global C
    from transformers import AutoProcessor
    from valen.data.compilers.qwen import Compiler
    C = Compiler(AutoProcessor.from_pretrained("external/valen/models/Qwen3.5-0.8B"),
                 max_length=8192)

def check(args):
    i, line = args
    try:
        C.compile(json.loads(line))
        return i, True
    except Exception:
        return i, False

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    lines = open(src).read().splitlines()
    with Pool(8, initializer=init) as p:
        ok = {i for i, good in p.map(check, enumerate(lines), chunksize=64) if good}
    kept = [l for i, l in enumerate(lines) if i in ok]
    open(dst, "w").write("\n".join(kept) + "\n")
    print(f"{len(lines)} -> {len(kept)} (dropped {len(lines)-len(kept)} overlong)")
