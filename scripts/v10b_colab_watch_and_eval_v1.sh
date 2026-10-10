#!/usr/bin/env bash
# v10b Colab completion watcher + auto-eval (2026-10-10)
# Polls Drive for the pipeline's .all_done marker, pulls the checkpoint,
# launches two fp32 valen-head sidecars, scores the three eval layers,
# writes a verdict table, then kills the sidecars.
# Usage: nohup bash scripts/v10b_colab_watch_and_eval_v1.sh > /tmp/v10b_watch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890

R=gdrive:nanojev-staging/runs/v10b
CKPT=external/valen/output/nano_sft_text_v10b_colab
PY=external/valen/.venv/bin/python
PORTS="8097,8098"
TAG=v10b_colab

echo "[$(date +%H:%M)] watcher start; polling $R/.all_done every 5min"
while ! rclone lsf "$R/.all_done" >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%H:%M)] .all_done seen; pulling checkpoint"

mkdir -p "$CKPT"
rclone copy "$R/output/nano_sft_text_v10b" "$CKPT" --transfers=4 --checksum
[ -f "$CKPT/latest/checkpoint.pt" ] || { echo "FAIL: no checkpoint.pt after pull"; exit 1; }
rclone copy "$R/logs" "/tmp/v10b_colab_logs" >/dev/null 2>&1 || true
(cd "$CKPT" && find . -type f | sort | xargs shasum -a 256) > "results/${TAG}_ckpt.sha256"
echo "[$(date +%H:%M)] pulled; sha256 receipt -> results/${TAG}_ckpt.sha256"

echo "[$(date +%H:%M)] launching heads on $PORTS (fp32)"
for p in ${PORTS//,/ }; do
  "$PY" scripts/valen_head_server_v1.py --port "$p" \
    --checkpoint "$CKPT/latest" --dtype fp32 >/tmp/head_$p.log 2>&1 &
done
for i in $(seq 1 60); do
  ok=1
  for p in ${PORTS//,/ }; do
    curl -sf -m2 "http://127.0.0.1:$p/health" >/dev/null || ok=0
  done
  [ $ok -eq 1 ] && break; sleep 5
done
echo "[$(date +%H:%M)] heads up; scoring 3 layers"

"$PY" scripts/parallel_eval_v1.py \
  --data data/real_context_eval_v1/eval.jsonl \
  --output "results/${TAG}_frozen_main.jsonl" \
  --ports "$PORTS" --timeout 90
"$PY" scripts/parallel_eval_v1.py \
  --data data/real_context_eval_v1/drop_supp/eval.jsonl \
  --output "results/${TAG}_frozen_supp.jsonl" \
  --ports "$PORTS" --timeout 90
"$PY" scripts/parallel_eval_v1.py \
  --data data/real_context_eval_v5_ext/eval.jsonl \
  --output "results/${TAG}_v5ext.jsonl" \
  --ports "$PORTS" --timeout 90

echo "[$(date +%H:%M)] scoring done; aggregating"
python3 - "$TAG" <<'EOF'
import json, sys
tag = sys.argv[1]
layers = {
    "main":  f"results/{tag}_frozen_main.jsonl",
    "supp":  f"results/{tag}_frozen_supp.jsonl",
    "v5ext": f"results/{tag}_v5ext.jsonl",
}
TAU, CONF = 0.5, 0.8
rows, agg = {}, []
for name, path in layers.items():
    items = [json.loads(l) for l in open(path) if l.strip()]
    tp = fp = fn = tn = 0
    conf_fp = 0
    for it in items:
        noul, tgt = it["noul"], it["target"]
        if noul is None:
            continue
        hat, drop = noul >= TAU, tgt >= 0.5
        if hat and drop: tp += 1
        elif hat: fp += 1; conf_fp += noul >= CONF
        elif drop: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    drec = tp / (tp + fn) if tp + fn else 0.0
    rows[name] = dict(n=n, acc=acc, fp=fp, conf_fp=conf_fp, drec=drec)
    agg += [tp, fp, fn, tn]
tp, fp, fn, tn = agg
merged = (tp + tn) / (tp + fp + fn + tn)
out = [f"# {tag} verdict (tau={TAU}, confFP noul>={CONF})", "",
       "| layer | n | acc | FP | confFP | dRec |", "|---|---|---|---|---|---|"]
for name, r in rows.items():
    out.append(f"| {name} | {r['n']} | {r['acc']:.4f} | {r['fp']} "
               f"| {r['conf_fp']} | {r['drec']:.2f} |")
out.append(f"| **merged** | {sum(r['n'] for r in rows.values())} "
           f"| {merged:.4f} | | | |")
open(f"results/{tag}_table.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
EOF

for p in ${PORTS//,/ }; do pkill -f "valen_head_server_v1.py --port $p" 2>/dev/null; done
echo "[$(date +%H:%M)] heads killed; DONE"
