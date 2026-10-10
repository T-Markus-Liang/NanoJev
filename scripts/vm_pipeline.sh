#!/usr/bin/env bash
set -x
R=/content/nanojev; mkdir -p $R/data $R/logs $R/output
cd $R
curl -fsSL https://rclone.org/install.sh | bash > rclone_install.log 2>&1 || \
  (curl -fsSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip && unzip -o -q rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone /usr/local/bin/)
mkdir -p ~/.config/rclone && cp $R/rclone.conf ~/.config/rclone/rclone.conf
rclone copy gdrive:nanojev-staging/colab_kit $R/kit --checksum --transfers=8 -q
rclone copy gdrive:nanojev-staging/data/valen_nano_v10b $R/data/valen_nano_v10b --checksum -q
tar xzf $R/kit/env/valen_code.tgz -C $R
mkdir -p configs/cuda && cp $R/kit/env/sft_nano_v10b_cuda.json configs/cuda/
mkdir -p $R/output/nano_sft_v4/latest
cp $R/kit/ckpt/nano_sft_v4/checkpoint.pt $R/kit/ckpt/nano_sft_v4/config.json $R/output/nano_sft_v4/latest/
mv $R/kit/env/colab_remote_setup_v1.sh $R/kit/env/colab_requirements_v1.txt $R/kit/env/conv1d_sm89_src.tgz $R/ 2>/dev/null || true
bash $R/colab_remote_setup_v1.sh > $R/setup.log 2>&1
# manifests AFTER model download (setup skips dir if exists — now fixed order)
cp $R/kit/manifests/Qwen3.5-0.8B/valen_manifest.json $R/models/Qwen3.5-0.8B/ 2>/dev/null || true
cp $R/kit/manifests/Qwen3.5-2B.json $R/models/Qwen3.5-2B/valen_manifest.json 2>/dev/null || true
ls $R/models/Qwen3.5-0.8B/*.safetensors && echo MODELS_OK || echo MODELS_MISSING
pip uninstall -y torchao >> $R/setup.log 2>&1
# restore any previously pushed run state (resume-after-reclaim)
rclone copy gdrive:nanojev-staging/runs/v10b/output $R/output --transfers=4 -q 2>/dev/null || true
INIT=output/nano_sft_v4/latest
if [ -f $R/output/nano_sft_text_v10b/latest/checkpoint.pt ]; then
  # valen requires a NEW output dir for --initialize; move the restored run aside
  mkdir -p $R/resume_ref
  mv $R/output/nano_sft_text_v10b $R/resume_ref/v10b
  INIT=resume_ref/v10b/latest
  echo "RESUMING from pushed v10b checkpoint"
fi
# Drive push loop (checkpoint resilience — survives session death)
( while true; do
    sleep 300
    rclone copy $R/output gdrive:nanojev-staging/runs/v10b/output --transfers=4 -q 2>/dev/null
    rclone copy $R/logs gdrive:nanojev-staging/runs/v10b/logs -q 2>/dev/null
  done ) &
echo $! > $R/.pushloop_pid
cd $R && CUDA_VISIBLE_DEVICES=0 nohup python -m valen.train --config configs/cuda/sft_nano_v10b_cuda.json --initialize $INIT >> logs/v10b.log 2>&1
# blocking wait — script stays alive while training runs (colab run holds it)
TRAIN_PID=$!
wait $TRAIN_PID
rclone copy $R/output gdrive:nanojev-staging/runs/v10b/output --transfers=4 2>/dev/null
rclone copy $R/logs gdrive:nanojev-staging/runs/v10b/logs 2>/dev/null
touch $R/.all_done
rclone copy $R/.all_done gdrive:nanojev-staging/runs/v10b/ -q
echo PIPELINE_DONE
