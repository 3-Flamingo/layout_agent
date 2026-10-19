#!/bin/bash
# Train with reduced synth share (75% of CMU pairs). GPUs 0-3 only.
set -euo pipefail
cd /workspace/github/layout_agent
export PYTHONPATH=/workspace/github/layout_agent/src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

PY=/root/miniconda3/envs/layout_train/bin/python
TR=/root/miniconda3/envs/layout_train/bin/torchrun
OUT=runs/reward_human_pairs_synth_light
mkdir -p "$OUT"

echo "visible GPUs (must be 4):"
"$PY" - <<'PY'
import os, torch
n = torch.cuda.device_count()
print("cuda_available", torch.cuda.is_available(), "count", n)
assert n == 4, n
for i in range(n):
    print(i, torch.cuda.get_device_name(i))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

exec "$TR" --standalone --nproc_per_node=4 \
  -m layout_agent.training.train \
  --manifest data/evalsys/reward_human_pairs_synth/manifest_light.jsonl \
  --out "$OUT" \
  --config src/layout_agent/training/configs/a800_4x_bootstrap.yaml
