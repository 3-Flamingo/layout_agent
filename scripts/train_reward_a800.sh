#!/usr/bin/env bash
# Train reward model on 8× A800 with torchrun + DDP.
#
# Usage (from layout_agent repo root):
#   # current small set (~300 pairs) — use bootstrap config
#   bash scripts/train_reward_a800.sh
#
#   # large set (thousands of pairs)
#   CONFIG=src/layout_agent/training/configs/a800_8x.yaml \
#     bash scripts/train_reward_a800.sh data/evalsys/pairs_r1r2/manifest.jsonl runs/reward_a800
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${1:-data/evalsys/pairs_perturb/manifest.jsonl}"
OUT="${2:-runs/reward_a800}"
# Default to bootstrap (safe for hundreds of pairs). Override with CONFIG=... for full a800_8x.yaml.
CONFIG="${CONFIG:-src/layout_agent/training/configs/a800_8x_bootstrap.yaml}"
NPROC="${NPROC_PER_NODE:-8}"
if [[ -n "${TORCHRUN:-}" ]]; then
  TORCHRUN_BIN="$TORCHRUN"
elif [[ -x "${ROOT}/.venv/bin/torchrun" ]]; then
  TORCHRUN_BIN="${ROOT}/.venv/bin/torchrun"
else
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [[ -z "$TORCHRUN_BIN" ]]; then
  echo "torchrun not found; activate the training environment or set TORCHRUN=/path/to/torchrun" >&2
  exit 127
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "config=$CONFIG"
echo "nproc_per_node=$NPROC"
echo "torchrun=$TORCHRUN_BIN"
nvidia-smi -L || true

"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" \
  -m layout_agent.training.train \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --config "$CONFIG"
