#!/usr/bin/env bash
set -euo pipefail

# Run this on one 8xA100 node after the functional smoke test. The custom trainer
# emits NVTX ranges for student, 5x rollout preparation, and fake updates.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-"${REPO_ROOT}/third_party/LightX2V"}
CONFIG=${CONFIG:-"${REPO_ROOT}/configs/minimax_h3_t2av_dmd_a100_smoke.yaml"}
OUT=${OUT:-"${REPO_ROOT}/profiles/h3_a100_smoke"}
mkdir -p "$(dirname "${OUT}")"
export PYTHONPATH="${REPO_ROOT}:${LIGHTX2V_ROOT}:${LIGHTX2V_ROOT}/lightx2v_train:${PYTHONPATH:-}"

exec nsys profile \
  --trace=cuda,nvtx,osrt,nccl \
  --sample=none \
  --force-overwrite=true \
  --output="${OUT}" \
  torchrun --standalone --nproc_per_node=8 \
  -m h3_a100.entrypoint --config "${CONFIG}"
