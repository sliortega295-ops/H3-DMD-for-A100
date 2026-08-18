#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-"${REPO_ROOT}/third_party/LightX2V"}
CONFIG=${CONFIG:-"${REPO_ROOT}/configs/minimax_h3_t2av_dmd_a100_smoke.yaml"}
export PYTHONPATH="${REPO_ROOT}:${LIGHTX2V_ROOT}:${LIGHTX2V_ROOT}/lightx2v_train:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

if [[ ${H3_SKIP_PREFLIGHT:-0} != 1 ]]; then
  python "${REPO_ROOT}/scripts/preflight.py" --lightx2v-root "${LIGHTX2V_ROOT}"
fi

exec torchrun --standalone --nproc_per_node=8 \
  -m h3_a100.entrypoint --config "${CONFIG}"
