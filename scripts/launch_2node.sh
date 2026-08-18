#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-"${REPO_ROOT}/third_party/LightX2V"}
CONFIG=${CONFIG:-"${REPO_ROOT}/configs/minimax_h3_t2av_dmd_a100_2x8.yaml"}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:?Set NODE_RANK=0 on node 0 and NODE_RANK=1 on node 1}
MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the node-0 reachable address}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTHONPATH="${REPO_ROOT}:${LIGHTX2V_ROOT}:${LIGHTX2V_ROOT}/lightx2v_train:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

if [[ ${H3_SKIP_PREFLIGHT:-0} != 1 ]]; then
  python "${REPO_ROOT}/scripts/preflight.py" --lightx2v-root "${LIGHTX2V_ROOT}"
fi

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ -d "${LIGHTX2V_ROOT}/lightx2v_train" ]] || {
  echo "Missing LightX2V checkout. Run scripts/bootstrap_lightx2v.sh first." >&2
  exit 1
}

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m h3_a100.entrypoint \
  --config "${CONFIG}"
