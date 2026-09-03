#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:?Usage: launch_trajectory_50c.sh <exact|grid1000>}
: "${H3_TRAJECTORY_DIR:?Set H3_TRAJECTORY_DIR to a new persistent run directory}"
: "${H3_EXPECTED_HEAD:?Set H3_EXPECTED_HEAD to the exact branch commit under test}"
: "${NODE_RANK:?Set NODE_RANK=0 on node 0 and NODE_RANK=1 on node 1}"

[[ ${H3_TRAJECTORY_DIR} = /* ]] || {
  echo "H3_TRAJECTORY_DIR must be an absolute persistent path" >&2
  exit 2
}
ACTUAL_HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)
[[ ${ACTUAL_HEAD} == "${H3_EXPECTED_HEAD}" ]] || {
  echo "HEAD mismatch: actual=${ACTUAL_HEAD} expected=${H3_EXPECTED_HEAD}" >&2
  exit 2
}
[[ -z $(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no) ]] || {
  echo "Tracked worktree changes are forbidden for trajectory evidence" >&2
  exit 2
}

case "${VARIANT}" in
  exact)
    LAUNCHER="${REPO_ROOT}/scripts/launch_2node.sh"
    export CONFIG=${CONFIG:-"${REPO_ROOT}/configs/minimax_h3_t2av_dmd_a100_world16.yaml"}
    export H3_ATTN_BACKEND=_flash_3_hub
    export H3_ACTIVATION_CHECKPOINT_SEGMENT=1
    export H3_ACTIVATION_POLICY=checkpoint_boundary_cpu
    unset H3_ACTIVATION_OFFLOAD H3_ACTIVATION_OFFLOAD_MIN_BYTES || true
    unset H3_FA3_REPLAY_CACHE_BLOCKS H3_FA3_REPLAY_CACHE_STORAGE || true
    unset H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD || true
    unset H3_ADALN_GRID_MANIFEST || true
    ARM_ID=h3-exact-continuous-boundary-cpu/v1
    GRID_MANIFEST_SHA256=none
    ;;
  grid1000)
    LAUNCHER="${REPO_ROOT}/scripts/launch_2node_grid1000.sh"
    export CONFIG=${CONFIG:-"${REPO_ROOT}/configs/minimax_h3_t2av_dmd_a100_world16_grid1000.yaml"}
    : "${H3_ADALN_GRID_MANIFEST:?Grid trajectory requires H3_ADALN_GRID_MANIFEST}"
    [[ -f ${H3_ADALN_GRID_MANIFEST} ]] || { echo "Missing Grid manifest" >&2; exit 2; }
    export H3_ATTN_BACKEND=_flash_3_hub
    export H3_ACTIVATION_CHECKPOINT_SEGMENT=1
    export H3_ACTIVATION_POLICY=none
    unset H3_ACTIVATION_OFFLOAD H3_ACTIVATION_OFFLOAD_MIN_BYTES || true
    export H3_FA3_REPLAY_CACHE_BLOCKS=0-49
    export H3_FA3_REPLAY_CACHE_STORAGE=cpu_staged
    export H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT=2
    export H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD=true
    ARM_ID=h3-grid1000-iteration418/v1
    GRID_MANIFEST_SHA256=$(sha256sum "${H3_ADALN_GRID_MANIFEST}" | awk '{print $1}')
    ;;
  *)
    echo "Unknown trajectory variant: ${VARIANT}" >&2
    exit 2
    ;;
esac
[[ -x ${LAUNCHER} ]] || { echo "Missing launcher: ${LAUNCHER}" >&2; exit 2; }

# Both hosts may share the evidence root. Refuse only files owned by this
# node's eight global ranks, avoiding a node0/node1 directory-creation race.
mkdir -p "${H3_TRAJECTORY_DIR}"
rank_start=$(( NODE_RANK * 8 ))
for ((rank=rank_start; rank<rank_start+8; rank++)); do
  receipt=$(printf '%s/rank_%03d.trajectory.jsonl' "${H3_TRAJECTORY_DIR}" "${rank}")
  [[ ! -e ${receipt} ]] || { echo "Refusing to overwrite ${receipt}" >&2; exit 2; }
done
if [[ ${NODE_RANK} == 0 && -e ${H3_TRAJECTORY_DIR}/trajectory_manifest.json ]]; then
  echo "Refusing to overwrite ${H3_TRAJECTORY_DIR}/trajectory_manifest.json" >&2
  exit 2
fi
launcher_receipt="${H3_TRAJECTORY_DIR}/launcher_node${NODE_RANK}.txt"
if ! (set -o noclobber; printf 'head=%s\nvariant=%s\nnode_rank=%s\narm_id=%s\nattention=%s\nactivation_policy=%s\ncheckpoint_segment=%s\ngrid_manifest_sha256=%s\n' \
  "${ACTUAL_HEAD}" "${VARIANT}" "${NODE_RANK}" "${ARM_ID}" "${H3_ATTN_BACKEND}" \
  "${H3_ACTIVATION_POLICY}" "${H3_ACTIVATION_CHECKPOINT_SEGMENT}" "${GRID_MANIFEST_SHA256}" \
  >"${launcher_receipt}") 2>/dev/null; then
  echo "Refusing to reuse trajectory run root; marker exists: ${launcher_receipt}" >&2
  exit 2
fi

export H3_TRAJECTORY_MODE=coarse_v1
export H3_TRAJECTORY_VARIANT="${VARIANT}"
H3_TRAJECTORY_CYCLES=${H3_TRAJECTORY_CYCLES:-50}
[[ ${H3_TRAJECTORY_CYCLES} =~ ^[1-9][0-9]*$ ]] || {
  echo "H3_TRAJECTORY_CYCLES must be a positive integer" >&2
  exit 2
}
case "${H3_TRAJECTORY_CYCLES}" in
  1) DEFAULT_TRAJECTORY_ANCHORS=1 ;;
  5) DEFAULT_TRAJECTORY_ANCHORS=1,5 ;;
  50) DEFAULT_TRAJECTORY_ANCHORS=1,10,25,50 ;;
  *) DEFAULT_TRAJECTORY_ANCHORS="1,${H3_TRAJECTORY_CYCLES}" ;;
esac
export H3_TRAJECTORY_EXPECTED_CYCLES="${H3_TRAJECTORY_CYCLES}"
export H3_TRAJECTORY_EXPECTED_WORLD_SIZE=16
export H3_TRAJECTORY_ANCHORS=${H3_TRAJECTORY_ANCHORS:-${DEFAULT_TRAJECTORY_ANCHORS}}
export H3_TRAJECTORY_SAMPLES_PER_TENSOR=${H3_TRAJECTORY_SAMPLES_PER_TENSOR:-4}
export H3_TRAJECTORY_MAX_TENSORS_PER_ROLE=${H3_TRAJECTORY_MAX_TENSORS_PER_ROLE:-32}
export H3_MAX_ITERS="${H3_TRAJECTORY_CYCLES}"
export H3_SAVE_EVERY=0
export H3_OUTPUT_DIR=${H3_OUTPUT_DIR:-"${H3_TRAJECTORY_DIR}/training_output"}

exec "${LAUNCHER}"
