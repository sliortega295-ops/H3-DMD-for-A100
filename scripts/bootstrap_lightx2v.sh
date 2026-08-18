#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_COMMIT=$(cat "${REPO_ROOT}/UPSTREAM_COMMIT")
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-"${REPO_ROOT}/third_party/LightX2V"}

mkdir -p "$(dirname "${LIGHTX2V_ROOT}")"
if [[ ! -d "${LIGHTX2V_ROOT}/.git" ]]; then
  git clone https://github.com/ModelTC/LightX2V.git "${LIGHTX2V_ROOT}"
fi

git -C "${LIGHTX2V_ROOT}" fetch --all --tags
git -C "${LIGHTX2V_ROOT}" checkout --detach "${UPSTREAM_COMMIT}"

cat > "${REPO_ROOT}/.h3-a100-env" <<EOF
export H3_A100_REPO="${REPO_ROOT}"
export LIGHTX2V_ROOT="${LIGHTX2V_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${LIGHTX2V_ROOT}:${LIGHTX2V_ROOT}/lightx2v_train:\${PYTHONPATH:-}"
EOF

printf '\nPinned LightX2V at %s\n' "${UPSTREAM_COMMIT}"
printf 'Run: source %q\n' "${REPO_ROOT}/.h3-a100-env"
printf 'Then install the LightX2V H3 training dependencies in your existing CUDA environment.\n'
