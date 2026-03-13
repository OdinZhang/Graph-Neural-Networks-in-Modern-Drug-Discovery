#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/envs/rexgen_direct_tutorial_envs"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "环境不存在，先运行 setup_env.sh ..."
  bash "${SCRIPT_DIR}/setup_env.sh"
fi

conda activate "${ENV_DIR}"
exec jupyter lab --notebook-dir="${SCRIPT_DIR}"
