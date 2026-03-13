#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/envs/reaction_template_tutorial_envs"
NOTEBOOK_PATH="${SCRIPT_DIR}/RXNMapper_RDChiral_教学示例.ipynb"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${SCRIPT_DIR}/setup_env.sh"
fi

exec "${ENV_DIR}/bin/jupyter" lab "${NOTEBOOK_PATH}" --notebook-dir="${PROJECT_ROOT}"
