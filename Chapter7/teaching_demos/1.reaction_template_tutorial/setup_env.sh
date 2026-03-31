#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/envs/reaction_template_tutorial_envs"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -d "${PROJECT_ROOT}/source_repos/rxnmapper" || ! -d "${PROJECT_ROOT}/source_repos/rdchiral" ]]; then
  echo "缺少 source_repos/rxnmapper 或 source_repos/rdchiral，无法继续。" >&2
  exit 1
fi

echo "[1/5] 创建 conda 虚拟环境: ${ENV_DIR}"

if conda env list | grep -q "reaction_template_tutorial_envs"; then
  echo "    复用已有 conda 环境: reaction_template_tutorial_envs"
else
  conda create -y -p "${ENV_DIR}" python=3.12
fi

echo "[2/5] 升级 pip / wheel，并固定 setuptools<81"
"${ENV_DIR}/bin/pip" install --upgrade "pip>=25" "setuptools<81" wheel

echo "[3/5] 安装 CPU 版 PyTorch"
"${ENV_DIR}/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"

echo "[4/5] 安装 notebook 与化学依赖"
"${ENV_DIR}/bin/pip" install -r "${REQ_FILE}"

echo "[5/5] 安装本地源码仓库，并注册内核"
"${ENV_DIR}/bin/pip" install --no-build-isolation -e "${PROJECT_ROOT}/source_repos/rxnmapper" -e "${PROJECT_ROOT}/source_repos/rdchiral"
"${ENV_DIR}/bin/python" -m ipykernel install --prefix "${ENV_DIR}" --name reaction_template_tutorial_envs --display-name "Python (reaction_template_tutorial_envs)"

echo
echo "环境准备完成。"
echo "激活命令: source \"${ENV_DIR}/bin/activate\""
echo "启动教程: bash \"${SCRIPT_DIR}/launch_notebook.sh\""
