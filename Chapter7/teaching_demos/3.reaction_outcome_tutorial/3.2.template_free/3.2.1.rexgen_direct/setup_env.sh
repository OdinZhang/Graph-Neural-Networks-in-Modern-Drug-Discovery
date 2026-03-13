#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/envs/rexgen_direct_tutorial_envs"
ENV_NAME="rexgen_direct_tutorial_envs"

# 检查源码仓库
if [[ ! -d "${PROJECT_ROOT}/source_repos/rexgen_direct" ]]; then
  echo "缺少 source_repos/rexgen_direct，请先克隆源码仓库：" >&2
  echo "  cd ${PROJECT_ROOT}/source_repos && git clone https://github.com/connorcoley/rexgen_direct.git" >&2
  exit 1
fi

# 让 bash 能识别 conda 命令
source "$(conda info --base)/etc/profile.d/conda.sh"

# 创建 Conda 环境
if [[ ! -d "${ENV_DIR}" ]]; then
  echo "[1/4] 创建 Conda 环境: ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.11
else
  echo "[1/4] 复用已有 Conda 环境: ${ENV_DIR}"
fi

# 激活环境
conda activate "${ENV_DIR}"

echo "[2/4] 安装核心依赖（RDKit + 科学计算）"
conda install -y -c conda-forge \
    numpy \
    pandas \
    tqdm \
    rdkit \
    matplotlib \
    ipykernel

echo "[3/4] 安装 PyTorch（CPU 版）"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] 注册 Jupyter Kernel"
python -m ipykernel install --prefix "${ENV_DIR}" \
    --name "${ENV_NAME}" \
    --display-name "Python (${ENV_NAME})"

echo
echo "环境准备完成。"
echo "激活命令: conda activate ${ENV_DIR}"
echo "启动教程: bash ${SCRIPT_DIR}/launch_notebook.sh"
