# YieldMPNN 教学材料

本目录提供 `reaction_yield_nn`（论文名通常写作 YieldMPNN） 的教学版材料，遵循项目内既有的三段式结构：

1. `1_环境配置.ipynb`
2. `2_数据处理.ipynb`
3. `3_模型展示.ipynb`

## 教学目标

这组材料不是把原仓库脚本直接搬进 notebook，而是把原仓库的三条主线拆开讲清楚：

- 环境与依赖如何在现代 Python / PyTorch / RDKit / DGL 组合下复现
- `data/get_data.py` 如何把 reaction SMILES 压缩成 `dataset_*.npz`
- `dataset.py`、`util.py`、`model.py` 如何在推理阶段协同工作

## 关键路径

所有 notebook 都会先定位当前项目根目录，然后用“相对当前项目根目录”的方式去解析外部源码和环境目录：

- 源码仓库相对路径：`../../../../../data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/source_repos/reaction_yield_nn`
- 环境根目录相对路径：`../../../../../data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/envs`
- 教学环境建议路径：`../../../../../data/ubuntu_work_beta/other_work/GNN_AIDD/Chemical_Synthesis/envs/yieldmpnn_tutorial_envs`

## 辅助文件

- `data/yieldmpnn_demo_reactions.csv`
  - 从 Buchwald-Hartwig `data1_split_0.npz` 中抽出的 5 条小样本反应，用于教学演示。
- `yieldmpnn_tutorial_env.yml`
  - 教学环境的 conda 配置草案，当前采用 `PyTorch 2.6.0 + DGL 2.5.0 + RDKit 2025.09.6` 组合。

## 对应源码

- `data/get_data.py`
- `dataset.py`
- `util.py`
- `model.py`
- `run_code.py`
