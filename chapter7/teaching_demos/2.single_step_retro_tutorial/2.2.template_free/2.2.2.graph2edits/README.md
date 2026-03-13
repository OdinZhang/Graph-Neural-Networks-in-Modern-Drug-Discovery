# Graph2Edits 教学材料

本目录基于 `source_repos/Graph2Edits` 仓库整理了一套面向教学的最小教程，组织方式参考 `2.1.1.gln/good_reference_gln.md` 的大纲。

## 文件结构

- `1_环境配置.ipynb`：现代 CPU-only 环境配置、路径约束与源码 smoke test。
- `2_数据处理.ipynb`：从 raw reaction 到 edit 序列、词表、图状态序列、batch 张量的完整最小流程。
- `3_模型展示.ipynb`：图张量、`MPNEncoder`、`compute_edit_scores()`、oracle 解码与 beam search 原理。
- `data/demo_reactions.csv`：从 USPTO_50k 训练集抽出的 3 条教学样本。
- `data/processed/uspto_50k_demo/train/`：数据 notebook 运行后生成的教学版中间产物。
- `task.md`：原始任务说明。

## 路径约束

本教程中的代码统一先定位项目根目录 `Chemical_Synthesis`，再使用如下相对路径：

- `source_repos/Graph2Edits`
- `teaching_demos/2.single_step_retro_tutorial/2.2.template_free/2.2.2.graph2edits`
- `envs/graph2edits_tutorial_envs`

## 教学样本

`data/demo_reactions.csv` 目前包含 3 条反应，分别覆盖：

- `Change Atom + Attaching LG`
- `Delete Bond + Attaching LG`
- `Change Bond`

这样可以在极小样本上把 Graph2Edits 的主要 edit 类型讲完整。

## 环境说明

教学环境使用 CPU-only `torch`。建议从项目根目录运行：

```bash
/usr/bin/python3 -m venv envs/graph2edits_tutorial_envs
source envs/graph2edits_tutorial_envs/bin/activate
python -m pip install --upgrade pip
python -m pip install rdkit pandas joblib numpy matplotlib ipykernel nbformat nbclient
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name graph2edits_tutorial_envs --display-name "Python (graph2edits_tutorial_envs)"
```
