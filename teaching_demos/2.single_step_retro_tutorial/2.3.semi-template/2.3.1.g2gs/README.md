# G2Gs 教学版

本目录基于 `torchdrug` 的 retrosynthesis 教程与源码，抽取出最小可教学的 G2Gs 代码与 notebook。

目录说明：

- `1_环境配置.ipynb`：创建教学环境，安装最新版兼容依赖，并验证 notebook kernel。
- `2_数据处理.ipynb`：展示从反应 SMILES 到反应中心 / synthon / 图张量的完整最小流程。
- `3_模型展示.ipynb`：展示 G2Gs 的中心识别、synthon completion 与整体推理链路。
- `code/g2gs_tutorial.py`：从 `torchdrug` 源码裁剪出的教学版底层代码。
- `data/demo_reactions.csv`：微型教学数据集。
- `data/processed/`：由数据处理 notebook 导出的中间结果。
- `build_notebooks.py`：生成三个 notebook 的脚本。

源码映射：

- `source_repos/torchdrug/torchdrug/data/feature.py`
- `source_repos/torchdrug/torchdrug/datasets/uspto50k.py`
- `source_repos/torchdrug/torchdrug/tasks/retrosynthesis.py`
- `source_repos/torchdrug/torchdrug/models/gcn.py`

教学版原则：

- 保留 G2Gs 的核心数据流与模块边界。
- 移除 `torchdrug` 的注册系统、PackedGraph 抽象与训练框架耦合。
- 用最小 PyTorch + RDKit 实现可运行、可讲解、可逐步调试的版本。
