# Chapter 3: 虚拟筛选（Virtual Screening）

本章介绍图神经网络在虚拟筛选中的应用，涵盖三个核心任务：

1. **结合位点预测**（Binding Site Prediction）— 预测配体在蛋白质上的结合位置
2. **分子对接**（Molecular Docking）— 预测配体与蛋白质的结合构象
3. **打分**（Scoring）— 预测蛋白质-配体复合物的结合亲和力

## 目录结构

```
Chapter3/
├── demo_data/                  # 教学用小数据集（CASF-2016 core set 中 20 个复合物）
│   ├── CoreSet.dat             # 标注信息
│   └── coreset/                # 每个复合物：蛋白PDB、口袋PDB、配体MOL2/SDF
├── source_repos/               # 原始模型代码仓库（git submodule）
└── teaching_demos/             # 教学 Jupyter Notebook
    ├── 1.binding_site_prediction/
    │   └── 1.1.grasp/          # GrASP
    ├── 2.docking/
    │   ├── 2.1.semi_flexible/  # 半柔性对接：DiffDock, TankBind, KarmaDock, CarsiDock, MetalloDock
    │   └── 2.2.flexible/      # 柔性对接：FlexPose, CarsiInduce, DynamicBind, DiffDock-Pocket
    └── 3.scoring/              # 打分：IGN, PIGNet, RTMScore
```

## 模型列表

### 1. 结合位点预测

| 模型 | 方法 | 论文 |
|------|------|------|
| GrASP | GATv2 逐原子分类 + 软标签 + Noisy Nodes | Graph-based Site Prediction |

### 2. 分子对接

**半柔性对接（受体刚性）：**

| 模型 | 方法 | 论文 |
|------|------|------|
| DiffDock | SE(3) 扩散模型（平移+旋转+扭转） | Corso et al., 2023 |
| TankBind | 三角几何感知 GNN | Lu et al., 2022 |
| KarmaDock | E(n) 等变 GNN | Zhang et al., 2023 |
| CarsiDock | 距离矩阵预测 | Hua et al., 2024 |
| MetalloDock | 金属蛋白专用对接 | - |

**柔性对接（受体和配体均柔性）：**

| 模型 | 方法 | 论文 |
|------|------|------|
| FlexPose | 端到端柔性对接 | Dong et al., 2023 |
| CarsiInduce | 诱导拟合对接 | - |
| DynamicBind | 动态结合预测 | Lu et al., 2024 |
| DiffDock-Pocket | 口袋柔性扩散 | Plainer et al., 2024 |

### 3. 打分

| 模型 | 方法 | 论文 |
|------|------|------|
| IGN | 交互图 + 边级消息传递 | Jiang et al., 2021 |
| PIGNet | 物理信息 GNN（vdW + H-bond + 疏水） | Moon et al., 2022 |
| RTMScore | MDN + 残基-原子距离分布 | Shen et al., 2022 |

## 数据集

教学演示使用 PDBbind CASF-2016 core set 的 20 个代表性复合物子集，覆盖不同靶标类别和亲和力范围。

## 快速开始

每个模型的教学 notebook 是自包含的，可以独立运行：

```bash
cd teaching_demos/3.scoring/3.1.ign/
jupyter notebook IGN_虚拟筛选教程.ipynb
```
