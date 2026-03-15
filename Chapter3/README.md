# Chapter 3: 虚拟筛选（Virtual Screening）

本章介绍图神经网络在虚拟筛选中的应用，涵盖三个核心任务：

1. **结合位点预测**（Binding Site Prediction）— 预测配体在蛋白质上的结合位置
2. **结合构象预测**（Binding Pose Prediction: Molecular Docking）— 预测配体与蛋白质的结合构象
3. **结合亲和力预测**（Binding Affinity Prediction: Scoring）— 预测蛋白质-配体复合物的结合亲和力

每类任务中，不同的技术路线选择了一个具有代表性的工作进行解释

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
    ├── 2.binding_pose_prediction/
    │   ├── 2.1.semi_flexible/  # 半柔性对接：DiffDock, TankBind, KarmaDock, CarsiDock, MetalloDock
    │   └── 2.2.flexible/      # 柔性对接：FlexPose, CarsiInduce, DynamicBind, DiffDock-Pocket
    └── 3.binding_affinity_prediction/              # 打分：IGN, PIGNet, RTMScore
```

## 模型列表

### 1. 结合位点预测

| 模型 | 技术路线 |
|------|------|
| GrASP | 通过图神经网络预测每个蛋白节点是结合位点的概率 |

### 2. 分子对接

**半柔性对接（受体刚性）：**

| 模型 | 技术路线 |
|------|------|
| DiffDock | 采用扩散模型建模分子在口袋中的变换（平移+旋转+扭转）|
| TankBind | 预测距离矩阵，然后通过梯度下降算法重建分子结合构象 |
| CarsiDock | 预测距离矩阵，然后通过构象搜索算法重建分子结合构象 |
| KarmaDock | 预测分子坐标的偏移量，逐步更新分子构象 |
| MetalloDock | 采用自回归算法，以口袋内金属离子作为起点，逐原子预测分子的结合构象 |

**柔性对接（受体和配体均柔性）：**

| 模型 | 技术路线 |
|------|------|
| FlexPose | 同时预测蛋白和分子配体的结合构象 |
| CarsiInduce | 根据分子配体预测蛋白的结合态（holo），再采用CarsiDock把分子对接到holo蛋白中 |
| DiffDock-Pocket | 利用扩散模型预测分子在口袋中的变换（平移+旋转+扭转）以及蛋白口袋残基侧链的扭转角 |
| DynamicBind | 利用扩散模型预测蛋白残基和小分子的变换（平移+旋转+扭转） |

### 3. 打分

| 模型 | 技术路线 |
|------|------|
| IGN | 模型直接拟合实验解析的亲和力数值 |
| PIGNet | 构造物理启发的相互作用项（vdW + H-bond + 疏水）来拟合实验解析的亲和力数值 |
| RTMScore | 通过混合密度网络拟合蛋白残基-配体的最小距离的分布，根据测试样本的距离分布与模型学习得到的分布的似然来近似结合亲和力，而非拟合实验解析的亲和力数值 |

## 数据集

教学演示使用 PDBbind CASF-2016 core set 的 20 个代表性复合物子集，覆盖不同靶标类别和亲和力范围。

## 快速开始

每个模型的教学 notebook 是自包含的，可以独立运行：

