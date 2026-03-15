# 药物组合协同预测: 宏观图神经网络 (Macro GNN Demo)

本项目是一个端到端的算法教学与演示框架，旨在展示如何利用图神经网络（GNN）构建药物组合协同效应（Drug Synergy）的预测模型。

---

## 🧠 核心架构设计

本项目摒弃了传统简单的特征拼接，采用了图架构。整个前向传播过程分为三个核心阶段：

1. **特征空间映射 (Projection)**：
   考虑到药物节点（1024 维稀疏分子指纹）与细胞系节点（基因表达谱）的初始特征分布与维度差异极大，模型首先通过节点专属的线性层（Linear Projection），将多模态数据对齐到统一的隐层表示空间（Hidden Space）。
2. **消息传递 (Message Passing)**：
   将药物与细胞系视为同一网络中的不同节点。利用 `PyTorch Geometric (PyG)` 的 `T.ToUndirected()` 将单向交互转化为双向无向图，并使用包裹了 `to_hetero` 转换器的 `SAGEConv` 网络，在对齐后的特征空间中实现全局节点间的特征聚合与双向消息传递。
3. **局部子图抽取与预测 (Sub-graph Readout & Prediction)**：
   在每个 Epoch 中，模型首先基于全图拓扑结构更新所有节点的表征。随后，根据 `DataLoader` 传入的当前 Batch 样本（药物A、药物B、细胞系），精准抽取对应节点的最新 Embedding 进行拼接，最后通过 MLP 输出协同得分（Synergy Loewe）的回归预测。

---

## 📁 项目目录结构

```text
synergy_demo_project/
├── data/
│   ├── demo_synergy_labels.csv   # 交互标签 (基于 FLOBAK 子集筛选)
│   ├── demo_drug_features.csv    # 药物特征 (1024D 分子指纹)
│   ├── demo_cell_features.csv    # 细胞系特征 (基因表达谱)
│   ├── macro_hetero_graph.pt     # (首次运行后自动生成) 持久化异质图缓存
│   ├── macro_hetero_model.pth    # (训练结束后生成) 模型权重
│   └── test_predictions.csv      # (训练结束后生成) 测试集预测结果对比
├── run_demo.py                 # 核心主程序：包含图构建、模型定义与端到端训练循环
└── README.md                     # 项目说明文档
```

---

## ⚙️ 环境依赖与安装

本项目推荐在 Linux 服务器或 Windows WSL2 环境下，使用 Conda 进行环境隔离与管理。请按照以下步骤配置你的运行环境：

```bash
# 1. 创建并激活独立环境
conda create -n synergy_gnn python=3.10
conda activate synergy_gnn

# 2. 安装 PyTorch (请根据你本地或集群显卡的 CUDA 版本修改，此处以 CUDA 12.1 为例)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装 PyTorch Geometric 及相关数据处理
pip install torch_geometric pandas
```

---

## 🚀 快速开始

直接运行核心主程序。该脚本会自动加载 CSV 并构建 PyG 异质图对象，按照 8:2 随机划分训练集与测试集，并启动训练流程。训练完成后，模型权重与预测结果会自动保存在 `data/` 目录下。

```bash
python run_demo.py
```


