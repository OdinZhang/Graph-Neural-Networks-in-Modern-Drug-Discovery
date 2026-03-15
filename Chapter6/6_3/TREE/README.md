# TREE：基于 Transformer 的癌症基因识别框架

> **论文**：Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning
> **期刊**：*Nature Biomedical Engineering*, 2025
> **关键词**：图表示学习 · Graphormer · 癌症基因识别 · 多组学 · 可解释性

![TREE框架架构](TREE_architecture.png)

---

## 1. 研究背景

癌症基因的识别是精准医学的核心任务之一。现有方法大多依赖单一网络或单一组学数据，难以同时捕获**多组学特征**和**网络拓扑结构**信息。

TREE（**T**ransformer-powered g**R**aph r**E**presentation l**E**arning）的核心思路：

| 挑战 | TREE 的解决方案 |
|------|----------------|
| 网络拓扑复杂，节点关系难以建模 | Graphormer 编码器 + 空间编码（最短路径） |
| 局部结构多样性难以捕获 | 多通道随机游走子图采样 |
| 节点重要性差异大 | 中心性编码（Centrality Encoding） |
| 多组学特征融合困难 | 注意力融合层（Attention Fusion） |
| 预测结果缺乏可解释性 | SHAP 值 + 注意力权重可视化 |

---

## 2. 框架架构

```
输入：节点 ID
  │
  ▼
┌─────────────────────────────────────────────┐
│  子图采样（随机游走，n_graphs 个通道）          │  ← Step 1
│  每个节点 → n_graphs 个含 n_neighbors 邻居的子图│
└──────────────────────┬──────────────────────┘
                       │ 对每个子图通道 g：
  ┌────────────────────▼────────────────────┐
  │  中心性编码 CentralityEncoding           │  ← 节点度数 → 嵌入
  │  + 节点多组学特征（基因表达、突变等）      │
  └────────────────────┬────────────────────┘
                       │
  ┌────────────────────▼────────────────────┐
  │  Graphormer Block × n_layers            │  ← 核心编码器
  │    ├─ 多头自注意力（带空间编码偏置）      │
  │    └─ 前馈网络 FFN                      │
  └────────────────────┬────────────────────┘
                       │ 取目标节点嵌入
  ┌────────────────────▼────────────────────┐
  │  注意力融合层 AttentionFusion            │  ← 融合 n_graphs 个通道
  └────────────────────┬────────────────────┘
                       │
  ┌────────────────────▼────────────────────┐
  │  二分类器（Sigmoid）                     │  → 癌症基因概率
  └─────────────────────────────────────────┘
```

### 关键模块说明

| 模块 | 文件 | 作用 |
|------|------|------|
| `CentralityEncoding` | `layers/centralityEncoding.py` | 将节点度数映射为嵌入向量，编码节点中心性 |
| `SpatialEncoding` | `layers/spatialEncoding.py` | 将子图内最短路径距离映射为注意力偏置 |
| `MultiHeadAttention` | `layers/multiHeadAttention.py` | 带空间编码偏置的多头自注意力 |
| `GraphormerBlock` | `layers/graphormerEncoder.py` | Graphormer 编码块（Pre-LN + 残差连接） |
| `AttentionFusion` | `layers/attentionFusion.py` | 跨通道注意力融合，整合多子图表示 |
| `WeightedBinaryCrossEntropy` | `losses/weightedBinaryCrossEntropy.py` | 处理正负样本严重不平衡的加权损失 |
| `TREE` | `models/tree.py` | 主模型（端到端图分类） |
| `InterTREE` | `models/interpre_tree.py` | 可解释版本（用于 SHAP 分析） |

---

## 3. 目录结构

```
TREE/
├── run_demo.py          ← 课堂演示脚本（推荐新手从此开始）
├── run.py               ← 原始完整训练脚本
├── main.py              ← 训练主函数（train）
├── config.py            ← 全局配置（路径、超参数模板）
│
├── layers/              ← 核心神经网络层
│   ├── centralityEncoding.py   中心性编码
│   ├── spatialEncoding.py      空间编码
│   ├── multiHeadAttention.py   多头注意力
│   ├── graphormerEncoder.py    Graphormer 块
│   └── attentionFusion.py      注意力融合
│
├── models/              ← 完整模型
│   ├── tree.py                 TREE 主模型
│   ├── interpre_tree.py        TREE 可解释版本
│   └── base_model.py           基类（训练/评估/加载）
│
├── losses/              ← 损失函数
│   └── weightedBinaryCrossEntropy.py
│
├── callbacks/           ← 训练回调
│   ├── eval.py                 每 epoch 计算 AUC/AUPR/F1
│   └── ensemble.py             SWA 模型集成
│
├── utils/               ← 工具函数
│   ├── node2vec.py             Node2Vec 随机游走
│   ├── walker.py               图游走器
│   ├── data_loader.py          数据加载
│   └── io.py                   文件读写工具
│
├── dataset/networks/    ← 原始网络数据（.h5 文件，需下载）
├── pdata/               ← 预处理子图缓存（自动生成或下载）
├── sp/                  ← 最短路径矩阵缓存（自动生成或下载）
├── ckpt/                ← 训练好的模型权重（自动生成）
└── log/                 ← 训练日志（自动生成）
```

---

## 4. 环境安装

**Python 版本**：推荐 Python 3.6–3.8

```bash
pip install h5py==3.1.0
pip install keras==2.6.0
pip install numpy==1.19.5
pip install networkx==2.5.1
pip install pandas==1.1.5
pip install scikit-learn==0.24.2
pip install tensorflow==2.6.2
pip install shap==0.41.0
```

---

## 5. 数据下载

| 数据类型 | 下载链接 | 存放路径 |
|----------|----------|----------|
| Pan-cancer 网络（.h5） | [Zenodo 11648891](https://zenodo.org/records/11648891) | `dataset/networks/` |
| 预计算子图（pdata） | [Zenodo 15045885](https://zenodo.org/records/15045885) | `pdata/` |
| 预计算最短路径（sp） | [Zenodo 15045711](https://zenodo.org/records/15045711) | `sp/` |

> **注意**：`pdata/` 和 `sp/` 可不下载，首次运行时会自动生成（耗时较长）。

---

## 6. 快速上手

```bash
python run_demo.py
```

脚本固定使用 **CPDB** 数据集（同构 PPI 网络，约 2,000 节点，规模最小，适合课堂演示）。

修改脚本顶部的参数即可调整训练行为：

```python
N_EPOCH     = 10    # 训练轮次（论文为 100）
CV_FOLDS    = 3     # 交叉验证折数（论文为 10）
```

若需复现论文完整结果，运行原始脚本：

```bash
python run.py
```

---

## 7. 核心概念讲解

### 7.1 为什么用 Graphormer？

传统 GNN（如 GCN、GAT）通过消息传递聚合**直接邻居**信息，难以捕获长距离依赖。
Graphormer 将 Transformer 引入图学习，利用**全局自注意力**让每个节点可以关注图中任意节点，并通过**空间编码**（最短路径距离）为注意力注入图结构信息。

### 7.2 随机游走子图采样

```
目标节点 v
  ├── 子图 1（游走路径）: v → u₁ → u₂ → u₃ → ...
  ├── 子图 2（游走路径）: v → u₄ → u₁ → u₅ → ...
  └── 子图 k（游走路径）: v → u₂ → u₆ → u₁ → ...
```
每条游走路径捕获节点周围的一种**局部拓扑视角**，多条路径共同描述节点的多样邻域结构。

### 7.3 空间编码（Spatial Encoding）

子图内节点对 (i, j) 的最短路径距离 d(i,j) 通过两层 Dense 网络映射为**注意力偏置**，叠加到自注意力得分上：

```
Attention(Q, K, V) = softmax((QKᵀ/√d + B_spatial) · mask) · V
```

其中 `B_spatial[i,j]` 编码了节点 i 到 j 的结构距离——距离越近，注意力偏置越大。

### 7.4 中心性编码（Centrality Encoding）

节点度数（Degree）反映其在网络中的重要性（中心性）。TREE 将度数通过 Embedding 层映射为与特征等维度的向量，**叠加到节点特征上**：

```
node_input = node_feature × √d_model + centrality_embedding(degree)
```

### 7.5 加权损失函数

癌症基因（正样本）在基因组中极少（约占 1–5%），存在严重**类别不平衡**问题。
TREE 使用加权二元交叉熵，自动提升正样本的损失权重：

```python
# loss_mul 为正样本比例（如 0.27 表示正样本占 27%）
sample_weight = (1 - loss_mul) / loss_mul  # 负/正 权重比
```

---

## 8. 评估指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| AUC | ROC 曲线下面积，衡量整体排序能力 | `roc_auc_score` |
| AUPR | PR 曲线下面积，对不平衡数据更敏感 | `auc(recall, precision)` |
| ACC | 准确率（阈值 0.5） | `accuracy_score` |
| F1 | F1 分数，兼顾精确率与召回率 | `f1_score` |

实验采用 **10 折交叉验证**，报告均值 ± 标准差。

---

## 9. 输出文件说明

| 文件/目录 | 内容 |
|-----------|------|
| `log/train_history.txt` | 每 epoch 的验证集 AUC/AUPR/ACC/F1 |
| `log/TREE_performance` | 每折训练的完整评估指标 |
| `log/{dataset}_result.txt` | 各数据集最终 K 折汇总结果 |
| `ckpt/{exp_name}.hdf5` | 最佳模型权重（按 val_AUC 选择） |
| `pdata/` | 随机游走子图缓存（首次运行后生成） |
| `sp/{dataset}_sp.h5` | 最短路径矩阵缓存 |

---

## 10. 引用

```bibtex
@article{su2025interpretable,
  title   = {Interpretable identification of cancer genes across biological
             networks via transformer-powered graph representation learning},
  author  = {Su, Xiaorui and Hu, Pengwei and Li, Dongxu and Zhao, Bowei and
             Niu, Zhaomeng and Herget, Thomas and Yu, Philip S and Hu, Lun},
  journal = {Nature Biomedical Engineering},
  pages   = {1--19},
  year    = {2025},
  publisher = {Nature Publishing Group UK London}
}
```
