# 第六章：知识图谱在生物医学中的应用 — 操作手册

本章围绕**知识图谱 + 图表示学习**在生物医学领域的典型应用展开，包含三个小节的完整可运行案例，涵盖知识图谱构建、药物重定位预测和癌症基因识别三大主题。

---

## 目录结构总览

```
kg-chapter6/
├── 6_1/   知识图谱构建与可视化案例（Jupyter Notebook）
├── 6_2/   药物重定位应用（三个子案例 + 药物协同）
│   ├── drug repositioning/dda/HINGRL/     药物-疾病关联预测
│   ├── drug repositioning/dti/iGRLDTI/   药物-靶点相互作用预测
│   ├── drug repositioning/interpretability/TxGNN/  零样本可解释预测
│   └── drug synergy/                      药物组合协同效应预测
└── 6_3/   TREE：基于 Transformer 的癌症基因识别
    └── TREE/
```

---

## 6_1 知识图谱构建与可视化

**目标**：从零开始构建一个生物医学知识图谱，并进行多维度可视化分析。

**文件说明**

| 文件 | 说明 |
|------|------|
| `知识图谱构建案例.ipynb` | 主 Notebook，包含完整的构建与分析流程 |
| `kg_demo.csv` | 演示用知识图谱数据集 |
| `kg_visualization_static.png` | 静态图谱可视化 |
| `kg_visualization_multirel.png` | 多关系图谱可视化 |
| `kg_interactive.html` | 交互式图谱（浏览器打开） |
| `relation_distribution.png` | 关系类型分布图 |
| `degree_analysis.png` | 节点度数分析图 |
| `centrality_analysis.png` | 中心性分析图 |
| `ego_graph_acetaminophen.png` | 对乙酰氨基酚的自我中心子图 |

**运行方式**

```bash
jupyter notebook 6_1/知识图谱构建案例.ipynb
```

---

## 6_2 药物重定位应用

### 案例一：HINGRL — 药物-疾病关联预测

**路径**：`6_2/drug repositioning/dda/HINGRL/`

**核心思路**：构建包含药物、疾病、蛋白质三类节点的异构信息网络（HIN），利用 DeepWalk 生成节点嵌入，拼接节点属性特征后用随机森林进行药物-疾病关联的二分类预测。

```
多源生物数据 → 构建异构信息网络(HIN) → DeepWalk 节点嵌入 → 随机森林分类 → AUC 评估
```

**环境安装**

```bash
pip install numpy pandas scipy scikit-learn matplotlib
# Python >= 3.7
```

**快速运行**

```bash
# B-Dataset，5 折交叉验证，100 棵决策树
python run_demo.py -d 1 -f 5 -n 100

# F-Dataset
python run_demo.py -d 2 -f 5 -n 100
```

**参数说明**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-d` | 数据集（1=B-Dataset, 2=F-Dataset） | 1 |
| `-f` | K 折交叉验证折数 | 5 |
| `-n` | 随机森林决策树棵数 | 100 |

**预期结果**：B-Dataset 上平均 AUC ≈ 0.95+，ROC 曲线保存至 `data/tmp/roc_B-Dataset_5fold.png`

**数据集规模**

| 数据集 | 药物数 | 疾病数 | 正样本数 |
|--------|--------|--------|---------|
| B-Dataset（`-d 1`） | ~269 | ~598 | ~18,416 |
| F-Dataset（`-d 2`） | ~593 | ~313 | ~1,933 |

---

### 案例二：iGRLDTI — 药物-靶点相互作用预测

**路径**：`6_2/drug repositioning/dti/iGRLDTI/`

**核心思路**：将药物与靶点蛋白构建为异质生物信息网络，通过**非线性扩散局部平滑（NDLS）**为每个节点自适应确定最优传播跳数，获取节点嵌入后用梯度提升树（GBM）预测药物-靶点相互作用（DTI）。

```
原始图数据 → 构建增广随机游走归一化邻接矩阵 → NDLS 特征平滑 → DNN 投影 → GBM 10折交叉验证 → AUC/AUPRC
```

**环境安装**

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
# Python >= 3.6, PyTorch >= 1.4
```

**快速运行**

```bash
# 在 iGRLDTI/ 目录下
python run_demo.py
```

**数据说明**

| 文件 | 说明 | 规模 |
|------|------|------|
| `Allnode_DrPr.csv` | 节点索引（0=药物, 1=蛋白质） | 973 节点 |
| `DrPrNum_DrPr.csv` | 正样本：已知 DTI 边 | 1,923 条 |
| `AllNodeAttribute_DrPr.csv` | 节点特征向量 | 973 × 64 |
| `AllNegative_DrPr.csv` | 负样本候选池 | 230,853 条 |

**关键参数**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `k1` | 200 | NDLS 最大传播跳数 |
| `epsilon1` | 0.03 | NDLS 收敛阈值 |
| `hidden` | 64 | DNN 嵌入维度 |
| `N_ESTIMATORS` | 499 | GBM 弱学习器数量 |

---

### 案例三：TxGNN — 零样本可解释药物重定向

**路径**：`6_2/drug repositioning/interpretability/TxGNN/`

**核心思路**：基于包含 17,080 种疾病、7,957 种药物的大规模生物医学知识图谱，利用图神经网络预训练 + 度量学习微调，实现对罕见病（训练集中无已知治疗药物）的**零样本药物重定向预测**，并通过 GraphMask 提供可解释性分析。

```
知识图谱 → GNN 预训练（30种关系）→ 度量学习微调（相似疾病辅助）→ 零样本推断
```

**环境安装**

```bash
# Step 1：创建 Conda 环境
conda create --name txgnn_env python=3.8
conda activate txgnn_env

# Step 2：安装 PyTorch（根据 CUDA 版本选择）
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# 无 GPU：conda install pytorch==1.12.1 torchvision torchaudio cpuonly -c pytorch

# Step 3：安装 DGL（版本必须为 0.5.2）
conda install -c dglteam dgl-cuda11.3==0.5.2
# 无 GPU：conda install -c dglteam dgl==0.5.2

# Step 4：安装 TxGNN
pip install -e .
```

**核心工作流**

```python
from txgnn import TxData, TxGNN, TxEval

# 1. 加载数据（首次运行自动下载 ~400MB）
TxData = TxData(data_folder_path='./data')
TxData.prepare_split(split='complex_disease', seed=42)

# 2. 初始化模型
TxGNN = TxGNN(data=TxData, device='cpu')  # 无 GPU 改为 'cpu'
TxGNN.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

# 3. 加载预训练权重（教学演示推荐）
TxGNN.load_pretrained('./model_ckpt')

# 4. 微调
TxGNN.finetune(n_epoch=30, learning_rate=5e-4, train_print_per_n=5)

# 5. 评估
TxEval = TxEval(model=TxGNN)
result = TxEval.eval_disease_centric(disease_idxs='test_set', verbose=True)

# 6. 可解释性分析
TxGNN.train_graphmask(relation='indication', learning_rate=3e-4)
gates = TxGNN.retrieve_save_gates('./model_ckpt')
```

**完整演示**：打开 `TxGNN_Demo.ipynb`

**数据集划分策略**

| 划分名称 | 说明 |
|---------|------|
| `complex_disease`（推荐） | 测试疾病在训练集中无任何已知药物（零样本场景） |
| `random` | 随机划分（基线对比） |
| `full_graph` | 全量训练，无测试集（部署前使用） |

---

### 案例四：药物组合协同预测（Macro GNN）

**路径**：`6_2/drug synergy/`

**核心思路**：将药物与细胞系建模为异质图节点，通过三阶段图神经网络（特征映射 → 消息传递 → 子图读出）预测药物组合的协同效应（Synergy Loewe 分数）。

```
药物特征（1024D 分子指纹）+ 细胞系特征（基因表达谱）→ 异质图 → SAGEConv 消息传递 → MLP 回归
```

**环境安装**

```bash
# 创建环境
conda create -n synergy_gnn python=3.10
conda activate synergy_gnn

# 安装 PyTorch（以 CUDA 12.1 为例）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装 PyTorch Geometric
pip install torch_geometric pandas
```

**快速运行**

```bash
python run_demo.py
```

脚本自动完成：加载 CSV → 构建 PyG 异质图 → 8:2 划分训练/测试集 → 训练 → 保存权重与预测结果至 `data/` 目录。

**数据文件**

| 文件 | 说明 |
|------|------|
| `data/demo_synergy_labels.csv` | 交互标签（FLOBAK 子集） |
| `data/demo_drug_features.csv` | 药物特征（1024D 分子指纹） |
| `data/demo_cell_features.csv` | 细胞系特征（基因表达谱） |

---

## 6_3 TREE：基于 Transformer 的癌症基因识别

**路径**：`6_3/TREE/`

**论文**：*Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning*，Nature Biomedical Engineering，2025

**核心思路**：将 Transformer（Graphormer）引入图学习，通过多通道随机游走子图采样、空间编码（最短路径距离）、中心性编码（节点度数）和注意力融合，从多组学网络中识别癌症驱动基因，并提供 SHAP 可解释性分析。

```
节点 ID → 随机游走采样多子图 → 中心性编码 + 多组学特征 → Graphormer 编码器 → 注意力融合 → 二分类（癌症基因概率）
```

**环境安装**

```bash
# Python 3.6–3.8（推荐 3.8）
pip install h5py==3.1.0 keras==2.6.0 numpy==1.19.5 networkx==2.5.1
pip install pandas==1.1.5 scikit-learn==0.24.2 tensorflow==2.6.2 shap==0.41.0
```

**数据下载**

| 数据类型 | 下载地址 | 存放路径 |
|----------|----------|----------|
| Pan-cancer 网络（.h5） | [Zenodo 11648891](https://zenodo.org/records/11648891) | `dataset/networks/` |
| 预计算子图（pdata，可选） | [Zenodo 15045885](https://zenodo.org/records/15045885) | `pdata/` |
| 预计算最短路径（sp，可选） | [Zenodo 15045711](https://zenodo.org/records/15045711) | `sp/` |

> `pdata/` 和 `sp/` 不下载时首次运行自动生成（耗时较长）。

**快速运行**（课堂演示）

```bash
# 在 TREE/ 目录下
python run_demo.py
```

默认使用 CPDB 数据集（约 2,000 节点，规模最小），可修改脚本顶部参数：

```python
N_EPOCH  = 10   # 训练轮次（论文为 100）
CV_FOLDS = 3    # 交叉验证折数（论文为 10）
```

**复现论文完整结果**

```bash
python run.py
```

**评估指标**

| 指标 | 说明 |
|------|------|
| AUC | ROC 曲线下面积，衡量整体排序能力 |
| AUPR | PR 曲线下面积，对类别不平衡更敏感 |
| ACC | 准确率（阈值 0.5） |
| F1 | 兼顾精确率与召回率 |

**输出文件**

| 路径 | 内容 |
|------|------|
| `log/train_history.txt` | 每 epoch 验证集指标 |
| `log/{dataset}_result.txt` | K 折汇总结果 |
| `ckpt/{exp_name}.hdf5` | 最佳模型权重 |
| `pdata/` | 随机游走子图缓存 |

---

## 各案例一览

| 小节 | 案例 | 任务类型 | 核心模型 | 运行入口 |
|------|------|----------|----------|----------|
| 6_1 | 知识图谱构建 | 图构建与可视化 | NetworkX | `知识图谱构建案例.ipynb` |
| 6_2 | HINGRL | 药物-疾病关联预测 | DeepWalk + 随机森林 | `run_demo.py -d 1` |
| 6_2 | iGRLDTI | 药物-靶点相互作用预测 | NDLS + GBM | `run_demo.py` |
| 6_2 | TxGNN | 零样本药物重定向 | GNN + 度量学习 | `TxGNN_Demo.ipynb` |
| 6_2 | Macro GNN | 药物协同效应预测 | SAGEConv 异质图 | `run_demo.py` |
| 6_3 | TREE | 癌症基因识别 | Graphormer + SHAP | `run_demo.py` |


| … | （持续更新中） | — | — | — |

---

## 后续更新说明

本手册将随教材内容持续迭代，后续会陆续补充新的应用案例。每次新增案例后，请按以下规范同步更新本文档：

1. **目录结构总览**：在对应小节路径下追加新案例路径
2. **新增内容节**：按已有格式撰写（背景 → 算法流程 → 环境安装 → 快速运行 → 参数/数据说明）
3. **各案例一览表**：在表格末尾补充一行（小节、案例名、任务类型、核心模型、运行入口）
4. **参考文献**：追加对应论文信息

> 如发现文档描述与实际代码不一致，欢迎直接在对应小节修改并注明修改原因。

---

## 参考文献

- **PrimeKG**：Chandak, et al. (2022). *Scientific Data*.
- **HINGRL**：Zhao B W, et al. (2022). *Briefings in Bioinformatics*.
- **iGRLDTI**：Zhao B W, et al. (2022).*Bioinformatics*.
- **TxGNN**：Huang K, et al. (2023). *Nature Medicine*.
- **TREE**：Su X, et al. (2025). *Nature Biomedical Engineering*.
- **DeepWalk**：Perozzi B, et al. (2014). *KDD 2014*.
