# 第六章：知识图谱在生物医学中的应用 — 操作手册

本章围绕**知识图谱 + 图表示学习**在生物医学领域的典型应用展开，包含三个小节的完整可运行案例，涵盖知识图谱构建、药物重定位预测和癌症基因识别三大主题。

---

## 目录结构总览

```
kg-chapter6/
├── 6_1/   知识图谱构建与可视化案例（Jupyter Notebook）
├── 6_2/   药物重定位应用（三个子案例 + 药物协同 + Notebook 构建脚本）
│   ├── drug repositioning/dda/HINGRL/            药物-疾病关联预测
│   ├── drug repositioning/dti/iGRLDTI/           药物-靶点相互作用预测
│   ├── drug repositioning/interpretability/TxGNN/  零样本可解释预测
│   ├── drug synergy/                             药物组合协同效应预测
│   └── notebooks/                                教学 Notebook 生成与整理脚本
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

**定位**：当前目录已整理为课堂教学版，主入口为 `HINGRL_demo.ipynb`。

**核心思路**：构建包含药物、疾病、蛋白质三类节点的异构网络，利用节点属性特征与 DeepWalk 图嵌入共同表示药物-疾病样本，再通过监督学习完成关联预测。

```
多源生物数据 → 构建异构信息网络(HIN) → DeepWalk 节点嵌入 → 随机森林分类 → AUC 评估
```

**目录结构**

```text
HINGRL/
├─ data/
│  ├─ B-Dataset/
│  └─ F-Dataset/
├─ demo_outputs/
│  └─ hingrl_b_dataset_metrics.png
├─ HINGRL_demo.ipynb
└─ README.md
```

**主要文件**

| 文件 | 说明 |
|------|------|
| `HINGRL_demo.ipynb` | 课堂教学版 Notebook，主体展示 `B-Dataset`，并预留 `F-Dataset` 课后练习 |
| `data/B-Dataset/` | 课堂主线使用的数据集 |
| `data/F-Dataset/` | 迁移练习数据集 |
| `demo_outputs/hingrl_b_dataset_metrics.png` | Notebook 生成的课堂结果图 |

**数据说明**

| 文件 | 说明 |
|------|------|
| `DrDiNum.csv` | 药物-疾病关联 |
| `DrPrNum.csv` | 药物-蛋白关联 |
| `DiPrNum.csv` | 疾病-蛋白关联 |
| `AllNodeAttribute.csv` | 节点属性特征 |
| `AllEmbedding_DeepWalk.txt` | 预计算 DeepWalk 嵌入 |

样本特征拼接形式为：

```text
[attr(drug) | emb(drug) | attr(disease) | emb(disease)]
```

`data/README.md` 进一步说明了当前仓库只版本化数据契约与已准备好的嵌入文件，默认不重新运行 OpenNE。

**环境依赖**

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

**推荐运行方式**

```bash
jupyter notebook 6_2/drug repositioning/dda/HINGRL/HINGRL_demo.ipynb
```

**教学建议**：课堂以 `B-Dataset` 为主，`F-Dataset` 更适合作为课后独立迁移练习。

---

### 案例二：iGRLDTI — 药物-靶点相互作用预测

**路径**：`6_2/drug repositioning/dti/iGRLDTI/`

**定位**：当前目录已整理为课堂教学版，主入口为 `iGRLDTI_demo.ipynb`。

**核心思路**：将药物与靶点构造成异质生物信息网络，通过 NDLS-F 缓解图传播中的过平滑问题，再由节点嵌入构造药物-靶点样本完成预测。

```
原始图数据 → 构建增广随机游走归一化邻接矩阵 → NDLS 特征平滑 → DNN 投影 → GBM 10折交叉验证 → AUC/AUPRC
```

**目录结构**

```text
iGRLDTI/
├─ data/
│  ├─ Allnode_DrPr.csv
│  ├─ DrPrNum_DrPr.csv
│  ├─ AllNodeAttribute_DrPr.csv
│  ├─ AllNegative_DrPr.csv
│  └─ Emdebding_GCN2_DrPr.csv
├─ results/
│  └─ roc_curve.png
├─ src/
│  ├─ main.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ iGRLDTI_demo.ipynb
└─ README.md
```

**主要文件**

| 文件 | 说明 |
|------|------|
| `iGRLDTI_demo.ipynb` | 课堂教学版 Notebook |
| `src/utils.py` | 图预处理与数据加载工具 |
| `src/train.py` | NDLS-F 与训练辅助函数 |
| `src/model.py` | DNN 投影模型定义 |
| `results/roc_curve.png` | 运行后的 ROC 曲线图 |

**数据说明**

| 文件 | 说明 |
|------|------|
| `Allnode_DrPr.csv` | 节点列表 |
| `DrPrNum_DrPr.csv` | 已知药物-靶点相互作用边 |
| `AllNodeAttribute_DrPr.csv` | 节点属性特征 |
| `AllNegative_DrPr.csv` | 候选负样本池 |

课堂讲解可以重点强调：正样本来自已知 DTI 边，负样本来自未知配对随机采样，最终样本特征由药物嵌入与靶点嵌入拼接得到。

**环境依赖**

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
```

**推荐运行方式**

```bash
jupyter notebook 6_2/drug repositioning/dti/iGRLDTI/iGRLDTI_demo.ipynb
```

**教学建议**：建议把重点放在 NDLS-F 如何缓解过平滑、为什么同时关注 `AUC` 与 `AUPRC`，以及 5 折与 10 折结果差异上。

---

### 案例三：TxGNN — 零样本可解释药物重定向

**路径**：`6_2/drug repositioning/interpretability/TxGNN/`

**定位**：该目录提供 TxGNN 的教学版仓库，主入口为 `TxGNN_Demo.ipynb`，并保留 `reproduce/` 目录用于论文结果复现。

**核心思路**：基于大规模生物医学知识图谱，利用 GNN 预训练与度量学习微调，实现对训练集中无已知治疗药物疾病的零样本药物重定向预测，并通过 GraphMask 进行可解释性分析。

```
知识图谱 → GNN 预训练（30种关系）→ 度量学习微调（相似疾病辅助）→ 零样本推断
```

**项目结构**

```text
TxGNN/
├── TxGNN/                  # 核心模块
├── data/
│   └── disease_files/      # 九大疾病领域节点列表
├── reproduce/              # 论文复现脚本
├── fig/
├── TxGNN_Demo.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

**环境安装**

```bash
conda create --name txgnn_env python=3.8
conda activate txgnn_env

conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c dglteam dgl-cuda11.3==0.5.2
pip install -e .
```

无 GPU 时，可分别改用 `cpuonly` 的 PyTorch 与 `dgl==0.5.2`。

**推荐运行方式**

```bash
jupyter notebook 6_2/drug repositioning/interpretability/TxGNN/TxGNN_Demo.ipynb
```

**核心工作流**

```python
from txgnn import TxData, TxGNN, TxEval

# 1. 加载数据（首次运行自动下载约 400MB）
TxData = TxData(data_folder_path='./data')
TxData.prepare_split(split='complex_disease', seed=42)

# 2. 初始化模型
TxGNN = TxGNN(data=TxData, device='cpu')
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

**补充说明**

| 目录/文件 | 说明 |
|----------|------|
| `TxGNN_Demo.ipynb` | 课堂演示主入口 |
| `reproduce/run_txgnn.sh`、`reproduce/train.py` | 论文复现实验脚本 |
| `reproduce/result_more_metrics.csv` | 复现实验原始指标 |

`reproduce/README.md` 中还给出了论文 checkpoint 下载地址，可通过 `TxGNN.load_pretrained` 直接载入评估。

**数据集划分策略**

| 划分名称 | 说明 |
|---------|------|
| `complex_disease`（推荐） | 测试疾病在训练集中无任何已知药物（零样本场景） |
| `random` | 随机划分（基线对比） |
| `full_graph` | 全量训练，无测试集（部署前使用） |

---

### 案例四：药物组合协同预测（Macro GNN）

**路径**：`6_2/drug synergy/`

**定位**：当前目录已整理为课堂教学版，主入口为 `drug_synergy_demo.ipynb`。

**核心思路**：将药物与细胞系建模为异质图节点，通过异构 GraphSAGE 学习 `[drugA, drugB, cell]` 的联合表示，完成协同效应回归预测。

```
药物特征（1024D 分子指纹）+ 细胞系特征（基因表达谱）→ 异质图 → SAGEConv 消息传递 → MLP 回归
```

**目录结构**

```text
drug synergy/
├─ data/
│  ├─ demo_synergy_labels.csv
│  ├─ demo_drug_features.csv
│  ├─ demo_cell_features.csv
│  ├─ macro_hetero_graph.pt
│  ├─ macro_hetero_model.pth
│  └─ test_predictions.csv
├─ results/
│  ├─ loss_curve.png
│  └─ prediction_scatter.png
├─ drug_synergy_demo.ipynb
└─ README.md
```

**环境依赖**

```bash
pip install torch torch_geometric pandas matplotlib
```

**推荐运行方式**

```bash
jupyter notebook "6_2/drug synergy/drug_synergy_demo.ipynb"
```

**主要文件**

| 文件 | 说明 |
|------|------|
| `data/demo_synergy_labels.csv` | 药物组合在不同细胞系上的协同分数标签 |
| `data/demo_drug_features.csv` | 药物分子特征 |
| `data/demo_cell_features.csv` | 细胞系特征（基因表达谱） |
| `results/loss_curve.png` | 训练曲线 |
| `results/prediction_scatter.png` | 测试集预测散点图 |

**教学建议**：这是一个回归任务，不应再以 `AUC` 为主，而应围绕 `MSE`、`MAE`、`RMSE` 和相关性解释模型效果。

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
| 6_2 | HINGRL | 药物-疾病关联预测 | DeepWalk + 属性特征 + 随机森林 | `HINGRL_demo.ipynb` |
| 6_2 | iGRLDTI | 药物-靶点相互作用预测 | NDLS-F + DNN + GBM | `iGRLDTI_demo.ipynb` |
| 6_2 | TxGNN | 零样本药物重定向 | GNN + 度量学习 | `TxGNN_Demo.ipynb` |
| 6_2 | Macro GNN | 药物协同效应预测 | 异构 GraphSAGE + MLP 回归 | `drug_synergy_demo.ipynb` |
| 6_3 | TREE | 癌症基因识别 | Graphormer + SHAP | `run_demo.py` |

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

- **HINGRL**：Zhao B W, et al. (2022). *Briefings in Bioinformatics*, 23(4).
- **iGRLDTI**：*Bioinformatics*, 2023.
- **TxGNN**：Huang K, et al. (2023). *medRxiv*. doi:10.1101/2023.03.19.23287458
- **TREE**：Su X, et al. (2025). *Nature Biomedical Engineering*.
- **DeepWalk**：Perozzi B, et al. (2014). *KDD 2014*.
