# iGRLDTI — 基于图表示学习的药物-靶点相互作用预测

> 教学案例 | 第6章 · 知识图谱与药物重定位

## 1. 案例背景

**药物重定位（Drug Repositioning）** 是指为已上市或已在研发中的药物寻找新的适应症，
相比从头研发新药可显著缩短研发周期和降低成本。

本案例基于论文：
> *iGRLDTI: An Improved Graph Representation Learning Method for Predicting Drug-Target Interactions over Heterogeneous Biological Information Network*

核心思想：将药物与靶点蛋白的异质生物信息网络建模为图，通过图上的 **非线性扩散局部平滑（NDLS）** 获取节点嵌入，再用机器学习分类器预测未知的药物-靶点相互作用（DTI）。

---

## 2. 算法流程

```
原始图数据（节点特征 + 边列表）
        │
        ▼
  [图预处理] 构建增广随机游走归一化邻接矩阵 Â
        │
        ▼
  [NDLS-F] 非线性扩散局部平滑
    ├─ 逐跳传播特征：X^(k) = Â · X^(k-1)
    ├─ 计算稳态特征：X_∞ = π^T · X
    └─ 为每个节点确定最优跳数 h_v（收敛阈值 ε₁）
        │
        ▼
  [DNN 投影] 将平滑特征映射到低维嵌入空间
        │
        ▼
  [样本构建] 药物-靶点嵌入拼接 → 正样本(已知DTI) + 负样本(随机采样)
        │
        ▼
  [GBM 分类] 梯度提升树 10折交叉验证
        │
        ▼
  评估指标：AUC / AUPRC
```

---

## 3. 项目结构

```
iGRLDTI/
├── run_demo.py              # 主运行脚本（教学版，含详细注释）
├── data/
│   ├── Allnode_DrPr.csv          # 节点列表（药物549个 + 蛋白质424个）
│   ├── DrPrNum_DrPr.csv          # 正样本边列表（已知DTI，1923条）
│   ├── AllNodeAttribute_DrPr.csv # 节点特征矩阵（973 × 64）
│   ├── AllNegative_DrPr.csv      # 负样本候选池（230853条未知交互对）
│   └── Emdebding_GCN2_DrPr.csv   # 预计算好的节点嵌入（可选）
├── src/
│   ├── main.py    # 原始实验脚本
│   ├── model.py   # DNN 模型定义
│   ├── train.py   # NDLS 工具函数（aver / propagate / cal_hops）
│   └── utils.py   # 数据加载与图处理工具
└── results/
    └── roc_curve.png   # 运行后自动生成的 ROC 曲线图
```

---

## 4. 数据说明

| 文件 | 说明 | 规模 |
|---|---|---|
| `Allnode_DrPr.csv` | 节点索引与类型（0=药物, 1=蛋白质） | 973 节点 |
| `DrPrNum_DrPr.csv` | 正样本：已知药物-靶点相互作用边 | 1,923 条 |
| `AllNodeAttribute_DrPr.csv` | 节点特征向量（化学/生物属性） | 973 × 64 |
| `AllNegative_DrPr.csv` | 负样本候选：尚未报道的药物-靶点对 | 230,853 条 |

**节点编号约定：**
- 节点 0 ~ 548：药物（549 个）
- 节点 549 ~ 972：靶点蛋白质（424 个）

---

## 5. 快速开始

### 环境要求

```
Python  >= 3.6
PyTorch >= 1.4
scikit-learn
numpy
pandas
scipy
matplotlib   # 可选，用于 ROC 曲线可视化
```

### 安装依赖

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
```

### 运行演示

```bash
# 在项目根目录（iGRLDTI/）下运行
python run_demo.py
```

### 预期输出

```
============================================================
阶段1: 数据加载
============================================================
  节点总数    : 973
  正样本边数  : 1923
  特征维度    : 64
  药物节点数  : 549
  蛋白质节点数: 424

阶段2: 图预处理
...
阶段3: NDLS-F 特征平滑（计算每个节点最优传播跳数 h_v）
  节点收敛比例: XX.X%
...
阶段6: 10 折分层交叉验证
  Fold  1 | AUC = 0.XXXX | AUPRC = 0.XXXX
  ...
  Fold 10 | AUC = 0.XXXX | AUPRC = 0.XXXX

阶段7: 结果汇总
  Mean AUC   : 0.XXXX ± 0.XXXX
  Mean AUPRC : 0.XXXX ± 0.XXXX
  ROC 曲线已保存至: .../results/roc_curve.png
```

---

## 6. 核心概念解析

### 6.1 非线性扩散局部平滑（NDLS）

传统图神经网络对所有节点使用相同的传播跳数，容易导致 **过平滑**（over-smoothing）：
节点特征趋于一致，丧失局部结构信息。

NDLS 的解决思路：
- 逐跳传播特征 $X^{(k)} = \hat{A} \cdot X^{(k-1)}$
- 计算全局稳态 $X_\infty = \pi^\top X$（随机游走平稳分布加权）
- 为每个节点独立确定停止跳数：$h_v = \min\{k : \|X^{(k)}_v - X^\infty_v\|_2 < \varepsilon\}$

### 6.2 增广随机游走归一化

$$\hat{A} = \tilde{D}^{-1}\tilde{A}, \quad \tilde{A} = A + I$$

加入自环后行归一化，保证随机游走的马尔可夫性。

### 6.3 DTI 预测框架

```
药物节点嵌入 emb(d)  ──┐
                       ├─ 拼接 → GBM 分类器 → P(DTI 存在)
靶点节点嵌入 emb(t)  ──┘
```

---

## 7. 关键参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `k1` | 200 | NDLS-F 最大传播跳数 |
| `epsilon1` | 0.03 | NDLS-F 收敛阈值（越小越严格） |
| `hidden` | 64 | DNN 嵌入维度 |
| `dropout` | 0.5 | Dropout 比率 |
| `N_ESTIMATORS` | 499 | GBM 弱学习器数量 |
| `MAX_DEPTH` | 7 | GBM 每棵树最大深度 |
| `SUBSAMPLE` | 0.85 | GBM 样本采样比例 |

---

## 8. 参考文献

```
@article{iGRLDTI,
  title   = {iGRLDTI: An Improved Graph Representation Learning Method
             for Predicting Drug-Target Interactions over Heterogeneous
             Biological Information Network},
  journal = {Bioinformatics},
  year    = {2023}
}
```
