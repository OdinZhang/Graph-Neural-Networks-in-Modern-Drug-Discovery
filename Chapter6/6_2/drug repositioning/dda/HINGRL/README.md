# HINGRL 药物重定位教学案例

> **论文来源**：HINGRL: Predicting Drug-Disease Associations with Graph Representation Learning on Heterogeneous Information Networks
> **教学目标**：理解如何将异构信息网络（HIN）与图表示学习结合，用于预测药物-疾病关联

---

## 一、问题背景

**药物重定位（Drug Repositioning）**：为已知药物寻找新的治疗用途，比从零开发新药更快、成本更低。

传统方法依赖人工实验，而 HINGRL 通过以下思路实现计算预测：

```
多源生物数据 → 构建异构信息网络 → 图表示学习 → 分类器预测
```

---

## 二、核心算法思路

### 2.1 异构信息网络（HIN）结构

```
       药物 (Drug)
      /           \
 药物-蛋白        药物-疾病  ← 预测目标
    /                 \
蛋白质 (Protein) —— 疾病 (Disease)
         疾病-蛋白
```

三类节点：Drug、Disease、Protein
三类边（均取自公开数据库）：

| 边类型 | 文件 | 说明 |
|--------|------|------|
| 药物-疾病 | `DrDiNum.csv` | 已知关联（正样本） |
| 药物-蛋白 | `DrPrNum.csv` | 药物靶点互作 |
| 疾病-蛋白 | `DiPrNum.csv` | 疾病相关蛋白 |

### 2.2 特征表示

每个节点拥有两类特征，最终**拼接**为样本特征向量：

```
样本特征 = [attr(药物) | emb(药物) | attr(疾病) | emb(疾病)]
```

| 特征类型 | 来源 | 说明 |
|----------|------|------|
| 节点属性 (Attribute) | `AllNodeAttribute.csv` | 手工提取的生物属性向量 |
| DeepWalk 嵌入 (Embedding) | `AllEmbedding_DeepWalk.txt` | 在 HIN 上游走学到的图结构嵌入（64维） |

> **DeepWalk 直觉**：在图上做随机游走，结构相似的节点（邻居重叠多）会得到相近的嵌入向量，类似 Word2Vec 中语义相近的词。

### 2.3 训练流程

```
1. 构建 HIN
2. DeepWalk 生成节点嵌入
3. 随机采样负样本（不存在的药物-疾病对，数量 = 正样本）
4. 拼接特征向量
5. K 折交叉验证 + 随机森林分类
6. AUC-ROC 评估
```

---

## 三、数据集说明

| 数据集 | 编号 | 药物数 | 疾病数 | 正样本数 |
|--------|------|--------|--------|---------|
| B-Dataset | `-d 1` | ~269 | ~598 | ~18,416 |
| F-Dataset | `-d 2` | ~593 | ~313 | ~1,933 |

数据集路径：`data/B-Dataset/` 和 `data/F-Dataset/`

---

## 四、快速运行（教学演示）

### 4.1 环境要求

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

Python 版本：3.7+

### 4.2 运行教学版脚本

```bash
# 使用 B-Dataset，5 折交叉验证，100 棵决策树（快速演示）
python run_demo.py -d 1 -f 5 -n 100

# 使用 F-Dataset
python run_demo.py -d 2 -f 5 -n 100

# 查看所有参数
python run_demo.py --help
```

**参数说明：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-d` | 数据集（1=B-Dataset, 2=F-Dataset） | 1 |
| `-f` | K 折交叉验证的折数 | 5 |
| `-n` | 随机森林决策树棵数 | 100 |

### 4.3 预期输出

```
★★★★★★★★★★★ HINGRL 药物-疾病关联预测  教学演示 ★★★★★★★★★★★
  数据集: B-Dataset  |  折数: 5  |  决策树数: 100

[步骤 1] 加载异构交互网络数据...
  药物-疾病 正样本数   :  18416
  药物节点数           :    269
  节点属性维度         :    269
  DeepWalk 嵌入维度   :     64

[步骤 2] 生成负样本...
[步骤 3] 构建 5 折交叉验证数据集...
[步骤 4] 随机森林训练与评估...
  平均 AUC = 0.9xxx ± 0.00xx

[步骤 5] 绘制 ROC 曲线...
  ROC 图已保存至: data/tmp/roc_B-Dataset_5fold.png
```

---

## 五、代码结构（run_demo.py）

```
run_demo.py
├── partition()                  # 列表切块工具
├── generate_negative_samples()  # 随机负样本采样
├── load_data()                  # 加载 HIN 三类边 + 节点特征
├── build_feature_vector()       # 拼接属性向量 + DeepWalk 嵌入
├── prepare_cv_folds()           # 构建 K 折训练/测试集
├── run_cross_validation()       # 随机森林训练 + AUC 计算
├── plot_roc_curves()            # ROC 曲线可视化
└── main()                       # 主流程编排
```

---

## 六、关键设计说明

### 6.1 负样本采样策略

HINGRL 采用 **1:1 随机负采样**：从所有可能的（药物, 疾病）组合中，随机选取不存在已知关联的配对作为负样本。

```python
# 核心逻辑：不在正样本集合中 且 不重复
while len(negative_set) < target:
    d, dis = random.choice(all_drugs), random.choice(all_diseases)
    if (d, dis) not in positive_set and (d, dis) not in negative_set:
        negative_set.add((d, dis))
```

> **注意**：真实负样本（未被发现的关联）中可能含有未来会验证的正样本，这是药物重定位任务的固有挑战。

### 6.2 为何用随机森林

- 无需特征归一化，对混合特征（属性 + 嵌入）鲁棒
- 天然支持多线程（`n_jobs=-1`）
- `predict_proba` 输出概率，适合计算 AUC-ROC

### 6.3 AUC-ROC 评估

AUC（曲线下面积）衡量模型区分正负样本的能力：
- AUC = 1.0：完美预测
- AUC = 0.5：随机猜测（对角线）
- HINGRL 在 B-Dataset 上可达 AUC ≈ 0.95+

---

## 七、如何复现 DeepWalk 嵌入（可选）

若需重新生成 `AllEmbedding_DeepWalk.txt`，需安装 [OpenNE](https://github.com/thunlp/OpenNE/tree/pytorch)：

```bash
python -m openne \
  --method deepWalk \
  --input data/B-Dataset/AllDrDiIs_train.txt \
  --graph-format edgelist \
  --output data/B-Dataset/AllEmbedding_DeepWalk.txt \
  --representation-size 64
```

---

## 八、参考文献

- **HINGRL**: Zhao B W, et al. (2022). Predicting drug-disease associations with graph representation learning on heterogeneous information networks. *Briefings in Bioinformatics*, 23(4).
- **DeepWalk**: Perozzi B, et al. (2014). DeepWalk: Online learning of social representations. *KDD 2014*.
- **OpenNE**: [https://github.com/thunlp/OpenNE](https://github.com/thunlp/OpenNE)
