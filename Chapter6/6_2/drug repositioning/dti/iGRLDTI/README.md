# iGRLDTI 案例说明

本目录对应药物重定位中的药物-靶点相互作用预测案例，核心方法为 **iGRLDTI**  
论文名称：*An Improved Graph Representation Learning Method for Predicting Drug-Target Interactions over Heterogeneous Biological Information Network*

该案例已经整理为两种使用方式：

1. `iGRLDTI_demo.ipynb`
   适合课堂教学、逐步展示和板书讲解。

---

## 1. 案例目标

iGRLDTI 的核心任务是预测潜在的药物-靶点相互作用（DTI）。

在这个案例中，主要包含三层思路：

- 把药物和靶点构造成异质生物信息网络
- 用 NDLS-F 缓解图传播中的过平滑问题
- 用节点嵌入构造药物-靶点样本，并通过分类器完成预测

---

## 2. 目录结构

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

主要文件说明：

- `iGRLDTI_demo.ipynb`
  课堂教学版 Notebook，适合逐节讲解。
- `src/utils.py`
  图预处理和数据加载工具。
- `src/train.py`
  NDLS-F 与相关训练辅助函数。
- `src/model.py`
  DNN 投影模型定义。
- `results/roc_curve.png`
  运行后生成的 ROC 曲线图。

---

## 3. 数据说明

图中包含两类节点：

- 药物节点
- 蛋白质节点

主要数据文件：

- `Allnode_DrPr.csv`
  节点列表
- `DrPrNum_DrPr.csv`
  已知药物-靶点相互作用边
- `AllNodeAttribute_DrPr.csv`
  节点属性特征
- `AllNegative_DrPr.csv`
  候选负样本池

课堂上可以重点强调：

1. 正样本来自已知 DTI 边
2. 负样本来自未知配对中的随机采样
3. 最终样本特征由药物嵌入和靶点嵌入拼接得到

---

## 4. 方法流程

`iGRLDTI_demo.ipynb`：

1. 数据加载
2. 图预处理
3. NDLS-F 特征平滑
4. DNN 节点嵌入提取
5. 正负样本构造
6. 10 折分层交叉验证
7. AUC / AUPRC 评估与 ROC 可视化

其中最有教学价值的部分通常是：

- NDLS-F 为什么能缓解过平滑
- 样本是怎样从节点嵌入转换成药物-靶点对的
- 为什么要同时看 `AUC` 和 `AUPRC`

---

## 5. 运行方式

### 5.1 Notebook 运行

推荐直接打开：

- `iGRLDTI_demo.ipynb`

该 Notebook 已整理成课堂版结构，包括：

1. 教学目标
2. 方法准备
3. 数据与任务认识
4. NDLS-F 与节点嵌入
5. 样本构造与交叉验证
6. 模型结果与指标解释
7. 可视化结果解读
8. 课堂总结
9. 课后作业

---

## 6. 环境依赖

建议至少安装以下依赖：

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
```

如果你希望脚本运行结果与 Notebook 保持一致，建议始终在同一个 Jupyter 虚拟环境中执行。

---

## 7. 教学建议

课堂讲授时建议把重点放在以下三个问题上：

1. 为什么图传播会导致过平滑？
2. NDLS-F 中“每个节点不同跳数”的设计意义是什么？
3. 为什么在 DTI 预测中 `AUPRC` 往往和 `AUC` 一样重要？

如果作为作业扩展，可以进一步让学生：

- 修改 `k1`、`epsilon1`
- 比较 5 折与 10 折结果
- 增加结果导出功能

---

## 8. 参考文献

- *iGRLDTI: An Improved Graph Representation Learning Method for Predicting Drug-Target Interactions over Heterogeneous Biological Information Network*.
