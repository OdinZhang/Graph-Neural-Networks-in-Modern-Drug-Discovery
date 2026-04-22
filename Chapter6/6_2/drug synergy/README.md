# Drug Synergy 案例说明

本目录对应药物组合协同效应预测案例，采用异构图神经网络进行建模。

本案例已经整理为两种使用方式：

1. `drug_synergy_demo.ipynb`
   适合课堂教学、逐单元展示和实验讲解。

---

## 1. 案例目标

本案例关注的问题是：

**如何根据两个药物和一个细胞系的信息，预测该药物组合在该细胞系上的协同效应分数。**

与前两个案例不同，这里不是二分类任务，而是一个回归任务。

建模核心包括：

- 用异构图统一表示药物和细胞系
- 用 GraphSAGE 聚合图结构信息
- 用 `[drugA, drugB, cell]` 的联合表示预测协同分数

---

## 2. 目录结构

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

主要文件说明：

- `drug_synergy_demo.ipynb`
  课堂教学版 Notebook。
- `demo_synergy_labels.csv`
  药物组合在不同细胞系上的协同分数标签。
- `demo_drug_features.csv`
  药物分子特征。
- `demo_cell_features.csv`
  细胞系特征。
- `results/loss_curve.png`
  训练曲线图。
- `results/prediction_scatter.png`
  测试集预测散点图。

---

## 3. 数据与任务说明

本案例中的一个样本由三部分组成：

- 药物 A
- 药物 B
- 细胞系 C

输出目标是一个连续数值：

- `synergy_loewe`

因此这不是一个分类任务，而是协同分数预测回归任务。

课堂上可以重点提醒学生：

1. 药物与细胞系属于不同类型节点
2. 药物组合不是单节点问题，而是三元组问题
3. 评价指标应围绕误差和相关性，而不是 AUC

---

## 4. 方法流程

 `drug_synergy_demo.ipynb`：

1. 从 CSV 构建异构图
2. 将药物和细胞系特征映射到统一隐藏空间
3. 用异构 GraphSAGE 完成消息传递
4. 提取两个药物和一个细胞系的联合表示
5. 通过 MLP 预测协同分数
6. 用 `MSE`、`MAE`、`RMSE` 和相关性分析结果

---

## 5. 运行方式

### 5.1 Notebook 运行

推荐直接打开：

- `drug_synergy_demo.ipynb`

该 Notebook 已整理成课堂版结构，包括：

1. 教学目标
2. 方法准备
3. 数据与任务认识
4. 模型输入准备
5. 模型训练过程
6. 测试集结果与指标解释
7. 可视化结果解读
8. 课堂总结
9. 课后作业

---

## 6. 环境依赖

建议至少安装以下依赖：

```bash
pip install torch torch_geometric pandas matplotlib
```

如果使用的是你的 `jupyter` 虚拟环境，请确保该环境中已经安装：

- `torch`
- `torch_geometric`
- `pandas`
- `matplotlib`

---

## 7. 教学建议

课堂讲授时建议重点突出以下问题：

1. 为什么这个任务更适合异构图，而不是普通表格回归？
2. 为什么模型要同时读取两个药物和一个细胞系的表示？
3. 为什么这里只看损失还不够，还需要看 `MAE`、`RMSE` 和相关性？

如果作为课后扩展，可以让学生继续尝试：

- 增加 `R^2` 指标
- 比较不同隐藏维度
- 扩展图结构，例如引入药物-药物相似性边

---

## 8. 备注

由于不同版本的 PyTorch 在加载旧缓存图对象时可能存在兼容性差异，如果脚本在读取 `macro_hetero_graph.pt` 时报错，可以：

1. 删除旧缓存图文件后重新构建
2. 或在兼容版本的环境中重新运行

Notebook 版本已经按课堂演示用途做了更稳定的整理。
