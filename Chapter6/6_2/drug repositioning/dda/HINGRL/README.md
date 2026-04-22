# HINGRL 案例说明

本目录对应药物重定位中的药物-疾病关联预测案例，核心方法为 **HINGRL**  
论文名称：*Predicting Drug-Disease Associations with Graph Representation Learning on Heterogeneous Information Networks*

该案例已经整理为两种使用方式：


1. `HINGRL_demo.ipynb`
   适合课堂教学、逐单元讲解和演示。

---

## 1. 案例目标

HINGRL 的任务是预测潜在的药物-疾病关联，核心思路是：

- 构建包含药物、疾病、蛋白质三类节点的异构网络
- 利用节点属性特征和 DeepWalk 图嵌入共同表示样本
- 将药物-疾病对转化为监督学习样本
- 通过交叉验证评估模型预测能力

在本书代码中，课堂主线主要使用 `B-Dataset`，`F-Dataset` 更适合作为课后练习。

---

## 2. 目录结构

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

主要文件说明：

- `HINGRL_demo.ipynb`
  课堂教学版 Notebook，当前主体展示 `B-Dataset`，最后提供 `F-Dataset` 课后作业。
- `data/B-Dataset/` 与 `data/F-Dataset/`
  两套药物-疾病异构网络数据。
- `demo_outputs/hingrl_b_dataset_metrics.png`
  Notebook 生成的课堂图示结果。

---

## 3. 数据说明

本案例的异构网络包括三类节点：

- Drug
- Disease
- Protein

对应三类关系：

- 药物-疾病关联：`DrDiNum.csv`
- 药物-蛋白关联：`DrPrNum.csv`
- 疾病-蛋白关联：`DiPrNum.csv`

节点特征由两部分组成：

- 属性特征：`AllNodeAttribute.csv`
- DeepWalk 嵌入：`AllEmbedding_DeepWalk.txt`

样本最终采用以下拼接方式：

```text
[attr(drug) | emb(drug) | attr(disease) | emb(disease)]
```

---

## 4. 运行方式

### 4.1 Notebook 运行

推荐直接打开：

- `HINGRL_demo.ipynb`

该 Notebook 已经整理成课堂版结构，适合教学展示：

1. 教学目标
2. 方法准备
3. 数据与任务认识
4. 样本构造与特征表示
5. 模型训练与结果评估
6. 可视化结果解读
7. 课堂总结
8. 课后作业

---

## 5. 环境依赖

建议至少安装以下依赖：

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

推荐在你当前用于 Jupyter 的虚拟环境中运行，以保证脚本和 Notebook 使用一致的解释器。

---

## 6. 教学建议

如果用于课堂讲授，建议采用下面的节奏：

1. 先解释异构网络中的三类节点与三类关系
2. 再说明为什么要同时使用属性特征和图嵌入
3. 然后讲负样本构造与交叉验证
4. 最后通过 `AUC` 和 `AUPR` 解释模型效果

可以把 `F-Dataset` 留给学生课后独立迁移和比较。

---

## 7. 参考文献

- Zhao B W, et al. *Predicting Drug-Disease Associations with Graph Representation Learning on Heterogeneous Information Networks*. Briefings in Bioinformatics, 2022.
- Perozzi B, et al. *DeepWalk: Online Learning of Social Representations*. KDD, 2014.
