# coding: utf-8
"""
HINGRL 教学演示版 (train_demo.py)
=====================================
HINGRL: Heterogeneous Interaction Network-based Graph Representation Learning
        用于药物-疾病关联预测

算法核心思路：
  1. 构建异构交互网络 (HIN)：药物-疾病、药物-蛋白、疾病-蛋白
  2. 利用 DeepWalk 学习节点嵌入（图表示）+ 节点属性特征
  3. 拼接两类特征作为样本表示
  4. 用随机森林分类器进行 K 折交叉验证
  5. 用 AUC-ROC 评估预测性能

运行示例：
  python train_demo.py -d 1 -f 5 -n 100
"""

import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

# ─── 路径配置（相对于项目根目录）─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR  = os.path.join(DATA_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 1：工具函数
# ══════════════════════════════════════════════════════════════════════════════

def partition(ls, size):
    """将列表 ls 均匀切分为若干子列表，每块大小为 size。"""
    return [ls[i:i + size] for i in range(0, len(ls), size)]


def generate_negative_samples(positive_pairs, all_drugs, all_diseases):
    """
    随机负样本生成：
    - 数量与正样本相同（1:1 平衡）
    - 保证生成的 (药物, 疾病) 对不出现在正样本集合中
    - 保证负样本之间也不重复

    Parameters
    ----------
    positive_pairs : list of [drug_id, disease_id]
    all_drugs      : list，所有药物 ID
    all_diseases   : list，所有疾病 ID

    Returns
    -------
    negative_pairs : list of [drug_id, disease_id]
    """
    print("  [负样本生成] 正样本数量:", len(positive_pairs))
    positive_set = set(map(tuple, positive_pairs))  # 用 set 加速查重
    negative_set = set()

    target = len(positive_pairs)
    while len(negative_set) < target:
        d = random.choice(all_drugs)
        dis = random.choice(all_diseases)
        pair = (d, dis)
        if pair not in positive_set and pair not in negative_set:
            negative_set.add(pair)

    negative_pairs = [list(p) for p in negative_set]
    print("  [负样本生成] 负样本数量:", len(negative_pairs))
    return negative_pairs


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 2：数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_name):
    """
    加载异构交互网络的三类边以及节点特征/嵌入。

    HIN 包含三类节点：Drug（药物）、Disease（疾病）、Protein（蛋白质）
    三类边：
      DrDi  ── 药物-疾病关联（预测目标）
      DrPr  ── 药物-蛋白互作
      DiPr  ── 疾病-蛋白关联

    节点特征来源：
      Attribute  ── 手工提取的节点属性向量
      Embedding  ── DeepWalk 在 HIN 上学到的节点嵌入
    """
    dset = os.path.join(DATA_DIR, dataset_name)
    print(f"\n{'='*60}")
    print(f"  数据集：{dataset_name}")
    print(f"{'='*60}")

    dr_di = pd.read_csv(os.path.join(dset, "DrDiNum.csv"),    header=None)
    dr_pr = pd.read_csv(os.path.join(dset, "DrPrNum.csv"),    header=None)
    di_pr = pd.read_csv(os.path.join(dset, "DiPrNum.csv"),    header=None)

    print(f"  药物-疾病 正样本数   : {len(dr_di):>6}")
    print(f"  药物-蛋白 关联数     : {len(dr_pr):>6}")
    print(f"  疾病-蛋白 关联数     : {len(di_pr):>6}")

    drug_names    = pd.read_csv(os.path.join(dset, "drugName.csv"),    header=None, names=["id", "name"])
    disease_names = pd.read_csv(os.path.join(dset, "diseaseName.csv"), header=None, names=["id", "name"])
    print(f"  药物节点数           : {len(drug_names):>6}")
    print(f"  疾病节点数           : {len(disease_names):>6}")

    # 节点属性矩阵（去掉索引列后剩余的列即为特征）
    attribute = pd.read_csv(os.path.join(dset, "AllNodeAttribute.csv"), header=None, index_col=0)
    attribute = attribute.iloc[:, 1:]

    # DeepWalk 嵌入（首行是 "节点数 维度"，跳过；按节点 ID 排序后设为索引）
    embedding = pd.read_csv(
        os.path.join(dset, "AllEmbedding_DeepWalk.txt"),
        sep=" ", header=None, skiprows=1
    )
    embedding = embedding.sort_values(0, ascending=True)
    embedding.set_index([0], inplace=True)

    print(f"  节点属性维度         : {attribute.shape[1]:>6}")
    print(f"  DeepWalk 嵌入维度   : {embedding.shape[1]:>6}")

    return dr_di, dr_pr, di_pr, drug_names, disease_names, attribute, embedding


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 3：K 折数据集准备
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(pair_ids_col0, pair_ids_col1, attribute, embedding):
    """
    为一批 (实体A, 实体B) 对构建特征向量。

    拼接方式：[attr(A) | emb(A) | attr(B) | emb(B)]
    这样分类器可以同时看到两个节点的属性和图结构信息。
    """
    feat_a = pd.concat(
        [attribute.loc[pair_ids_col0], embedding.loc[pair_ids_col0]],
        axis=1
    ).reset_index(drop=True)
    feat_b = pd.concat(
        [attribute.loc[pair_ids_col1], embedding.loc[pair_ids_col1]],
        axis=1
    ).reset_index(drop=True)
    return pd.concat([feat_a, feat_b], axis=1).values


def prepare_cv_folds(dr_di, negative_df, attribute, embedding, fold_num):
    """
    将正样本随机打乱后切成 fold_num 份，逐折构建训练/测试集。

    正负样本同步按相同折号划分，保证每折训练/测试比例一致（1:1）。

    Returns
    -------
    folds : list of dict，每个 dict 含 X_train, X_test, y_train, y_test
    """
    n = len(dr_di)
    # 打乱正样本索引
    shuffled_idx = random.sample(range(n), n)
    fold_size = math.ceil(n / fold_num)
    fold_indices = partition(shuffled_idx, fold_size)

    # 用 NaN 填充使所有折等长（最后一折可能短一点）
    max_len = max(len(f) for f in fold_indices)
    fold_matrix = np.full((fold_num, max_len), -1, dtype=int)
    for fi, f in enumerate(fold_indices):
        fold_matrix[fi, :len(f)] = f

    neg_array = negative_df.values
    dr_di_array = dr_di.values

    folds = []
    for i in range(fold_num):
        test_idx  = fold_matrix[i]
        test_idx  = test_idx[test_idx >= 0]           # 去掉填充的 -1
        train_idx = np.concatenate([fold_matrix[j][fold_matrix[j] >= 0]
                                    for j in range(fold_num) if j != i])

        # 正样本
        pos_train = dr_di_array[train_idx]
        pos_test  = dr_di_array[test_idx]

        # 对应的负样本（使用相同折号划分，保持 1:1）
        neg_train = neg_array[train_idx]
        neg_test  = neg_array[test_idx]

        # 合并并打标签
        train_pairs = np.vstack([pos_train, neg_train])
        test_pairs  = np.vstack([pos_test,  neg_test])
        y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))
        y_test  = np.array([1] * len(pos_test)  + [0] * len(neg_test))

        X_train = build_feature_vector(
            train_pairs[:, 0].tolist(), train_pairs[:, 1].tolist(),
            attribute, embedding
        )
        X_test = build_feature_vector(
            test_pairs[:, 0].tolist(), test_pairs[:, 1].tolist(),
            attribute, embedding
        )

        folds.append(dict(X_train=X_train, X_test=X_test,
                          y_train=y_train, y_test=y_test))
        print(f"  折 {i+1:>2}/{fold_num}  训练样本: {len(y_train):>6}  测试样本: {len(y_test):>5}")

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 4：模型训练与评估
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(folds, n_estimators):
    """
    对每折数据训练随机森林，计算 AUC，收集 TPR 用于绘制平均 ROC 曲线。

    随机森林分类器说明：
      - n_estimators : 决策树棵数（越多越稳定，但速度越慢）
      - predict_proba: 输出每个样本属于正类的概率，用于计算 ROC

    Returns
    -------
    aucs    : 每折 AUC 列表
    tprs    : 每折在公共 FPR 轴上插值后的 TPR 列表
    mean_fpr: 公共 FPR 轴（0~1 均匀 1000 点）
    """
    mean_fpr = np.linspace(0, 1, 1000)
    tprs, aucs = [], []

    print(f"\n{'='*60}")
    print(f"  随机森林交叉验证（n_estimators={n_estimators}）")
    print(f"{'='*60}")

    for i, fold in enumerate(folds):
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        clf.fit(fold["X_train"], fold["y_train"])

        proba = clf.predict_proba(fold["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(fold["y_test"], proba)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        # 插值到公共 FPR 轴，便于后续计算均值
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        print(f"  折 {i+1:>2}  AUC = {fold_auc:.4f}")

    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))
    print(f"\n  平均 AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return aucs, tprs, mean_fpr


# ══════════════════════════════════════════════════════════════════════════════
# 步骤 5：结果可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(aucs, tprs, mean_fpr, dataset_name, fold_num, save_path=None):
    """
    绘制每折 ROC 曲线（细线）+ 均值 ROC 曲线（粗线）+ 置信区间（阴影）。

    ROC 曲线说明：
      X 轴 FPR（假正率）= FP / (FP + TN)，越低越好
      Y 轴 TPR（真正率）= TP / (TP + FN)，越高越好
      AUC 越接近 1 表示模型越好；随机猜测 AUC = 0.5（对角线）
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # 每折细线
    for i, (tpr_i, auc_i) in enumerate(zip(tprs, aucs)):
        ax.plot(mean_fpr, tpr_i, lw=1, alpha=0.4,
                label=f"折 {i+1} (AUC={auc_i:.3f})")

    # 均值曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc  = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="navy", lw=2.5,
            label=f"均值 ROC (AUC={mean_auc:.4f} ± {std_auc:.4f})")

    # 置信区间阴影
    std_tpr  = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper,
                    color="navy", alpha=0.15, label="± 1 标准差")

    # 随机基线
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, label="随机猜测 (AUC=0.5)")

    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="假正率 (FPR)", ylabel="真正率 (TPR)",
           title=f"HINGRL {fold_num} 折交叉验证 ROC 曲线\n数据集: {dataset_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  ROC 图已保存至: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main(options):
    dataset_name = "B-Dataset" if options.dataset == 1 else "F-Dataset"
    fold_num     = options.fold_num
    n_estimators = options.tree_number

    print("\n" + "★" * 60)
    print("  HINGRL 药物-疾病关联预测  教学演示")
    print("★" * 60)
    print(f"  数据集: {dataset_name}  |  折数: {fold_num}  |  决策树数: {n_estimators}")

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    print("\n[步骤 1] 加载异构交互网络数据...")
    dr_di, dr_pr, di_pr, drug_names, disease_names, attribute, embedding = load_data(dataset_name)

    # ── 2. 负样本生成 ────────────────────────────────────────────────────────
    print("\n[步骤 2] 生成负样本（随机采样不存在的药物-疾病对）...")
    negative_pairs = generate_negative_samples(
        dr_di.values.tolist(),
        drug_names["id"].tolist(),
        disease_names["id"].tolist()
    )
    negative_df = pd.DataFrame(negative_pairs)

    # ── 3. K 折数据准备 ──────────────────────────────────────────────────────
    print(f"\n[步骤 3] 构建 {fold_num} 折交叉验证数据集...")
    print("  特征拼接方式: [节点属性(A) | DeepWalk嵌入(A) | 节点属性(B) | DeepWalk嵌入(B)]")
    folds = prepare_cv_folds(dr_di, negative_df, attribute, embedding, fold_num)

    # ── 4. 训练与评估 ────────────────────────────────────────────────────────
    print(f"\n[步骤 4] 随机森林训练与评估...")
    aucs, tprs, mean_fpr = run_cross_validation(folds, n_estimators)

    # ── 5. 可视化 ────────────────────────────────────────────────────────────
    print("\n[步骤 5] 绘制 ROC 曲线...")
    save_path = os.path.join(TMP_DIR, f"roc_{dataset_name}_{fold_num}fold.png")
    plot_roc_curves(aucs, tprs, mean_fpr, dataset_name, fold_num, save_path=save_path)

    print("\n" + "★" * 60)
    print("  演示完成！")
    print("★" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import optparse
    import sys

    parser = optparse.OptionParser(description=__doc__)
    parser.add_option("-d", "--dataset", dest="dataset", default=1, type="int",
                      help="数据集选择 (1: B-Dataset; 2: F-Dataset)  [默认: 1]")
    parser.add_option("-f", "--fold-num", dest="fold_num", default=5, type="int",
                      help="K 折交叉验证的折数  [默认: 5]")
    parser.add_option("-n", "--tree-number", dest="tree_number", default=100, type="int",
                      help="随机森林决策树棵数  [默认: 100]")

    options, _ = parser.parse_args()
    sys.exit(main(options))
