"""
run_demo.py — iGRLDTI 药物-靶点相互作用预测演示脚本
=======================================================
基于图表示学习的药物重定位方法（iGRLDTI）教学版本

运行方式（在项目根目录下执行）：
    python run_demo.py

算法流程：
  阶段1  数据加载        — 读取节点、边、特征 CSV 文件
  阶段2  图预处理        — 构建归一化邻接矩阵（增广随机游走）
  阶段3  特征平滑(NDLS)  — 非线性扩散局部平滑(NDLS-F)计算每节点最优跳数
  阶段4  节点嵌入        — DNN 投影得到低维表示
  阶段5  样本构建        — 正负样本配对（药物-靶点对）
  阶段6  交叉验证        — 梯度提升树 10 折分层 CV
  阶段7  结果汇总        — AUC / AUPRC 统计 & ROC 曲线可视化

参考论文：
  iGRLDTI: An Improved Graph Representation Learning Method for
  Predicting Drug-Target Interactions over Heterogeneous Biological
  Information Network
"""

# ============================================================
# 0. 依赖导入
# ============================================================
import os
import sys

# 将 src/ 目录加入模块搜索路径，使 utils / model / train 可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score

# 工具函数与模型（位于 src/ 目录）
from utils import load_data, sparse_mx_to_torch_sparse_tensor, aug_random_walk
from model import DNN
from train import aver

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

pd.set_option('display.max_rows', 10)

# 数据目录（相对于项目根目录）
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


# ============================================================
# 阶段1: 数据加载
# ============================================================
print("=" * 60)
print("阶段1: 数据加载")
print("=" * 60)

# 节点列表：药物节点（0~548）+ 蛋白质节点（549~972），共 973 个节点
AllNode = pd.read_csv(
    os.path.join(DATA_DIR, 'Allnode_DrPr.csv'),
    names=[0, 1], skiprows=1
)

# 边列表：已知药物-靶点相互作用（正样本），格式：[药物节点ID, 靶点节点ID]
Alledge = pd.read_csv(
    os.path.join(DATA_DIR, 'DrPrNum_DrPr.csv'),
    header=None
)

# 节点特征矩阵（第0列为节点索引，去掉后剩64维特征）
features = pd.read_csv(
    os.path.join(DATA_DIR, 'AllNodeAttribute_DrPr.csv'),
    header=None
)
features = features.iloc[:, 1:]   # 去掉第0列索引列，保留 64 维特征

print(f"  节点总数    : {len(AllNode)}")
print(f"  正样本边数  : {len(Alledge)}")
print(f"  特征维度    : {features.shape[1]}")

# 构造节点标签：
#   前 549 个节点 → 药物  (label = 0)
#   其余节点      → 蛋白质 (label = 1)
labels = pd.DataFrame(np.zeros((len(AllNode), 1)))
labels.iloc[549:] = 1
labels = labels[0]

print(f"  药物节点数  : {int((labels == 0).sum())}")
print(f"  蛋白质节点数: {int((labels == 1).sum())}")


# ============================================================
# 阶段2: 图预处理 — 构建归一化邻接矩阵
# ============================================================
print("\n" + "=" * 60)
print("阶段2: 图预处理")
print("=" * 60)

# load_data：将边列表转成 scipy 稀疏邻接矩阵，并返回训练/验证/测试划分
adj, features, labels, idx_train, idx_val, idx_test = load_data(
    Alledge, features, labels
)

node_sum = adj.shape[0]          # 节点总数 N = 973
edge_sum = adj.sum() / 2         # 无向边总数 M
row_sum  = (adj.sum(1) + 1)      # 每节点度数 + 1（含自环）

# 稳态分布 π_v = (d_v + 1) / (2M + N)
# 含义：随机游走收敛后节点 v 被访问的概率，用于衡量全局平滑极限
norm_a_inf = row_sum / (2 * edge_sum + node_sum)

# 增广随机游走归一化邻接矩阵：Â = D̃^{-1} Ã（含自环后行归一化）
adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

print(f"  节点数 N    : {node_sum}")
print(f"  边数   M    : {int(edge_sum)}")
print(f"  邻接矩阵已归一化完成（增广随机游走）")


# ============================================================
# 超参数配置
# ============================================================
class Args:
    k1       = 200    # NDLS-F 最大传播跳数（特征平滑最多传播 k1 跳）
    epsilon1 = 0.03   # NDLS-F 收敛阈值 ε₁（节点特征变化小于此值视为收敛）
    hidden   = 64     # DNN 隐藏层 / 嵌入维度
    dropout  = 0.5    # Dropout 比率

args = Args()


# ============================================================
# 阶段3: 特征平滑 — NDLS-F（非线性扩散局部平滑）
# ============================================================
print("\n" + "=" * 60)
print("阶段3: NDLS-F 特征平滑（计算每个节点最优传播跳数 h_v）")
print("=" * 60)
print("  原理：不同节点的最优平滑跳数不同，")
print("        过平滑的节点提前停止传播，保留局部结构信息。")

# Step 3-1：L1 归一化节点特征
features = F.normalize(features, p=1)

# Step 3-2：逐跳传播，记录每跳的特征列表
#   feature_list[k] = Â^k · X  （X 经过 k 跳扩散后的特征）
feature_list = [features]
for i in range(1, args.k1):
    feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

# Step 3-3：计算全局稳态特征 X_∞ = π^T · X
#   当传播跳数 → ∞ 时，特征趋于的极限状态
norm_a_inf_tensor = torch.Tensor(norm_a_inf).view(-1, node_sum)
norm_fea_inf      = torch.mm(norm_a_inf_tensor, features)

# Step 3-4：为每个节点确定最优跳数 h_v
#   h_v = 最小的 k，使得 ||X^(k)_v - X_∞_v||₂ < ε₁
#   即：节点特征首次"足够接近"全局稳态时停止传播
hops        = torch.zeros(node_sum)
mask_before = torch.zeros(node_sum, dtype=torch.bool)

for i in range(args.k1):
    dist = (feature_list[i] - norm_fea_inf).norm(2, dim=1)
    mask = (dist < args.epsilon1) & (~mask_before)   # 本跳新收敛的节点
    mask_before[mask] = True
    hops[mask] = i

# 未收敛的节点：使用最大跳数 k1-1（充分平滑）
hops[~mask_before] = args.k1 - 1

converged_ratio = mask_before.float().mean().item() * 100
print(f"  最大传播跳数 k1   : {args.k1}")
print(f"  收敛阈值 epsilon1 : {args.epsilon1}")
print(f"  节点收敛比例      : {converged_ratio:.1f}%")
print("  NDLS-F 跳数计算完成")

# Step 3-5：对每个节点，将 0~h_v 跳的特征加权平均（带阻尼 α 回原始特征）
#   f_v = mean_{j=0}^{h_v-1} [(1-α)·X^(j)_v + α·X^(0)_v]
input_feature = aver(hops, adj, feature_list)
print("  NDLS-F 特征聚合完成")


# ============================================================
# 阶段4: 节点嵌入 — DNN 特征投影
# ============================================================
print("\n" + "=" * 60)
print("阶段4: DNN 节点嵌入提取")
print("=" * 60)
print("  说明：DNN 在此以推断模式使用（未经监督训练），")
print("        作用是将平滑后的 64 维特征投影到低维嵌入空间。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  运行设备    : {device}")

input_feature = input_feature.to(device)

# DNN 结构：输入维(64) → 隐藏层(64) → 嵌入层(64)
model = DNN(features.shape[1], args.hidden, args.hidden, args.dropout).to(device)
model.eval()

with torch.no_grad():
    _, embedding = model(input_feature)

# 转为 DataFrame，行索引即节点编号（0~972）
Embedding_GCN = pd.DataFrame(embedding.cpu().numpy())

print(f"  输入特征维度: {features.shape[1]}")
print(f"  输出嵌入维度: {args.hidden}")
print(f"  节点嵌入矩阵: {Embedding_GCN.shape}  (节点数 × 嵌入维)")


# ============================================================
# 阶段5: 样本构建 — 正负样本配对
# ============================================================
print("\n" + "=" * 60)
print("阶段5: 构建正负样本对")
print("=" * 60)
print("  策略：正负样本 1:1 采样，特征 = 拼接药物嵌入 + 靶点嵌入")

# 正样本：已知 DTI 边（药物节点ID, 靶点节点ID）→ label=1
Positive = Alledge.copy()
Positive[2] = 1

# 负样本：从未知交互对中随机采样，数量与正样本相同 → label=0
AllNegative = pd.read_csv(
    os.path.join(DATA_DIR, 'AllNegative_DrPr.csv'),
    header=None
)
Negative = AllNegative.sample(n=len(Positive), random_state=SEED)
Negative[2] = 0

# 合并正负样本
result = pd.concat([Positive, Negative]).reset_index(drop=True)

# 构造样本特征：[drug_emb || protein_emb]，维度 = 2 × hidden
drug_emb    = Embedding_GCN.loc[result[0].values].reset_index(drop=True)
protein_emb = Embedding_GCN.loc[result[1].values].reset_index(drop=True)
X = pd.concat([drug_emb, protein_emb], axis=1)
Y = result[2]

print(f"  正样本数    : {int((Y == 1).sum())}")
print(f"  负样本数    : {int((Y == 0).sum())}")
print(f"  样本特征维度: {X.shape[1]}  (= 2 × {args.hidden})")


# ============================================================
# 阶段6: 10 折交叉验证 — 梯度提升树（GBM）分类
# ============================================================
print("\n" + "=" * 60)
print("阶段6: 10 折分层交叉验证（GradientBoostingClassifier）")
print("=" * 60)

# GBM 超参数（与原论文一致）
N_ESTIMATORS = 499    # 弱学习器（决策树）数量
MAX_DEPTH     = 7     # 每棵树最大深度
SUBSAMPLE     = 0.85  # 样本采样比例（随机子采样，防止过拟合）
K_FOLD        = 10    # 折数

X_arr = np.array(X)
Y_arr = np.array(Y)

mean_fpr = np.linspace(0, 1, 1000)   # 公共 FPR 轴（用于绘制平均 ROC）
tprs     = []
aucs     = []
auprcs   = []

skf = StratifiedKFold(n_splits=K_FOLD, random_state=0, shuffle=True)
print(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, subsample={SUBSAMPLE}")
print()

for fold, (train_idx, test_idx) in enumerate(skf.split(X_arr, Y_arr)):
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]

    clf = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=SEED
    )
    clf.fit(X_train, Y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]   # 预测为正类（DTI存在）的概率

    # --- AUC（ROC 曲线下面积）---
    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    fold_auc = auc(fpr, tpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(fold_auc)

    # --- AUPRC（精确率-召回率曲线下面积，不平衡场景更稳健）---
    fold_auprc = average_precision_score(Y_test, y_prob)
    auprcs.append(fold_auprc)

    print(f"  Fold {fold + 1:2d} | AUC = {fold_auc:.4f} | AUPRC = {fold_auprc:.4f}")

# 计算均值 ROC 及统计量
mean_tpr      = np.mean(tprs, axis=0)
mean_tpr[-1]  = 1.0
mean_auc      = auc(mean_fpr, mean_tpr)
std_auc       = np.std(aucs)
mean_auprc    = np.mean(auprcs)
std_auprc     = np.std(auprcs)


# ============================================================
# 阶段7: 结果汇总 & ROC 曲线可视化
# ============================================================
print("\n" + "=" * 60)
print("阶段7: 结果汇总")
print("=" * 60)
print(f"  Mean AUC   : {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  Mean AUPRC : {mean_auprc:.4f} ± {std_auprc:.4f}")

# 绘制 ROC 曲线并保存（若无 matplotlib 则跳过）
try:
    import matplotlib
    matplotlib.use('Agg')   # 非交互后端，直接保存为文件
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    # 各折 ROC 曲线（半透明，便于观察折间差异）
    for i, tpr_fold in enumerate(tprs):
        ax.plot(mean_fpr, tpr_fold, alpha=0.2, lw=1)

    # 平均 ROC 曲线（加粗）
    ax.plot(
        mean_fpr, mean_tpr, color='steelblue', lw=2.5,
        label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})'
    )

    # 随机猜测基线
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        'iGRLDTI — 10-Fold CV ROC Curve\n(Drug-Target Interaction Prediction)',
        fontsize=13
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  ROC 曲线已保存至: {out_path}")

except Exception as e:
    print(f"\n  (可视化跳过: {e})")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
