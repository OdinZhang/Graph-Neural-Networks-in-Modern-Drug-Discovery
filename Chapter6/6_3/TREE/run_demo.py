# -*- coding: utf-8 -*-
"""
TREE 框架 — 课堂教学演示脚本
==============================
本脚本以 CPDB（蛋白质互作网络）为例，完整展示 TREE 的训练流程：
  Step 1  读取 HDF5 网络数据
  Step 2  随机游走子图采样 + 最短路径矩阵
  Step 3  K 折交叉验证划分
  Step 4  模型训练（Graphormer + 注意力融合）
  Step 5  评估并打印结果

运行方式：
  python run_demo.py
"""

import os
import sys
import numpy as np
import h5py
import networkx as nx
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.getcwd())
from main import train
from config import (
    H5_FILE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR,
    SUBGRAPHA_TEMPLATE, SPATIAL_TEMPLATE, ADJ_TEMPLATE, FEATURE_TEMPLATE
)
from utils import Node2vec, pickle_dump, format_filename, write_log

# ──────────────────────────────────────────────
#  演示参数（可根据课堂需要调整）
# ──────────────────────────────────────────────
DATASET     = 'CPDB'   # 演示数据集：CPDB 同构 PPI 网络
N_GRAPHS    = 6        # 每个节点的子图（随机游走通道）数量
N_NEIGHBORS = 8        # 每个子图的邻居节点数（路径长度）
N_LAYERS    = 3        # Graphormer 编码层数
LR          = 0.001    # 学习率
CV_FOLDS    = 3        # K 折交叉验证折数（论文用 10，课堂用 3 即可）
N_EPOCH     = 10       # 训练轮次（论文用 100，课堂演示用 10）
DFF         = 128      # FFN 中间维度
BATCH_SIZE  = 32       # 批次大小
DROPOUT     = 0.5      # Dropout 比率
LOSS_MUL    = 0.27     # 正样本权重系数（CPDB 中正样本约占 27%）


# ──────────────────────────────────────────────
#  Step 1：读取网络数据
# ──────────────────────────────────────────────

def read_h5file(path):
    """
    从 HDF5 文件读取网络数据。

    文件结构：
      network  — 邻接矩阵，shape (N, N)
      features — 多组学特征矩阵，shape (N, d)
      y_train / y_val / y_test  — 标签向量（1=癌症基因）
      mask_train / mask_val / mask_test — 节点掩码
    """
    with h5py.File(path, 'r') as f:
        network    = f['network'][:]
        features   = f['features'][:]
        y_train    = f['y_train'][:]
        y_test     = f['y_test'][:]
        y_val      = f['y_val'][:] if 'y_val' in f else None
        train_mask = f['mask_train'][:]
        test_mask  = f['mask_test'][:]
        val_mask   = f['mask_val'][:] if 'mask_val' in f else None

    print(f"  节点数 N = {network.shape[0]}，特征维度 d = {features.shape[1]}")
    print(f"  训练集正样本（癌症基因）= {int(y_train.sum())}，"
          f"测试集正样本 = {int(y_test.sum())}")
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


# ──────────────────────────────────────────────
#  Step 2：随机游走子图采样 + 最短路径矩阵
# ──────────────────────────────────────────────

def build_subgraphs(adj):
    """
    为每个节点生成 N_GRAPHS 个子图，每个子图含 N_NEIGHBORS 个邻居节点。

    流程：
      1. Node2Vec 随机游走 → 每节点 N_GRAPHS 条路径，每条长 N_NEIGHBORS
      2. 计算全图最短路径矩阵 → 作为 Graphormer 空间编码的输入

    返回：
      walks          shape (N, N_GRAPHS, N_NEIGHBORS)   游走路径
      subgraphs_list shape (N, N_GRAPHS, 1, N_NEIGHBORS, N_NEIGHBORS)  空间偏置
    """
    n_nodes = adj.shape[0]
    pmat    = np.ones((n_nodes, n_nodes), dtype=int) * np.inf

    # 构建 NetworkX 图
    r, c = np.where(adj > 0)
    G = nx.Graph()
    G.add_edges_from(zip(r.tolist(), c.tolist()))
    print(f"  构建图：{n_nodes} 个节点，{len(r)} 条边")

    # 随机游走
    print(f"  Node2Vec 随机游走（n_graphs={N_GRAPHS}, n_neighbors={N_NEIGHBORS}）...")
    walks = Node2vec(graph=G, path_length=N_NEIGHBORS,
                     num_paths=N_GRAPHS, workers=6, dw=True).get_walks()

    # 对齐节点顺序（异构网络兼容）
    new_walks = np.zeros((n_nodes, N_GRAPHS, N_NEIGHBORS), dtype=int)
    for i in range(walks.shape[0]):
        new_walks[walks[i][0][0], :, :] = walks[i]
    walks = new_walks

    # 最短路径矩阵（有缓存则加载，否则计算并保存）
    sp_path = f"sp/{DATASET}_sp.h5"
    os.makedirs("sp", exist_ok=True)
    if os.path.exists(sp_path):
        print(f"  加载缓存最短路径矩阵：{sp_path}")
        with h5py.File(sp_path, 'r') as f:
            pmat = f["sp"][:]
        pmat[pmat == np.inf] = -1
    else:
        print("  计算全图最短路径矩阵（首次运行稍慢）...")
        for node_i, nbrs in nx.all_pairs_shortest_path_length(G):
            if node_i % 500 == 0:
                print(f"    进度：{node_i}/{n_nodes}")
            for node_j, d in nbrs.items():
                pmat[node_i, node_j] = d
        pmat[pmat == np.inf] = -1
        with h5py.File(sp_path, 'w') as f:
            f.create_dataset("sp", data=pmat)
        print(f"  最短路径矩阵已保存：{sp_path}")

    # 为每个节点、每个子图构建空间偏置矩阵
    subgraphs_list = []
    for nid in range(n_nodes):
        sub = []
        for g in range(N_GRAPHS):
            idx = walks[nid, g, :].astype(int)
            sub.append(np.expand_dims(pmat[idx, :][:, idx], 0))
        subgraphs_list.append(sub)

    return walks, np.array(subgraphs_list)


# ──────────────────────────────────────────────
#  Step 3：K 折交叉验证划分
# ──────────────────────────────────────────────

def make_cv_splits(y, mask):
    """
    对有标签节点按分层 K 折划分，确保每折正负样本比例一致。
    """
    label_idx = np.where(mask == 1)[0]
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=100)
    splits = []
    for tr, va in kf.split(label_idx, y[label_idx]):
        ti, vi = label_idx[tr], label_idx[va]
        splits.append((ti, y[ti], vi, y[vi]))
    return splits


# ──────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TREE 框架演示  —  数据集：CPDB")
    print("=" * 60)

    # 创建必要目录
    for d in [PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1 ──────────────────────────────────
    print("\n[Step 1] 读取网络数据")
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        read_h5file(H5_FILE[DATASET])

    max_degree = int(np.sum(adj, axis=-1).max()) + 1
    degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=DATASET), degree)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=DATASET), features)

    # ── Step 2 ──────────────────────────────────
    print("\n[Step 2] 子图采样")
    subgraph_path = format_filename(
        PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE,
        dataset=DATASET, strategy='rw',
        n_channel=N_GRAPHS, n_neighbor=N_NEIGHBORS
    )
    if os.path.exists(subgraph_path):
        print(f"  检测到缓存，跳过采样：{subgraph_path}")
    else:
        walks, spatial = build_subgraphs(adj)
        pickle_dump(subgraph_path, walks)
        pickle_dump(
            format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE,
                            dataset=DATASET, strategy='rw',
                            n_channel=N_GRAPHS, n_neighbor=N_NEIGHBORS),
            spatial
        )

    # ── Step 3 ──────────────────────────────────
    print(f"\n[Step 3] {CV_FOLDS} 折交叉验证划分")
    y_tv      = np.logical_or(y_train, y_val)
    mask_tv   = np.logical_or(train_mask, val_mask)
    cv_splits = make_cv_splits(y_tv, mask_tv)
    test_id   = np.where(test_mask == 1)[0]
    y_test    = y_test[test_id]
    print(f"  测试集：{len(test_id)} 个节点，正样本 {int(y_test.sum())} 个")

    # ── Step 4 ──────────────────────────────────
    print(f"\n[Step 4] 模型训练（{CV_FOLDS} 折 × {N_EPOCH} epoch）")
    results = {'auc': [], 'aupr': [], 'acc': []}

    for fold in range(CV_FOLDS):
        print(f"\n  ── 第 {fold + 1}/{CV_FOLDS} 折 ──")
        tr_id, y_tr, va_id, y_va = cv_splits[fold]
        log = train(
            Kfold=fold, dataset=DATASET,
            train_label=y_tr, test_label=y_test, val_label=y_va,
            train_id=tr_id, test_id=test_id, val_id=va_id,
            n_graphs=N_GRAPHS, n_neighbors=N_NEIGHBORS, n_layers=N_LAYERS,
            spatial_type='rw', max_degree=max_degree,
            batch_size=BATCH_SIZE, embed_dim=64, num_heads=4,
            d_sp_enc=DFF, dff=DFF, l2_weights=5e-7,
            lr=LR, dropout=DROPOUT, loss_mul=LOSS_MUL,
            optimizer='adam', n_epoch=N_EPOCH,
            callbacks_to_add=['modelcheckpoint', 'earlystopping']
        )
        results['auc'].append(log['test_auc'])
        results['aupr'].append(log['test_aupr'])
        results['acc'].append(log['test_acc'])

    # ── Step 5 ──────────────────────────────────
    print("\n[Step 5] 评估结果")
    avg_auc  = np.mean(results['auc'])
    avg_aupr = np.mean(results['aupr'])
    avg_acc  = np.mean(results['acc'])
    std_auc  = np.std(results['auc'])
    std_aupr = np.std(results['aupr'])

    print(f"""
  数据集  : {DATASET}
  AUC     : {avg_auc:.4f} ± {std_auc:.4f}
  AUPR    : {avg_aupr:.4f} ± {std_aupr:.4f}
  ACC     : {avg_acc:.4f}
    """)

    summary = {
        'dataset': DATASET, 'cv_folds': CV_FOLDS, 'n_epoch': N_EPOCH,
        'avg_auc': round(avg_auc, 4), 'auc_std': round(std_auc, 4),
        'avg_aupr': round(avg_aupr, 4), 'aupr_std': round(std_aupr, 4),
        'avg_acc': round(avg_acc, 4),
    }
    write_log(format_filename(LOG_DIR, RESULT_LOG[DATASET]), summary, 'a')
    print(f"  结果已保存至 log/{RESULT_LOG[DATASET]}")


if __name__ == '__main__':
    main()
