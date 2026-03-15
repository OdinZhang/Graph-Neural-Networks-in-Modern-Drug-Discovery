"""
utils.py — iGRLDTI 数据加载与图处理工具
==========================================
本模块提供以下核心函数：

  aug_random_walk(adj)
      增广随机游走归一化：Â = D̃^{-1} Ã，其中 Ã = A + I

  sparse_mx_to_torch_sparse_tensor(sparse_mx)
      将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量

  accuracy(output, labels)
      计算节点分类准确率

  load_file_as_Adj_matrix(Alledge, features)
      将边列表转为 scipy 稀疏邻接矩阵

  load_data(edgelist, node_features, node_labels)
      加载 DTI 图数据，返回 (adj, features, labels, idx_train, idx_val, idx_test)
"""

import numpy as np
import scipy.sparse as sp

import torch


# ------------------------------------------------------------------
# 图归一化
# ------------------------------------------------------------------

def aug_random_walk(adj):
    """增广随机游走归一化邻接矩阵：Â = D̃^{-1} Ã

    先加自环（Ã = A + I），再做行归一化，使每行和为 1。
    等价于在含自环的图上做随机游走的转移矩阵。

    Args:
        adj (scipy.sparse): 原始对称邻接矩阵

    Returns:
        scipy.sparse.coo_matrix: 归一化后的矩阵 Â
    """
    adj = adj + sp.eye(adj.shape[0])    # 加自环
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv   = np.power(row_sum, -1.0).flatten()
    d_mat   = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量。

    Args:
        sparse_mx: scipy 稀疏矩阵

    Returns:
        torch.sparse.FloatTensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices   = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape  = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ------------------------------------------------------------------
# 评估
# ------------------------------------------------------------------

def accuracy(output, labels):
    """计算节点分类准确率。

    Args:
        output (Tensor): 模型输出 log-softmax，shape = (N, C)
        labels (Tensor): 真实标签，shape = (N,)

    Returns:
        Tensor: 标量准确率
    """
    preds   = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


# ------------------------------------------------------------------
# 数据加载（DTI 专用）
# ------------------------------------------------------------------

def load_file_as_Adj_matrix(Alledge, features):
    """将边列表（药物-靶点对）转换为 scipy 稀疏邻接矩阵。

    Args:
        Alledge  (DataFrame): 边列表，每行为 [src_node_id, dst_node_id]
        features (DataFrame): 节点特征矩阵（仅用于获取节点总数）

    Returns:
        scipy.sparse.csr_matrix: 形状为 (N, N) 的邻接矩阵
    """
    n = len(features)
    relation_matrix = np.zeros((n, n))
    for i, j in np.array(Alledge):
        relation_matrix[int(i), int(j)] = 1
    return sp.csr_matrix(relation_matrix, dtype=np.float32)


def load_data(edgelist, node_features, node_labels):
    """加载 DTI 图数据，返回训练/验证/测试划分。

    节点划分约定（固定划分，非随机）：
        训练集：节点 0   ~ 499   （500 个）
        验证集：节点 500 ~ 659   （160 个）
        测试集：节点 660 ~ N-1   （剩余节点）

    Args:
        edgelist      (DataFrame): 边列表 [src, dst]
        node_features (DataFrame): 节点特征矩阵，shape = (N, d)
        node_labels   (Series)   : 节点标签，shape = (N,)

    Returns:
        tuple: (adj, features, labels, idx_train, idx_val, idx_test)
            adj      — scipy 稀疏邻接矩阵，shape = (N, N)
            features — torch.FloatTensor，shape = (N, d)
            labels   — torch.LongTensor，shape = (N,)
            idx_*    — torch.LongTensor，各集合节点索引
    """
    n = node_features.shape[0]

    features = sp.csr_matrix(node_features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels   = torch.LongTensor(np.array(node_labels))
    adj      = load_file_as_Adj_matrix(edgelist, node_features)

    idx_train = torch.LongTensor(range(500))
    idx_val   = torch.LongTensor(range(500, 660))
    idx_test  = torch.LongTensor(range(660, n))

    return adj, features, labels, idx_train, idx_val, idx_test
