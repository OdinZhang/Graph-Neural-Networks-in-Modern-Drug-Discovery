"""
train.py — iGRLDTI 训练工具函数
==================================
提供以下核心函数供 run_demo.py 调用：

  aver(hops, adj, feature_list, alpha=0.15)
      根据每个节点的最优跳数 h_v，对各跳特征做阻尼加权平均，
      得到 NDLS-F 平滑后的节点表示。

  propagate(features, k, adj_norm)
      将特征在图上传播 k 跳，返回各跳特征列表。

  cal_hops(adj, feature_list, norm_fea_inf, k, epsilon)
      计算每个节点的收敛跳数（NDLS 核心逻辑）。

  train(epoch, model, optimizer, feature, record,
        idx_train, idx_val, idx_test, labels)
      单轮监督训练，返回验证集准确率（用于节点分类任务）。
"""

import numpy as np
import torch
import torch.nn.functional as F

from utils import accuracy

# 固定随机种子，保证实验可复现
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def aver(hops, adj, feature_list, alpha=0.15):
    """对每个节点，将 0~h_v 跳的特征做阻尼加权平均。

    公式：
        若 h_v == 0：直接使用原始特征 X^(0)_v
        否则：f_v = mean_{j=0}^{h_v-1} [(1-α)·X^(j)_v + α·X^(0)_v]

    Args:
        hops        (Tensor): 每个节点的最优跳数，shape = (N,)
        adj         (sparse): 稀疏邻接矩阵，用于获取节点数 N
        feature_list(list)  : 各跳特征，feature_list[k] = Â^k · X
        alpha       (float) : 阻尼系数，控制保留原始特征的比重

    Returns:
        Tensor: 平滑后的节点特征，shape = (N, d)
    """
    input_feature = []
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
        else:
            fea = 0
            for j in range(hop):
                fea += (1 - alpha) * feature_list[j][i].unsqueeze(0) \
                     + alpha       * feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    return torch.cat(input_feature, dim=0)


def propagate(features, k, adj_norm):
    """将特征在图上传播 k 跳。

    Args:
        features (Tensor): 初始节点特征，shape = (N, d)
        k        (int)   : 传播跳数
        adj_norm (sparse): 归一化邻接矩阵 Â

    Returns:
        list: 长度为 k 的特征列表，第 i 项为 Â^i · X
    """
    feature_list = [features]
    for _ in range(1, k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list


def cal_hops(adj, feature_list, norm_fea_inf, k, epsilon=0.02):
    """计算每个节点的收敛跳数（NDLS 核心）。

    h_v = 最小的 k，使得 ||X^(k)_v - X_∞_v||₂ < epsilon

    Args:
        adj         (sparse): 邻接矩阵（用于获取节点数 N）
        feature_list(list)  : 各跳特征列表
        norm_fea_inf(Tensor): 全局稳态特征 X_∞，shape = (1, d) 或 (N, d)
        k           (int)   : 最大跳数
        epsilon     (float) : 收敛阈值

    Returns:
        Tensor: 每个节点的收敛跳数，shape = (N,)
    """
    hops        = torch.zeros(adj.shape[0])
    mask_before = torch.zeros(adj.shape[0], dtype=torch.bool)

    for i in range(k):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist < epsilon) & (~mask_before)
        mask_before[mask] = True
        hops[mask] = i

    # 未收敛的节点使用最大跳数
    hops[~mask_before] = k - 1
    return hops


def train(model, optimizer, feature, record,
          idx_train, idx_val, idx_test, labels):
    """单轮监督训练（用于节点分类任务）。

    Args:
        model     (nn.Module): DNN 模型
        optimizer            : Adam 优化器
        feature   (Tensor)   : 平滑后的节点特征
        record    (dict)     : {val_acc -> test_acc} 记录字典
        idx_train/idx_val/idx_test (Tensor): 各集合节点索引
        labels    (Tensor)   : 节点标签

    Returns:
        float: 当前轮次验证集准确率
    """
    model.train()
    optimizer.zero_grad()
    output, _ = model(feature)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output, _ = model(feature)
    acc_val  = accuracy(output[idx_val],  labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()
    return acc_val
