"""
model.py — iGRLDTI 神经网络模型定义
=====================================
DNN：两层全连接网络，用于将平滑后的节点特征投影到低维嵌入空间。

网络结构：
    输入层 (nfeat)
        ↓  Dropout + Linear
    隐藏层 (nhid)
        ↓  Dropout + Linear
    输出层 (nclass)  →  log_softmax  （分类头，可用于监督训练）
                     →  线性输出 x   （嵌入向量，用于下游相似度计算）
"""

import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """两层全连接网络（Deep Neural Network）。

    Args:
        nfeat  (int): 输入特征维度
        nhid   (int): 隐藏层维度
        nclass (int): 输出维度（嵌入维度 / 类别数）
        dropout(float): Dropout 比率，用于防止过拟合
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DNN, self).__init__()
        self.fcn1    = nn.Linear(nfeat, nhid)
        self.fcn2    = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        # 第一层：Dropout → 全连接
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn1(x)
        # 第二层：Dropout → 全连接
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        # 返回 (分类概率的对数, 原始嵌入向量)
        return F.log_softmax(x, dim=1), x
