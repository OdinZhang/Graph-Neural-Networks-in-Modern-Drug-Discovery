"""
VTJNN (Variational Junction Tree Encoder-Decoder) 图风格转换分子优化

基于嵌入的分子优化方法：在 VAE 隐空间中学习风格转换向量 (Style Transfer Vector)，
实现从源分子到目标分子的属性迁移。

核心思想 (来自文献):
  图风格转换的思路可类比图像风格转换，即在隐空间中直接学习如何将初始分子转化为目标分子。
  VTJNN 在 JTVAE 框架下，将分子映射到联合树 (Junction Tree) 与图级别 (Graph-level) 的
  隐空间表征，并在其中实现分子间的"风格转换"。

数学公式:
  对于两个分子 X 和 Y (起始分子和优化分子)，VTJNN 的训练目标为学习风格转换向量:
    delta_z_T = z_T^Y - z_T^X    (树级别)
    delta_z_G = z_G^Y - z_G^X    (图级别)
  其中 z_T, z_G 分别为联合树和图级别的表征，delta_z 为对风格转换向量分布的采样。

  推理阶段，将原始分子的 z_T^X 和 z_G^X 与采样的风格转换向量相加:
    z_T^Y = z_T^X + delta_z_T
    z_G^Y = z_G^X + delta_z_G
  最后通过解码器将 (z_T^Y, z_G^Y) 解码得到风格转换后的分子。

简化说明:
  完整的 Junction Tree 算法 (环分解、树结构编解码) 在单文件中过于复杂。
  本实现用"双通道 GCN 编码器"近似联合树与图级别的双重表征:
    - 树编码器 (TreeEncoder): 在稀疏骨架图 (spanning tree) 上做消息传递，模拟联合树表征
    - 图编码器 (GraphEncoder): 在完整分子图上做消息传递，捕获环和全局结构信息
  两个编码器分别输出 z_T 和 z_G，拼接后送入解码器重构分子图。

参考:
  Jin et al. (2018) "Junction Tree Variational Autoencoder for Molecular Graph Generation"
  Jin et al. (2019) "Learning Multimodal Graph-to-Graph Translation for Molecular Optimization"
"""

import os
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 基础组件：图卷积层 (Graph Convolution Layer)
# ============================================================


class GraphConvolution(nn.Module):
    """
    简单的图卷积层 (GCN Layer)
    实现归一化的邻接矩阵消息传递:
      H^{(l+1)} = sigma( D^{-1/2} A_hat D^{-1/2} H^{(l)} W^{(l)} )
    其中 A_hat = A + I (加自环), D 为度矩阵。
    """

    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征矩阵 [batch, N, F_in]
            adj: 邻接矩阵 [batch, N, N]
        Returns:
            输出特征 [batch, N, F_out]
        """
        # 添加自环: A_hat = A + I
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj_hat = adj + eye

        # 度矩阵 D: d_i = sum_j A_hat_{ij}
        degree = adj_hat.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # 归一化: D^{-1} A_hat (简化版对称归一化)
        adj_norm = adj_hat / degree

        # 消息传递: H' = A_norm * X * W
        support = self.linear(x)  # [batch, N, F_out]
        output = torch.bmm(adj_norm, support)  # [batch, N, F_out]
        return output


# ============================================================
# 模型实现：树编码器 / 图编码器 / 解码器 / VTJNN
# ============================================================


class TreeEncoder(nn.Module):
    """
    树编码器 (Tree Encoder) — 近似联合树 (Junction Tree) 表征

    在真实的 JTVAE 中，联合树是通过对分子图进行环分解 (Ring Decomposition) 得到的树结构，
    树的每个节点代表一个化学子结构（原子、环、片段）。树编码器在该树上做消息传递。

    本简化实现: 从分子邻接矩阵中提取一棵生成树 (Spanning Tree)，
    并在该稀疏结构上用 GCN 做消息传递，模拟联合树级别的编码。
    输出: 树级隐变量的均值 mu_T 和对数方差 logvar_T。
    """

    def __init__(self, node_dim: int, hidden_dim: int, latent_dim: int):
        super(TreeEncoder, self).__init__()
        self.gcn1 = GraphConvolution(node_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        # 隐变量参数: mu 和 logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    @staticmethod
    def extract_spanning_tree(adj: torch.Tensor) -> torch.Tensor:
        """
        从邻接矩阵中提取生成树 (Spanning Tree) 作为联合树的近似。
        对每个 batch 中的图，使用 BFS 提取一棵树。

        Args:
            adj: [batch, N, N] 邻接矩阵
        Returns:
            tree_adj: [batch, N, N] 生成树的邻接矩阵
        """
        batch_size, N, _ = adj.shape
        tree_adj = torch.zeros_like(adj)

        for b in range(batch_size):
            adj_np = adj[b].detach().cpu().numpy()
            # 找到有连接的节点
            active = np.where(adj_np.sum(axis=1) > 0)[0]
            if len(active) <= 1:
                continue

            # BFS 构建生成树
            visited = set()
            queue = [active[0]]
            visited.add(active[0])

            while queue:
                node = queue.pop(0)
                neighbors = np.where(adj_np[node] > 0.5)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        tree_adj[b, node, nb] = 1.0
                        tree_adj[b, nb, node] = 1.0
                        queue.append(nb)

        return tree_adj

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 节点特征 [batch, N, node_dim]
            adj: 完整邻接矩阵 [batch, N, N]
        Returns:
            mu_T, logvar_T: 树级隐变量参数 [batch, latent_dim]
        """
        # 提取生成树邻接矩阵
        tree_adj = self.extract_spanning_tree(adj).to(x.device)

        # 在生成树上做 GCN 消息传递
        h = F.relu(self.gcn1(x, tree_adj))
        h = F.relu(self.gcn2(h, tree_adj))

        # 图级别 Readout: 平均池化
        # 使用 adj 的行和来识别有效节点
        node_mask = (adj.sum(dim=-1) > 0).float().unsqueeze(-1)  # [batch, N, 1]
        h_masked = h * node_mask
        num_nodes = node_mask.sum(dim=1).clamp(min=1.0)  # [batch, 1]
        graph_feat = h_masked.sum(dim=1) / num_nodes  # [batch, hidden_dim]

        mu = self.fc_mu(graph_feat)
        logvar = self.fc_logvar(graph_feat)
        return mu, logvar


class GraphEncoder(nn.Module):
    """
    图编码器 (Graph Encoder) — 在完整分子图上做消息传递

    捕获分子图的全局结构信息（包括环结构、全局连接等），
    输出图级隐变量的均值 mu_G 和对数方差 logvar_G。
    """

    def __init__(self, node_dim: int, hidden_dim: int, latent_dim: int):
        super(GraphEncoder, self).__init__()
        self.gcn1 = GraphConvolution(node_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 节点特征 [batch, N, node_dim]
            adj: 完整邻接矩阵 [batch, N, N]
        Returns:
            mu_G, logvar_G: 图级隐变量参数 [batch, latent_dim]
        """
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))

        # 图级别 Readout: 平均池化
        node_mask = (adj.sum(dim=-1) > 0).float().unsqueeze(-1)
        h_masked = h * node_mask
        num_nodes = node_mask.sum(dim=1).clamp(min=1.0)
        graph_feat = h_masked.sum(dim=1) / num_nodes

        mu = self.fc_mu(graph_feat)
        logvar = self.fc_logvar(graph_feat)
        return mu, logvar


class GraphDecoder(nn.Module):
    """
    图解码器 (Graph Decoder)

    将拼接的隐变量 z = [z_T; z_G] 解码为分子图的节点特征和邻接矩阵。
    在真实的 JTVAE 中，解码分为树解码和图解码两步:
    1. 树解码: 从 z_T 生成联合树结构
    2. 图解码: 根据联合树和 z_G 还原完整分子图

    本简化实现: 使用 MLP 直接从 z 解码节点特征和邻接矩阵。
    """

    def __init__(self, latent_dim: int, hidden_dim: int, max_nodes: int, node_dim: int):
        super(GraphDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim

        # 输入维度为拼接的 z_T 和 z_G
        input_dim = latent_dim * 2

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 节点特征预测头
        self.fc_nodes = nn.Linear(hidden_dim, max_nodes * node_dim)
        # 邻接矩阵预测头
        self.fc_adj = nn.Linear(hidden_dim, max_nodes * max_nodes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: 拼接的隐变量 [batch, latent_dim * 2]
        Returns:
            x_hat: 重构的节点特征 [batch, N, node_dim]
            adj_hat: 重构的邻接矩阵 [batch, N, N]
        """
        h = self.decoder(z)

        # 节点特征
        x_hat = torch.sigmoid(self.fc_nodes(h))
        x_hat = x_hat.view(-1, self.max_nodes, self.node_dim)

        # 邻接矩阵 (保证对称性、无自环)
        adj_hat = torch.sigmoid(self.fc_adj(h))
        adj_hat = adj_hat.view(-1, self.max_nodes, self.max_nodes)
        adj_hat = (adj_hat + adj_hat.transpose(1, 2)) / 2.0
        # 去除自环
        eye = torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
        adj_hat = adj_hat * (1.0 - eye)

        return x_hat, adj_hat


class StyleTransferMLP(nn.Module):
    """
    风格转换向量的参数化网络 (Style Transfer Vector Parameterization)

    参考 HierG2G 的做法: 使用 MLP 对风格转换向量进行参数化，
    输出高斯分布的均值和方差:
      mu_delta, logvar_delta = MLP(z_source_T, z_source_G)

    这样风格转换向量不再是一个固定的全局参数，而是根据源分子的隐表征
    动态预测的条件分布，从而适应不同的源分子结构。
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super(StyleTransferMLP, self).__init__()
        # 输入: z_T 和 z_G 拼接
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 分别预测树级和图级的风格转换向量分布参数
        self.fc_mu_T = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_T = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu_G = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_G = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, z_T: torch.Tensor, z_G: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_T: 源分子树级隐变量 [batch, latent_dim]
            z_G: 源分子图级隐变量 [batch, latent_dim]
        Returns:
            mu_delta_T, logvar_delta_T: 树级风格转换向量分布参数
            mu_delta_G, logvar_delta_G: 图级风格转换向量分布参数
        """
        h = self.net(torch.cat([z_T, z_G], dim=-1))
        mu_dT = self.fc_mu_T(h)
        logvar_dT = self.fc_logvar_T(h)
        mu_dG = self.fc_mu_G(h)
        logvar_dG = self.fc_logvar_G(h)
        return mu_dT, logvar_dT, mu_dG, logvar_dG


class VTJNN(nn.Module):
    """
    VTJNN: Variational Junction Tree Encoder-Decoder for Molecular Optimization

    完整的图风格转换分子优化模型，包含:
    1. 双通道编码器 (TreeEncoder + GraphEncoder) -> z_T, z_G
    2. 解码器: (z_T, z_G) -> 分子图
    3. 风格转换网络: 学习 delta_z 的条件分布

    训练目标:
      L = L_recon(source) + L_recon(target)             # VAE 重构损失
        + beta * (L_KL(source) + L_KL(target))          # KL 散度正则化
        + alpha * L_transfer                             # 风格转换损失
        + gamma * L_KL(delta_z)                          # 风格转换向量的 KL 正则化

    其中风格转换损失确保:
      z_T^source + delta_z_T ≈ z_T^target
      z_G^source + delta_z_G ≈ z_G^target
    """

    def __init__(
        self,
        max_nodes: int = 9,
        node_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ):
        super(VTJNN, self).__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.latent_dim = latent_dim

        # 双通道编码器
        self.tree_encoder = TreeEncoder(node_dim, hidden_dim, latent_dim)
        self.graph_encoder = GraphEncoder(node_dim, hidden_dim, latent_dim)

        # 解码器
        self.decoder = GraphDecoder(latent_dim, hidden_dim, max_nodes, node_dim)

        # 风格转换向量参数化网络
        self.style_transfer = StyleTransferMLP(latent_dim, hidden_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick):
          z = mu + sigma * epsilon, 其中 epsilon ~ N(0, I)
        使采样过程可微，允许梯度通过采样操作回传。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        双通道编码: 同时获取树级和图级隐变量。

        Returns:
            mu_T, logvar_T, z_T: 树级隐变量参数和采样值
            mu_G, logvar_G, z_G: 图级隐变量参数和采样值
        """
        mu_T, logvar_T = self.tree_encoder(x, adj)
        mu_G, logvar_G = self.graph_encoder(x, adj)
        z_T = self.reparameterize(mu_T, logvar_T)
        z_G = self.reparameterize(mu_G, logvar_G)
        return mu_T, logvar_T, z_T, mu_G, logvar_G, z_G

    def decode(
        self, z_T: torch.Tensor, z_G: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码: 拼接 z_T 和 z_G 后重构分子图"""
        z = torch.cat([z_T, z_G], dim=-1)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        前向传播: 编码 -> 重参数化 -> 解码

        Returns:
            x_hat, adj_hat: 重构的分子图
            mu_T, logvar_T, z_T: 树级隐变量
            mu_G, logvar_G, z_G: 图级隐变量
        """
        mu_T, logvar_T, z_T, mu_G, logvar_G, z_G = self.encode(x, adj)
        x_hat, adj_hat = self.decode(z_T, z_G)
        return x_hat, adj_hat, mu_T, logvar_T, z_T, mu_G, logvar_G, z_G

    def compute_style_transfer(
        self,
        z_T_src: torch.Tensor,
        z_G_src: torch.Tensor,
        z_T_tgt: torch.Tensor,
        z_G_tgt: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        计算风格转换损失:
          1. 用 StyleTransferMLP 预测 delta_z 的分布参数
          2. 采样 delta_z
          3. 计算 z_src + delta_z 与 z_tgt 之间的 MSE

        Returns:
            transfer_loss: 风格转换 MSE 损失
            kl_delta: delta_z 分布的 KL 散度
            delta_z_T, delta_z_G: 采样的风格转换向量
        """
        # 预测风格转换向量的分布参数
        mu_dT, logvar_dT, mu_dG, logvar_dG = self.style_transfer(z_T_src, z_G_src)

        # 采样风格转换向量
        delta_z_T = self.reparameterize(mu_dT, logvar_dT)
        delta_z_G = self.reparameterize(mu_dG, logvar_dG)

        # 风格转换损失: z_src + delta_z 应接近 z_tgt
        transfer_loss_T = F.mse_loss(z_T_src + delta_z_T, z_T_tgt.detach())
        transfer_loss_G = F.mse_loss(z_G_src + delta_z_G, z_G_tgt.detach())
        transfer_loss = transfer_loss_T + transfer_loss_G

        # delta_z 的 KL 散度 (鼓励风格转换向量分布接近标准正态)
        kl_dT = -0.5 * torch.mean(1 + logvar_dT - mu_dT.pow(2) - logvar_dT.exp())
        kl_dG = -0.5 * torch.mean(1 + logvar_dG - mu_dG.pow(2) - logvar_dG.exp())
        kl_delta = kl_dT + kl_dG

        return transfer_loss, kl_delta, delta_z_T, delta_z_G, mu_dT, mu_dG

    def transfer(
        self, x_src: torch.Tensor, adj_src: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        推理阶段的风格转换 (Molecular Optimization):
          1. 编码源分子得到 z_T^src, z_G^src
          2. 预测风格转换向量 delta_z (使用均值，不采样)
          3. z_T^opt = z_T^src + delta_z_T
             z_G^opt = z_G^src + delta_z_G
          4. 解码得到优化后的分子

        Returns:
            x_opt, adj_opt: 优化后的分子图
            z_T_src, z_G_src: 源分子隐变量
            z_T_opt, z_G_opt: 优化后隐变量
        """
        # 编码 (使用均值，不添加噪声)
        mu_T, _, _, mu_G, _, _ = self.encode(x_src, adj_src)
        z_T_src = mu_T
        z_G_src = mu_G

        # 预测风格转换向量 (使用均值)
        mu_dT, _, mu_dG, _ = self.style_transfer(z_T_src, z_G_src)

        # 执行风格转换
        z_T_opt = z_T_src + mu_dT
        z_G_opt = z_G_src + mu_dG

        # 解码
        x_opt, adj_opt = self.decode(z_T_opt, z_G_opt)
        return x_opt, adj_opt, z_T_src, z_G_src, z_T_opt, z_G_opt


# ============================================================
# 数据集：成对分子数据集 (Paired Molecule Dataset)
# ============================================================


class PairedMoleculeDataset(Dataset):
    """
    成对分子数据集 (Paired Molecule Dataset)

    模拟 (X, Y) 分子对的训练数据，其中:
    - X: 源分子 (Source Molecule)，例如低溶解度的先导化合物
    - Y: 目标分子 (Target Molecule)，在 X 基础上经过局部修改后获得更优属性

    模拟策略:
    1. 生成源分子图: 随机图结构 + 随机原子类型 (One-hot 编码)
    2. 生成目标分子图: 在源分子基础上做小幅修改:
       - 可能增加少量节点 (模拟添加功能基团)
       - 扰动节点特征 (模拟原子类型替换)
       - 添加/删除少量边 (模拟成键变化)
    """

    def __init__(
        self,
        num_samples: int = 500,
        max_nodes: int = 9,
        node_dim: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.node_dim = node_dim

        rng = np.random.RandomState(seed)
        self.data = []

        for _ in range(num_samples):
            # ---- 生成源分子 ----
            num_src = rng.randint(4, max_nodes + 1)  # 至少 4 个原子
            # 节点特征: 随机 one-hot 原子类型
            types_src = rng.randint(0, node_dim, size=num_src)
            x_src = np.zeros((max_nodes, node_dim), dtype=np.float32)
            for i, t in enumerate(types_src):
                x_src[i, t] = 1.0

            # 邻接矩阵: 先生成连通的链式结构，再随机添加边
            adj_src = np.zeros((max_nodes, max_nodes), dtype=np.float32)
            for i in range(num_src - 1):
                adj_src[i, i + 1] = 1.0
                adj_src[i + 1, i] = 1.0
            # 随机添加额外的边 (模拟环结构)
            for i in range(num_src):
                for j in range(i + 2, num_src):
                    if rng.rand() < 0.15:
                        adj_src[i, j] = 1.0
                        adj_src[j, i] = 1.0

            # ---- 生成目标分子 (在源分子基础上修改) ----
            # 可能增加 0-2 个节点
            extra = rng.randint(0, min(3, max_nodes - num_src + 1))
            num_tgt = num_src + extra

            # 复制源分子特征并扰动
            x_tgt = np.zeros((max_nodes, node_dim), dtype=np.float32)
            x_tgt[:num_src] = x_src[:num_src]
            # 对部分已有节点替换原子类型 (风格转换)
            for i in range(num_src):
                if rng.rand() < 0.2:
                    new_type = rng.randint(0, node_dim)
                    x_tgt[i] = 0.0
                    x_tgt[i, new_type] = 1.0
            # 新增节点
            for i in range(num_src, num_tgt):
                t = rng.randint(0, node_dim)
                x_tgt[i, t] = 1.0

            # 邻接矩阵: 在源分子基础上修改
            adj_tgt = np.copy(adj_src)
            # 新增节点与现有节点连接
            for i in range(num_src, num_tgt):
                attach = rng.randint(0, num_src)
                adj_tgt[i, attach] = 1.0
                adj_tgt[attach, i] = 1.0
            # 随机添加/删除少量边
            for i in range(num_tgt):
                for j in range(i + 2, num_tgt):
                    if rng.rand() < 0.1:
                        adj_tgt[i, j] = 1.0 - adj_tgt[i, j]
                        adj_tgt[j, i] = adj_tgt[i, j]
            # 确保对角线为 0
            np.fill_diagonal(adj_tgt, 0.0)

            self.data.append(
                (
                    torch.from_numpy(x_src),
                    torch.from_numpy(adj_src),
                    torch.from_numpy(x_tgt),
                    torch.from_numpy(adj_tgt),
                )
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]


# ============================================================
# 训练函数
# ============================================================


def compute_vae_loss(
    x_hat: torch.Tensor,
    adj_hat: torch.Tensor,
    x: torch.Tensor,
    adj: torch.Tensor,
    mu_T: torch.Tensor,
    logvar_T: torch.Tensor,
    mu_G: torch.Tensor,
    logvar_G: torch.Tensor,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 损失函数: ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))

    包含:
    1. 重构损失 (Reconstruction Loss): BCE(x_hat, x) + BCE(adj_hat, adj)
    2. KL 散度 (KL Divergence): 树级 + 图级的 KL
    """
    # 重构损失
    recon_x = F.binary_cross_entropy(x_hat, x, reduction="mean")
    recon_adj = F.binary_cross_entropy(adj_hat, adj, reduction="mean")
    recon_loss = recon_x + recon_adj

    # KL 散度: D_KL(N(mu, sigma^2) || N(0, I))
    # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_T = -0.5 * torch.mean(1 + logvar_T - mu_T.pow(2) - logvar_T.exp())
    kl_G = -0.5 * torch.mean(1 + logvar_G - mu_G.pow(2) - logvar_G.exp())
    kl_loss = kl_T + kl_G

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vtjnn(
    model: VTJNN,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 0.5,
    alpha: float = 10.0,
    gamma: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    训练 VTJNN 模型

    损失函数:
      L_total = L_VAE(source) + L_VAE(target)           # 双端 VAE 重构
              + alpha * L_transfer                       # 风格转换向量 MSE
              + gamma * L_KL(delta_z)                    # 风格转换向量正则化

    Args:
        beta: KL 散度权重 (VAE 部分)
        alpha: 风格转换损失权重
        gamma: 风格转换向量 KL 权重
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    history = {
        "total_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "transfer_loss": [],
    }

    print("开始训练...")
    for epoch in range(num_epochs):
        epoch_total, epoch_recon, epoch_kl, epoch_transfer = 0, 0, 0, 0
        num_batches = 0

        for x_src, adj_src, x_tgt, adj_tgt in dataloader:
            x_src = x_src.to(device)
            adj_src = adj_src.to(device)
            x_tgt = x_tgt.to(device)
            adj_tgt = adj_tgt.to(device)

            optimizer.zero_grad()

            # ---- 1. VAE 重构 (源分子) ----
            (
                x_src_hat,
                adj_src_hat,
                mu_T_src,
                logvar_T_src,
                z_T_src,
                mu_G_src,
                logvar_G_src,
                z_G_src,
            ) = model(x_src, adj_src)
            vae_loss_src, recon_src, kl_src = compute_vae_loss(
                x_src_hat,
                adj_src_hat,
                x_src,
                adj_src,
                mu_T_src,
                logvar_T_src,
                mu_G_src,
                logvar_G_src,
                beta,
            )

            # ---- 2. VAE 重构 (目标分子) ----
            (
                x_tgt_hat,
                adj_tgt_hat,
                mu_T_tgt,
                logvar_T_tgt,
                z_T_tgt,
                mu_G_tgt,
                logvar_G_tgt,
                z_G_tgt,
            ) = model(x_tgt, adj_tgt)
            vae_loss_tgt, recon_tgt, kl_tgt = compute_vae_loss(
                x_tgt_hat,
                adj_tgt_hat,
                x_tgt,
                adj_tgt,
                mu_T_tgt,
                logvar_T_tgt,
                mu_G_tgt,
                logvar_G_tgt,
                beta,
            )

            # ---- 3. 风格转换损失 ----
            # delta_z = z_target - z_source
            # 训练: 使预测的 delta_z 分布逼近真实的 delta_z
            transfer_loss, kl_delta, _, _, _, _ = model.compute_style_transfer(
                z_T_src, z_G_src, z_T_tgt, z_G_tgt
            )

            # ---- 4. 总损失 ----
            loss = (
                vae_loss_src + vae_loss_tgt + alpha * transfer_loss + gamma * kl_delta
            )

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_src.item() + recon_tgt.item()
            epoch_kl += kl_src.item() + kl_tgt.item()
            epoch_transfer += transfer_loss.item()
            num_batches += 1

        # 记录平均损失
        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        avg_transfer = epoch_transfer / num_batches

        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["transfer_loss"].append(avg_transfer)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {avg_total:.4f} "
                f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
                f"Transfer: {avg_transfer:.4f})"
            )

    return history


# ============================================================
# 可视化函数
# ============================================================


def adj_to_graph(adj_np: np.ndarray, threshold: float = 0.5) -> nx.Graph:
    """将邻接矩阵转换为 NetworkX 图，移除孤立节点"""
    adj_binary = (adj_np > threshold).astype(float)
    np.fill_diagonal(adj_binary, 0)
    G = nx.from_numpy_array(adj_binary)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def get_node_colors(x_np: np.ndarray, node_list: list) -> list:
    """根据 one-hot 节点特征返回颜色列表"""
    palette = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]
    colors = []
    for n in node_list:
        if n < len(x_np):
            atom_type = int(np.argmax(x_np[n]))
            colors.append(palette[atom_type % len(palette)])
        else:
            colors.append("#cccccc")
    return colors


def visualize_transfer(
    model: VTJNN,
    dataset: PairedMoleculeDataset,
    device: str = "cpu",
    num_examples: int = 3,
    save_path: str = "vtjnn_transfer.png",
):
    """
    可视化风格转换结果:
      左列: 源分子 (Source)
      中列: 真实目标分子 (Ground Truth Target)
      右列: 模型生成的优化分子 (Generated / Transferred)
    """
    model.eval()
    fig, axes = plt.subplots(num_examples, 3, figsize=(14, 4.5 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Source Molecule (X)", "Target (GT) (Y)", "Transferred (Y')"]

    rng = np.random.RandomState(123)
    indices = rng.choice(len(dataset), size=num_examples, replace=False)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            x_src, adj_src, x_tgt, adj_tgt = dataset[idx]

            # 风格转换
            x_src_t = x_src.unsqueeze(0).to(device)
            adj_src_t = adj_src.unsqueeze(0).to(device)
            x_opt, adj_opt, _, _, _, _ = model.transfer(x_src_t, adj_src_t)

            # 转为 numpy
            graphs = [
                (x_src.numpy(), adj_src.numpy()),
                (x_tgt.numpy(), adj_tgt.numpy()),
                (x_opt.squeeze(0).cpu().numpy(), adj_opt.squeeze(0).cpu().numpy()),
            ]
            node_colors_list = ["#7fbfff", "#7fff7f", "#ff9f7f"]  # 蓝/绿/红

            for col, (x_np, adj_np) in enumerate(graphs):
                ax = axes[row, col]
                G = adj_to_graph(adj_np, threshold=0.5)

                if G.number_of_nodes() > 0:
                    colors = get_node_colors(x_np, list(G.nodes()))
                    pos = nx.spring_layout(
                        G, seed=np.random.RandomState(42 + idx * 3 + col)
                    )
                    nx.draw(
                        G,
                        pos,
                        ax=ax,
                        with_labels=True,
                        node_color=colors,
                        node_size=450,
                        font_size=8,
                        font_weight="bold",
                        edge_color="gray",
                        width=1.5,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "(empty)",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="gray",
                        transform=ax.transAxes,
                    )

                if row == 0:
                    ax.set_title(col_titles[col], fontsize=12, fontweight="bold")
                ax.axis("off")

    # 添加颜色图例 (原子类型)
    palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
    legend_patches = [
        mpatches.Patch(color=palette[i], label=f"Atom Type {i}")
        for i in range(min(dataset.node_dim, len(palette)))
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle(
        "VTJNN Style Transfer: Source → Target",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"风格转换可视化已保存至 {save_path}")


def plot_training_curve(history: dict, save_path: str = "vtjnn_training_curve.png"):
    """绘制训练损失曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["total_loss"]) + 1)

    # 左图: 总损失
    ax1.plot(epochs, history["total_loss"], "b-", linewidth=2, label="Total Loss")
    ax1.set_title("VTJNN Total Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 各分量损失
    ax2.plot(
        epochs,
        history["recon_loss"],
        "-",
        color="#e74c3c",
        linewidth=1.8,
        label="Reconstruction",
    )
    ax2.plot(
        epochs,
        history["kl_loss"],
        "-",
        color="#3498db",
        linewidth=1.8,
        label="KL Divergence",
    )
    ax2.plot(
        epochs,
        history["transfer_loss"],
        "-",
        color="#2ecc71",
        linewidth=1.8,
        label="Style Transfer",
    )
    ax2.set_title("VTJNN Loss Components", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


# ============================================================
# 主程序 (Main)
# ============================================================


def main():
    print("=" * 60)
    print("VTJNN 图风格转换分子优化")
    print("=" * 60)

    # ---- 配置参数 ----
    max_nodes = 9
    node_dim = 4
    hidden_dim = 64
    latent_dim = 32
    batch_size = 32
    num_epochs = 50
    lr = 1e-3
    beta = 0.5  # KL 权重
    alpha = 10.0  # 风格转换损失权重
    gamma = 0.1  # 风格转换向量 KL 权重
    num_samples = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("配置:")
    print(f"  最大原子数 (Max Nodes): {max_nodes}")
    print(f"  原子类型数 (Node Dim): {node_dim}")
    print(f"  隐藏层维度 (Hidden Dim): {hidden_dim}")
    print(f"  隐变量维度 (Latent Dim): {latent_dim}")
    print(f"  批次大小 (Batch Size): {batch_size}")
    print(f"  训练轮数 (Epochs): {num_epochs}")
    print(f"  学习率 (LR): {lr}")
    print(f"  KL 权重 beta: {beta}")
    print(f"  风格转换权重 alpha: {alpha}")
    print(f"  Delta-z KL 权重 gamma: {gamma}")
    print(f"  训练样本数: {num_samples}")
    print(f"  设备 (Device): {device}")

    # ---- 创建数据集 ----
    print("\n创建数据集...")
    dataset = PairedMoleculeDataset(
        num_samples=num_samples, max_nodes=max_nodes, node_dim=node_dim
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  成对分子数据: {len(dataset)} 对")

    # ---- 创建模型 ----
    print("创建模型...")
    model = VTJNN(
        max_nodes=max_nodes,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # ---- 训练 ----
    history = train_vtjnn(
        model,
        dataloader,
        num_epochs=num_epochs,
        lr=lr,
        beta=beta,
        alpha=alpha,
        gamma=gamma,
        device=device,
    )

    # ---- 保存图 ----
    # 确定保存路径 (在脚本所在目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    curve_path = os.path.join(script_dir, "vtjnn_training_curve.png")
    transfer_path = os.path.join(script_dir, "vtjnn_transfer.png")

    print("\n生成可视化结果...")
    plot_training_curve(history, save_path=curve_path)
    visualize_transfer(model, dataset, device=str(device), save_path=transfer_path)

    print("\n" + "=" * 60)
    print("VTJNN 运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
