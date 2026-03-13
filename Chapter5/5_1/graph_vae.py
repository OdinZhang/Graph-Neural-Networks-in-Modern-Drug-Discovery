"""
Graph Variational Autoencoder (GraphVAE) 分子生成
基于变分自编码器的分子图生成模型，整合基类和示例

参考: Simonovsky & Komogorov (2017) "GraphVAE: Learning Generative Models for Graphs"
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 基础类：变分自编码器
# ==========================================


class BaseVAE(nn.Module, ABC):
    """变分自编码器基础抽象类"""

    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码器：将输入映射到隐空间的分布参数 (mu, logvar)"""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码器：将隐变量映射回数据空间"""
        pass

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick)
        实现从分布中采样并保持梯度回传
        公式: z = mu + sigma * epsilon, 其中 epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播：编码 -> 重参数化采样 -> 解码"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 ELBO (Evidence Lower Bound) 损失
        L = Reconstruction Loss + KL Divergence
        """
        # 1. 重构损失 (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # 2. KL 散度损失
        # D_KL(N(mu, sigma) || N(0, I)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / x.size(0)

        total_loss = recon_loss + kld_weight * kld_loss
        return total_loss, recon_loss, kld_loss


# ==========================================
# 模型实现：GraphVAE
# ==========================================


class GraphVAE(BaseVAE):
    """图变分自编码器：直接生成分子图的节点特征和邻接矩阵"""

    def __init__(
        self,
        num_atoms: int = 10,
        feature_dim: int = 64,
        hidden_dims: list = [256, 128],
        latent_dim: int = 32,
    ):
        super(GraphVAE, self).__init__()

        self.num_atoms = num_atoms
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 数据总维度：节点特征 (N*F) + 邻接矩阵 (N*N)
        self.data_dim = num_atoms * feature_dim + num_atoms * num_atoms

        # 编码器网络 (MLP)
        encoder_layers = []
        prev_dim = self.data_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # 解码器网络 (MLP)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.decoder_backbone = nn.Sequential(*decoder_layers)

        self.fc_out = nn.Linear(prev_dim, self.data_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码数据到隐空间分布"""
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从隐变量解码回展平的数据"""
        h = self.decoder_backbone(z)
        x = self.fc_out(h)
        return x

    def decode_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将展平向量解析为分子图组件"""
        batch_size = x.size(0)

        # 1. 节点特征
        node_flat = x[:, : self.num_atoms * self.feature_dim]
        node_features = node_flat.view(batch_size, self.num_atoms, self.feature_dim)

        # 2. 邻接矩阵
        edge_flat = x[:, self.num_atoms * self.feature_dim :]
        adjacency = torch.sigmoid(edge_flat).view(
            batch_size, self.num_atoms, self.num_atoms
        )

        # 3. 保证对称性和无自环
        adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2.0
        mask = 1.0 - torch.eye(self.num_atoms, device=x.device)
        adjacency = adjacency * mask

        return node_features, adjacency

    def sample(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """从隐空间先验分布 N(0, I) 采样并生成分子"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        x = self.decode(z)
        return self.decode_graph(x)


# ==========================================
# 数据集
# ==========================================


class SimpleMoleculeDataset:
    """简单的分子数据集：生成随机分子图"""

    def __init__(
        self, num_samples: int = 1000, num_atoms: int = 10, feature_dim: int = 64
    ):
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim

        # 模拟分子数据 (N*F + N*N)
        self.data = torch.randn(
            num_samples, num_atoms * feature_dim + num_atoms * num_atoms
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return self.num_samples


# ==========================================
# 训练和可视化函数
# ==========================================


def train_vae(
    model: GraphVAE,
    dataloader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    kld_weight: float = 1.0,
):
    """训练变分自编码器"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    history = {"total": [], "recon": [], "kld": []}

    for epoch in range(num_epochs):
        epoch_loss, epoch_recon, epoch_kld = 0, 0, 0

        for x in dataloader:
            # 前向传播
            x_recon, mu, logvar, _ = model(x)

            # 计算损失
            loss, recon, kld = model.loss_function(x, x_recon, mu, logvar, kld_weight)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kld += kld.item()

        avg_loss = epoch_loss / len(dataloader)
        history["total"].append(avg_loss)
        history["recon"].append(epoch_recon / len(dataloader))
        history["kld"].append(epoch_kld / len(dataloader))

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} (Recon: {history['recon'][-1]:.4f}, KLD: {history['kld'][-1]:.4f})"
            )

    return history


def visualize_molecules(
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    num_samples: int = 4,
    save_path: str = "vae_molecules.png",
):
    """可视化生成的分子图结构"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(num_samples, len(node_features))):
        ax = axes[i]
        adj_np = adjacency[i].detach().numpy()
        # 二值化邻接矩阵用于绘图
        adj_binary = (adj_np > 0.5).astype(float)
        G = nx.from_numpy_array(adj_binary)

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color="salmon",
            edge_color="gray",
            node_size=400,
            font_size=8,
        )
        ax.set_title(f"VAE Generated Molecule {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_interpolation(
    model, dataset, num_steps=10, save_path="vae_interpolation.png"
):
    """可视化隐空间中的分子插值"""
    model.eval()
    with torch.no_grad():
        # 选取数据集中的两个样本
        x1 = dataset[0].unsqueeze(0)
        x2 = dataset[1].unsqueeze(0)

        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        alphas = torch.linspace(0, 1, num_steps)
        interpolated_z = torch.zeros(num_steps, model.latent_dim)
        for i, alpha in enumerate(alphas):
            interpolated_z[i] = (1 - alpha) * mu1 + alpha * mu2

        x_recon = model.decode(interpolated_z)
        _, adj_seq = model.decode_graph(x_recon)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(num_steps):
        ax = axes[i]
        G = nx.from_numpy_array((adj_seq[i].numpy() > 0.5).astype(float))
        nx.draw(G, nx.spring_layout(G), ax=ax, node_size=100, node_color="lightgreen")
        ax.set_title(f"alpha = {i / (num_steps - 1):.1f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Interpolation saved to {save_path}")


# ==========================================
# 主程序
# ==========================================


def main():
    """主程序入口"""
    print("=" * 60)
    print("GraphVAE 分子生成模型训练与生成")
    print("=" * 60)

    # 参数设置
    num_atoms = 10
    feature_dim = 64
    hidden_dims = [256, 128]
    latent_dim = 32
    batch_size = 32
    num_epochs = 50
    lr = 1e-3

    # 1. 创建模型
    model = GraphVAE(num_atoms, feature_dim, hidden_dims, latent_dim)

    # 2. 创建数据
    dataset = SimpleMoleculeDataset(500, num_atoms, feature_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 训练
    history = train_vae(model, dataloader, num_epochs, lr)

    # 4. 生成与可视化
    node_features, adjacency = model.sample(num_samples=4)
    visualize_molecules(node_features, adjacency)
    visualize_interpolation(model, dataset)

    # 5. 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["total"], label="Total Loss")
    plt.plot(history["recon"], label="Recon Loss")
    plt.plot(history["kld"], label="KL Loss")
    plt.title("GraphVAE Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vae_training_curves.png")
    plt.close()

    print("\n完成！结果已保存。")


if __name__ == "__main__":
    main()
