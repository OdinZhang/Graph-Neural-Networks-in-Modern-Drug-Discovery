"""
MolGAN (Molecular Generative Adversarial Network) 分子生成
基于生成对抗网络的分子图生成模型，整合基类和示例

参考: De Cao & Kipf (2018) "MolGAN: An implicit generative model for small molecular graphs"
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
# 基础类：生成对抗网络 (GAN)
# ==========================================


class BaseGenerator(nn.Module, ABC):
    """生成器基类"""

    def __init__(self):
        super(BaseGenerator, self).__init__()

    @abstractmethod
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从隐变量 z 生成数据 (节点特征, 邻接矩阵)"""
        pass


class BaseDiscriminator(nn.Module, ABC):
    """判别器基类"""

    def __init__(self):
        super(BaseDiscriminator, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """评估分子图 (x, a) 的真实性得分"""
        pass


# ==========================================
# 模型实现：MolGAN
# ==========================================


class MolGANGenerator(BaseGenerator):
    """MolGAN 生成器：使用 Gumbel-Softmax 解决离散图生成的可微性问题"""

    def __init__(
        self,
        latent_dim: int = 128,
        num_atoms: int = 10,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        hidden_dims: list = [256, 512],
    ):
        super(MolGANGenerator, self).__init__()

        self.num_atoms = num_atoms
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types

        # MLP 主干网络
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        # 分支 1：生成节点特征 X
        self.nodes_layer = nn.Linear(prev_dim, num_atoms * num_atom_types)

        # 分支 2：生成邻接矩阵 A (多通道表示键类型)
        self.edges_layer = nn.Linear(prev_dim, num_bond_types * num_atoms * num_atoms)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播生成分子图组件"""
        batch_size = z.size(0)
        h = self.mlp(z)

        # 1. 节点特征 (Categorical)
        node_logits = self.nodes_layer(h).view(
            batch_size, self.num_atoms, self.num_atom_types
        )
        # 使用 Gumbel-Softmax 模拟离散采样
        node_features = F.gumbel_softmax(node_logits, tau=1.0, hard=True, dim=-1)

        # 2. 邻接张量 (Categorical edges)
        edge_logits = self.edges_layer(h).view(
            batch_size, self.num_bond_types, self.num_atoms, self.num_atoms
        )
        # 强制对称性: A = (A + A^T) / 2
        edge_logits = (edge_logits + edge_logits.transpose(-1, -2)) / 2.0
        # Gumbel-Softmax 采样边类别
        adjacency = F.gumbel_softmax(edge_logits, tau=1.0, hard=True, dim=1)

        return node_features, adjacency


class MolGANDiscriminator(BaseDiscriminator):
    """MolGAN 判别器：对分子图进行打分"""

    def __init__(
        self,
        num_atoms: int = 10,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        hidden_dims: list = [256, 128],
    ):
        super(MolGANDiscriminator, self).__init__()

        # 简化的判别器：将图展平输入 MLP (实际 MolGAN 使用图卷积)
        input_dim = (
            num_atoms * num_atom_types + num_bond_types * num_atoms * num_atoms
        )

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.3))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """计算分子图的真实性得分"""
        batch_size = x.size(0)
        # 展平分子图
        x_flat = x.view(batch_size, -1)
        a_flat = a.view(batch_size, -1)
        combined = torch.cat([x_flat, a_flat], dim=-1)

        return self.net(combined)


# ==========================================
# 数据集
# ==========================================


class SimpleMoleculeDataset:
    """简单的分子数据集：生成随机的 One-hot 分子图"""

    def __init__(
        self,
        num_samples: int = 1000,
        num_atoms: int = 10,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
    ):
        self.num_samples = num_samples

        # 节点特征: [Batch, N, AtomTypes]
        nodes = F.one_hot(
            torch.randint(0, num_atom_types, (num_samples, num_atoms)),
            num_atom_types,
        ).float()

        # 邻接矩阵: [Batch, BondTypes, N, N]
        adj = torch.zeros(num_samples, num_bond_types, num_atoms, num_atoms)
        for i in range(num_samples):
            for b in range(num_bond_types):
                # 随机生成对称邻接矩阵
                m = torch.bernoulli(torch.full((num_atoms, num_atoms), 0.1))
                m = torch.triu(m, diagonal=1)
                m = m + m.transpose(-1, -2)
                adj[i, b] = m

        self.nodes = nodes
        self.adj = adj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.nodes[idx], self.adj[idx]

    def __len__(self) -> int:
        return self.num_samples


# ==========================================
# 训练和可视化函数
# ==========================================


def train_molgan(
    generator,
    discriminator,
    dataloader,
    num_epochs: int = 100,
    lr: float = 1e-4,
    lambda_gp: float = 10.0,
):
    """训练 MolGAN 模型 (采用 WGAN-GP 策略)"""
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(num_epochs):
        epoch_g_loss, epoch_d_loss = 0, 0

        for real_x, real_a in dataloader:
            batch_size = real_x.size(0)
            device = real_x.device

            # --- 1. 训练判别器 ---
            opt_d.zero_grad()

            # 真实样本得分
            d_real = discriminator(real_x, real_a)

            # 生成样本得分
            z = torch.randn(batch_size, generator.mlp[0].in_features).to(device)
            fake_x, fake_a = generator(z)
            d_fake = discriminator(fake_x.detach(), fake_a.detach())

            # WGAN 损失
            wd = torch.mean(d_fake) - torch.mean(d_real)

            # 梯度惩罚 (Gradient Penalty)
            eps = torch.rand(batch_size, 1, 1).to(device)
            interp_x = (eps * real_x + (1 - eps) * fake_x).detach().requires_grad_(True)
            # 简化版 GP，仅对 X 进行
            d_interp = discriminator(interp_x, fake_a.detach())
            gradients = torch.autograd.grad(
                outputs=d_interp,
                inputs=interp_x,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
                retain_graph=True,
            )[0]
            gp = lambda_gp * ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()

            d_loss = wd + gp
            d_loss.backward()
            opt_d.step()

            # --- 2. 训练生成器 ---
            opt_g.zero_grad()
            fake_x, fake_a = generator(z)
            d_fake = discriminator(fake_x, fake_a)
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            opt_g.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        g_losses.append(epoch_g_loss / len(dataloader))
        d_losses.append(epoch_d_loss / len(dataloader))

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] G_Loss: {g_losses[-1]:.4f}, D_Loss: {d_losses[-1]:.4f}"
            )

    return g_losses, d_losses


def visualize_molecules(node_features, adjacency, save_path="molgan_molecules.png"):
    """可视化生成的分子图"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(4, len(node_features))):
        ax = axes[i]
        # 获取原子类型 (Argmax)
        atom_types = node_features[i].argmax(dim=-1).detach().numpy()
        # 获取边 (合并所有键类型通道)
        adj_sum = adjacency[i].sum(dim=0).detach().numpy()
        adj_binary = (adj_sum > 0.5).astype(float)

        G = nx.from_numpy_array(adj_binary)
        pos = nx.spring_layout(G)

        # 映射原子类型到颜色
        colors = ["lightgray", "skyblue", "orange", "palegreen", "pink"]
        node_colors = [colors[t % len(colors)] for t in atom_types]

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=500,
        )
        ax.set_title(f"MolGAN Generated {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


# ==========================================
# 主程序
# ==========================================


def main():
    """主程序入口"""
    print("=" * 60)
    print("MolGAN 分子生成模型训练与生成")
    print("=" * 60)

    # 参数设置
    latent_dim = 128
    num_atoms = 10
    num_atom_types = 5
    num_bond_types = 4
    batch_size = 32
    num_epochs = 50
    lr = 1e-4

    # 1. 创建模型
    generator = MolGANGenerator(latent_dim, num_atoms, num_atom_types, num_bond_types)
    discriminator = MolGANDiscriminator(num_atoms, num_atom_types, num_bond_types)

    # 2. 创建数据
    dataset = SimpleMoleculeDataset(500, num_atoms, num_atom_types, num_bond_types)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 训练
    g_losses, d_losses = train_molgan(generator, discriminator, dataloader, num_epochs, lr)

    # 4. 生成与可视化
    with torch.no_grad():
        z = torch.randn(4, latent_dim)
        fake_x, fake_a = generator(z)
        visualize_molecules(fake_x, fake_a)

    # 5. 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.title("MolGAN Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("molgan_training_curves.png")
    plt.close()

    print("\n完成！结果已保存。")


if __name__ == "__main__":
    main()
