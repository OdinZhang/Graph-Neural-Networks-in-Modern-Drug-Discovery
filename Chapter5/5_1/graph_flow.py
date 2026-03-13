"""
Graph Normalizing Flow 分子生成
基于规范化流的分子图生成模型，整合基类和示例
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 基础类：规范化流
# ==========================================


class CouplingLayer(nn.Module):
    """仿射耦合层：规范化流的基本构建模块"""

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, mask: Optional[torch.Tensor] = None
    ):
        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim

        # 创建交替掩码
        if mask is None:
            mask = torch.arange(input_dim) % 2 == 0
        condition_dim = int(mask.sum().item())
        self.register_buffer("mask", mask.float())

        # 计算缩放和平移的神经网络
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (input_dim - condition_dim)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向变换（用于训练）"""
        mask = self.mask > 0.5
        x1 = x[:, mask]
        x2 = x[:, ~mask]

        s_and_t = self.net(x1)
        s, t = torch.chunk(s_and_t, 2, dim=-1)

        z1 = x1
        z2 = x2 * torch.exp(-s) + t
        z = torch.cat([z1, z2], dim=-1)

        log_det_jacobian = -torch.sum(s, dim=1)
        return z, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """逆向变换（用于生成）"""
        mask = self.mask > 0.5
        z1 = z[:, mask]
        z2 = z[:, ~mask]

        s_and_t = self.net(z1)
        s, t = torch.chunk(s_and_t, 2, dim=-1)

        x1 = z1
        x2 = z2 * torch.exp(s) + t
        x = torch.cat([x1, x2], dim=-1)

        return x


class GraphNormalizingFlow(nn.Module):
    """图规范化流模型：用于生成分子图"""

    def __init__(
        self,
        num_atoms: int = 10,
        feature_dim: int = 64,
        num_flows: int = 4,
        hidden_dim: int = 128,
    ):
        super(GraphNormalizingFlow, self).__init__()

        self.num_atoms = num_atoms
        self.feature_dim = feature_dim

        # 数据维度：节点特征 + 邻接矩阵（简化为单一键类型）
        self.data_dim = num_atoms * feature_dim + num_atoms * num_atoms

        # 构建耦合层
        self.flows = nn.ModuleList(
            [CouplingLayer(self.data_dim, hidden_dim) for _ in range(num_flows)]
        )

        # 节点特征解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(self.data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms * feature_dim),
        )

        # 邻接矩阵解码器
        self.edge_decoder = nn.Sequential(
            nn.Linear(self.data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms * num_atoms),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向变换：数据 -> 隐空间"""
        log_det_jacobian = 0
        z = x
        for flow in self.flows:
            z, log_det = flow.forward(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """逆向变换：隐空间 -> 数据"""
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x

    def decode_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从展平数据解码为节点特征和邻接矩阵"""
        batch_size = x.size(0)

        # 解码节点特征
        node_features_flat = self.node_decoder(x)
        node_features = node_features_flat.view(
            batch_size, self.num_atoms, self.feature_dim
        )

        # 解码邻接矩阵
        edge_flat = self.edge_decoder(x)
        edge_probs = torch.sigmoid(edge_flat)
        adjacency = edge_probs.view(batch_size, self.num_atoms, self.num_atoms)

        # 消除对角线
        mask = 1.0 - torch.eye(self.num_atoms, device=x.device)
        adjacency = adjacency * mask

        return node_features, adjacency

    def sample(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成新样本"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.data_dim).to(device)
        x = self.inverse(z)
        return self.decode_graph(x)


# ==========================================
# 简单的分子数据集
# ==========================================


class SimpleMoleculeDataset:
    """简单的分子数据集：生成随机分子图"""

    def __init__(
        self, num_samples: int = 1000, num_atoms: int = 10, feature_dim: int = 64
    ):
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim

        # 生成随机节点特征
        self.node_features = torch.randn(num_samples, num_atoms, feature_dim)

        # 生成随机邻接矩阵（稀疏图）
        self.adjacency = torch.zeros(num_samples, num_atoms, num_atoms)
        for i in range(num_samples):
            # 随机生成边
            for j in range(num_atoms):
                for k in range(j + 1, num_atoms):
                    if torch.rand(1).item() < 0.15:  # 15%的概率有边
                        self.adjacency[i, j, k] = 1.0
                        self.adjacency[i, k, j] = 1.0

    def __getitem__(self, idx: int):
        return self.node_features[idx], self.adjacency[idx]

    def __len__(self) -> int:
        return self.num_samples


# ==========================================
# 训练和可视化函数
# ==========================================


def train_flow(
    model: GraphNormalizingFlow, dataloader, num_epochs: int = 50, lr: float = 1e-4
):
    """训练规范化流模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (node_features, adjacency) in enumerate(dataloader):
            # 展平数据
            batch_size = node_features.size(0)
            x = torch.cat(
                [node_features.view(batch_size, -1), adjacency.view(batch_size, -1)],
                dim=-1,
            )

            # 前向传播
            z, log_det_jacobian = model(x)

            # 检查NaN
            if torch.isnan(z).any() or torch.isnan(log_det_jacobian).any():
                print(f"Warning: NaN detected in epoch {epoch + 1}, batch {batch_idx}")
                continue

            # 计算损失：最大化对数似然
            # 先验分布（标准正态）的对数概率
            log_prob_z = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(
                2 * np.pi
            )
            loss = -torch.mean(log_prob_z + log_det_jacobian)

            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss in epoch {epoch + 1}, batch {batch_idx}")
                continue

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return losses


def visualize_molecules(
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    num_samples: int = 4,
    save_path: str = "generated_molecules.png",
):
    """可视化生成的分子图"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(num_samples, len(node_features))):
        ax = axes[i]
        ax.set_title(f"Generated Molecule Graph {i + 1}")

        # 使用NetworkX创建图
        adj_np = adjacency[i].detach().numpy()
        G = nx.from_numpy_array(adj_np)

        # 绘制图
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=500,
            font_size=8,
        )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {save_path}")


def print_molecule_stats(node_features: torch.Tensor, adjacency: torch.Tensor):
    """打印分子统计信息"""
    num_samples = len(node_features)

    print("\n" + "=" * 60)
    print("分子生成统计")
    print("=" * 60)

    for i in range(min(5, num_samples)):
        print(f"\n分子 {i + 1}:")
        print(f"  节点数量: {adjacency[i].shape[0]}")

        # 计算边数量
        adj_np = adjacency[i].detach().numpy()
        # 处理可能的NaN
        adj_np = np.nan_to_num(adj_np)
        num_edges = int(np.sum(adj_np) / 2)
        print(f"  边数量: {num_edges}")

        # 计算度数
        degrees = np.sum(adj_np, axis=1)
        # 过滤NaN
        degrees = degrees[~np.isnan(degrees)]
        if len(degrees) > 0:
            print(f"  平均度数: {np.mean(degrees):.2f}")
            print(f"  最大度数: {np.max(degrees):.2f}")
        else:
            print("  平均度数: NaN (invalid molecule)")
            print("  最大度数: NaN (invalid molecule)")

        # 显示邻接矩阵示例（前5x5）
        print("  邻接矩阵示例 (5x5):")
        print(adj_np[:5, :5])


# ==========================================
# 主程序
# ==========================================


def main():
    """主函数"""
    print("=" * 60)
    print("Graph Normalizing Flow 分子生成")
    print("=" * 60)

    # 设置参数
    num_atoms = 10
    feature_dim = 64
    num_flows = 4
    hidden_dim = 128
    batch_size = 32
    num_epochs = 50
    lr = 1e-4

    print("\n配置:")
    print(f"  原子数量: {num_atoms}")
    print(f"  特征维度: {feature_dim}")
    print(f"  耦合层数: {num_flows}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")

    # 创建模型
    print("\n创建模型...")
    model = GraphNormalizingFlow(
        num_atoms=num_atoms,
        feature_dim=feature_dim,
        num_flows=num_flows,
        hidden_dim=hidden_dim,
    )

    # 创建数据集
    print("创建数据集...")
    dataset = SimpleMoleculeDataset(
        num_samples=500, num_atoms=num_atoms, feature_dim=feature_dim
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # 训练模型
    print("\n开始训练...")
    losses = train_flow(model, dataloader, num_epochs=num_epochs, lr=lr)

    # 生成新分子
    print("\n生成新分子图...")
    num_generated = 10
    node_features, adjacency = model.sample(num_samples=num_generated)

    # 打印统计信息
    print_molecule_stats(node_features, adjacency)

    # 可视化
    visualize_molecules(
        node_features, adjacency, num_samples=4, save_path="gnf_molecules.png"
    )

    print("\n完成！")
    print("生成的分子图保存在 gnf_molecules.png")


if __name__ == "__main__":
    main()
