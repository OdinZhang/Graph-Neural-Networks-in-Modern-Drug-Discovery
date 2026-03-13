import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 基础组件与等变图神经网络 (EGNN)
# ==========================================


class EGNNLayer(nn.Module):
    """
    等变图神经网络层 (Equivariant Graph Neural Network Layer)

    该层保证了节点特征的平移不变性 (Translation Invariance) 和
    坐标的平移/旋转等变性 (Translation/Rotation Equivariance)。
    在 EDM 中，这对于处理 3D 分子结构至关重要。
    """

    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index

        # 计算相对坐标和距离平方
        coord_diff = x[row] - x[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)

        # 拼接边特征
        edge_inputs = [h[row], h[col], radial]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
        edge_feat = torch.cat(edge_inputs, dim=1)

        # 边消息传递
        m_ij = self.edge_mlp(edge_feat)

        # 坐标更新 (等变性)
        # x_i^(l+1) = x_i^l + sum_j (x_i - x_j) * phi_x(m_ij)
        coord_weight = self.coord_mlp(m_ij)
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_diff * coord_weight)
        x_new = x + x_update / (h.size(0) - 1)  # 简单的归一化

        # 节点特征更新 (不变性)
        # h_i^(l+1) = phi_h(h_i^l, sum_j m_ij)
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, m_ij)
        node_inputs = torch.cat([h, m_i], dim=1)
        h_new = h + self.node_mlp(node_inputs)

        return h_new, x_new


class EGNN(nn.Module):
    def __init__(self, in_node_dim, hidden_dim, out_node_dim, num_layers=4):
        super().__init__()
        self.node_emb = nn.Linear(in_node_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.node_out = nn.Linear(hidden_dim, out_node_dim)

    def forward(self, h, x, edge_index, t_emb):
        # t_emb 可以加到节点特征中
        h = self.node_emb(h) + t_emb
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        h_out = self.node_out(h)
        return h_out, x


# ==========================================
# 2. 等变扩散模型 (EDM)
# ==========================================


class EDM(nn.Module):
    """
    等变扩散模型 (Equivariant Diffusion Model)

    该模型执行联合扩散过程 (Joint Diffusion Process)：
    1. 连续坐标 (3D Coordinates): 使用高斯扩散 (Gaussian Diffusion)。
    2. 类别型原子类型 (Categorical Atom Types): 在连续空间中作为 one-hot 向量进行高斯扩散，
       或者使用离散扩散。这里为了简化和统一，我们将原子类型视为连续特征进行高斯扩散，
       并在最后一步通过 softmax 还原为类别。
    """

    def __init__(self, num_atom_types, hidden_dim=64, num_layers=4, timesteps=100):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.timesteps = timesteps

        # 时间步嵌入
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # 核心网络：预测噪声
        self.egnn = EGNN(
            in_node_dim=num_atom_types,
            hidden_dim=hidden_dim,
            out_node_dim=num_atom_types,
            num_layers=num_layers,
        )

        # 扩散过程参数 (线性 beta schedule)
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def remove_mean_batch(self, x, batch):
        # 移除质心，保证平移不变性
        # 简化版本：假设只有一个图在处理，或者 batch 是全 0
        mean = x.mean(dim=0, keepdim=True)
        return x - mean

    def forward(self, h, x, edge_index, t):
        # t: [batch_size]
        t_emb = self.time_emb(t.float().unsqueeze(-1) / self.timesteps)
        # 扩展 t_emb 到每个节点
        t_emb = t_emb.repeat(h.size(0), 1)

        # 预测噪声
        eps_h, eps_x = self.egnn(h, x, edge_index, t_emb)

        # 移除坐标噪声的质心
        eps_x = self.remove_mean_batch(eps_x, None)

        return eps_h, eps_x

    def compute_loss(self, h_0, x_0, edge_index):
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (1,), device=h_0.device).long()

        # 采样噪声
        noise_h = torch.randn_like(h_0)
        noise_x = torch.randn_like(x_0)
        noise_x = self.remove_mean_batch(noise_x, None)

        # 加噪过程 (Forward Diffusion)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        h_t = sqrt_alpha_bar * h_0 + sqrt_one_minus_alpha_bar * noise_h
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise_x

        # 预测噪声
        pred_noise_h, pred_noise_x = self(h_t, x_t, edge_index, t)

        # 计算损失 (MSE)
        loss_h = F.mse_loss(pred_noise_h, noise_h)
        loss_x = F.mse_loss(pred_noise_x, noise_x)

        return loss_h + loss_x

    @torch.no_grad()
    def sample(self, num_nodes, device):
        # 从纯噪声开始采样
        h = torch.randn((num_nodes, self.num_atom_types), device=device)
        x = torch.randn((num_nodes, 3), device=device)
        x = self.remove_mean_batch(x, None)

        # 全连接图的边索引
        adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        edge_index = torch.nonzero(adj).t().to(device)

        # 反向扩散过程 (Reverse Diffusion)
        for i in reversed(range(self.timesteps)):
            t = torch.tensor([i], device=device).long()

            pred_noise_h, pred_noise_x = self(h, x, edge_index, t)

            alpha = 1.0 - self.betas[i]
            alpha_bar = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise_h = torch.randn_like(h)
                noise_x = torch.randn_like(x)
                noise_x = self.remove_mean_batch(noise_x, None)
            else:
                noise_h = torch.zeros_like(h)
                noise_x = torch.zeros_like(x)

            # 更新 h 和 x
            h = (1 / torch.sqrt(alpha)) * (
                h - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise_h
            ) + torch.sqrt(beta) * noise_h
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise_x
            ) + torch.sqrt(beta) * noise_x
            x = self.remove_mean_batch(x, None)

        # 将连续的 h 转换为离散的原子类型
        atom_types = torch.argmax(h, dim=-1)
        return atom_types, x


# ==========================================
# 3. 数据集与模拟数据
# ==========================================


class Simple3DMoleculeDataset(Dataset):
    """
    轻量级模拟 3D 分子数据集
    生成具有随机 3D 坐标和原子类型的全连接图。
    """

    def __init__(self, num_samples=100, num_nodes=5, num_atom_types=4):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_atom_types = num_atom_types

        self.data = []
        for _ in range(num_samples):
            # 随机原子类型 (one-hot)
            atom_types = torch.randint(0, num_atom_types, (num_nodes,))
            h = F.one_hot(atom_types, num_classes=num_atom_types).float()

            # 随机 3D 坐标 (均值为 0)
            x = torch.randn(num_nodes, 3)
            x = x - x.mean(dim=0, keepdim=True)

            # 全连接边
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = torch.nonzero(adj).t()

            self.data.append((h, x, edge_index))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 4. 训练流程
# ==========================================


def train_edm(model, dataloader, epochs=50, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for h, x, edge_index in dataloader:
            h, x, edge_index = (
                h[0].to(device),
                x[0].to(device),
                edge_index[0].to(device),
            )

            optimizer.zero_grad()
            loss = model.compute_loss(h, x, edge_index)
            loss.backward()
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    return loss_history


# ==========================================
# 5. 可视化与主程序
# ==========================================


def visualize_3d_molecules(atom_types, coords, filename="edm_molecules.png"):
    """
    可视化生成的 3D 分子 (投影到 2D 平面展示)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    coords = coords.cpu().numpy()
    atom_types = atom_types.cpu().numpy()

    # 简单的颜色映射
    colors = ["red", "blue", "green", "orange", "purple"]

    # 绘制节点
    for i in range(len(coords)):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            coords[i, 2],
            c=colors[atom_types[i] % len(colors)],
            s=100,
            label=f"Atom {atom_types[i]}",
        )

    # 绘制边 (基于距离阈值)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if dist_matrix[i, j] < 2.0:  # 简单的成键阈值
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    "k-",
                    alpha=0.5,
                )

    ax.set_title("Generated 3D Molecule Graph via EDM")
    plt.savefig(filename)
    plt.close()


def main():
    print("=" * 60)
    print("EDM 3D 分子生成模型")
    print("=" * 60)

    # 参数设置
    num_nodes = 6
    num_atom_types = 4
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n配置:")
    print(f"  节点数量: {num_nodes}")
    print(f"  原子类型数: {num_atom_types}")
    print(f"  训练轮数: {epochs}")
    print(f"  设备: {device}")

    print("\n创建模型...")
    model = EDM(
        num_atom_types=num_atom_types, hidden_dim=64, num_layers=4, timesteps=100
    )

    print("创建数据集...")
    dataset = Simple3DMoleculeDataset(
        num_samples=100, num_nodes=num_nodes, num_atom_types=num_atom_types
    )
    # batch_size=1 简化实现，避免复杂的 batch 图拼接
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("\n开始训练...")
    loss_history = train_edm(model, dataloader, epochs=epochs, lr=1e-3, device=device)

    # 绘制训练曲线
    plt.figure()
    plt.plot(loss_history)
    plt.title("EDM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("edm_training_curve.png")
    plt.close()
    print("训练曲线已保存至 edm_training_curve.png")

    print("\n生成新分子并可视化...")
    model.eval()
    gen_atom_types, gen_coords = model.sample(num_nodes=num_nodes, device=device)

    # 可视化生成结果
    visualize_3d_molecules(gen_atom_types, gen_coords, filename="edm_molecules.png")
    print("生成的分子已保存至 edm_molecules.png")

    print("\n完成！结果已保存。")


if __name__ == "__main__":
    main()
