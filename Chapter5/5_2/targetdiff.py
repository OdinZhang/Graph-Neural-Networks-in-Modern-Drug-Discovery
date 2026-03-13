import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 2. 基础网络组件 (Base Classes & EGNN)
# ==========================================
class EGNNLayer(nn.Module):
    """
    等变图神经网络层 (Equivariant Graph Neural Network Layer)。
    用于更新节点特征和坐标，保持 E(3) 等变性。
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        row, col = edge_index

        # 计算相对距离的平方
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)

        # 边特征
        edge_inputs = [h[row], h[col], radial]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
        edge_feat = torch.cat(edge_inputs, dim=1)
        m_ij = self.edge_mlp(edge_feat)

        # 坐标更新 (等变)
        coord_weight = self.coord_mlp(m_ij)
        # 归一化坐标差，防止梯度爆炸
        norm = torch.sqrt(radial + 1e-8)
        coord_update = (coord_diff / norm) * coord_weight

        # 聚合边信息到节点
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, m_ij)

        # 节点特征更新 (不变)
        node_inputs = torch.cat([h, m_i], dim=1)
        h_new = h + self.node_mlp(node_inputs)

        # 聚合坐标更新
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, row, coord_update)
        pos_new = pos + pos_update / (
            torch.bincount(row, minlength=pos.size(0)).view(-1, 1).float() + 1e-6
        )

        return h_new, pos_new


class ConditionalEGNN(nn.Module):
    """
    条件等变图神经网络 (Conditional EGNN)。
    作为扩散模型的 Score Network。
    将靶点蛋白口袋的 3D 几何与特征作为条件约束 (Condition)。

    在 TargetDiff 中，蛋白质约束通常通过以下方式引入：
    1. 二分图 (Bipartite Graph): 构建配体原子和口袋原子之间的边，进行消息传递。
    2. k-NN 图: 为每个配体原子寻找最近的 k 个口袋原子建立连接。
    3. 交叉注意力 (Cross-Attention): 配体节点作为 Query，口袋节点作为 Key 和 Value。

    为了简化和避免内存泄漏，本实现采用将配体和口袋节点拼接构建联合图 (Joint Graph) 的方式，
    在同一个 batch 内构建全连接图，从而隐式地实现了配体与口袋之间的信息交互。
    """

    def __init__(self, in_node_dim, hidden_dim, out_node_dim, n_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(in_node_dim, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_node_dim),
        )

    def forward(self, l_pos, l_feat, p_pos, p_feat, t, l_batch, p_batch):
        """
        l_pos: 配体坐标 [N_l, 3]
        l_feat: 配体特征 [N_l, D_l]
        p_pos: 口袋坐标 [N_p, 3]
        p_feat: 口袋特征 [N_p, D_p]
        t: 时间步 [B, 1]
        l_batch: 配体 batch 索引 [N_l]
        p_batch: 口袋 batch 索引 [N_p]
        """
        # 1. 节点特征嵌入
        N_l = l_pos.size(0)
        N_p = p_pos.size(0)

        joint_pos = torch.cat([l_pos, p_pos], dim=0)
        joint_feat = torch.cat([l_feat, p_feat], dim=0)
        joint_batch = torch.cat([l_batch, p_batch], dim=0)

        # 节点嵌入
        h = self.node_emb(joint_feat)

        # 时间嵌入
        t_emb = self.time_emb(t)  # [B, hidden_dim]
        t_emb_joint = t_emb[joint_batch]  # [N_l + N_p, hidden_dim]
        h = h + t_emb_joint

        # 2. 构建图边 (全连接图，仅在同一个 batch 内)
        batch_size = t.size(0)
        row_list, col_list = [], []
        for b in range(batch_size):
            nodes_in_b = (joint_batch == b).nonzero(as_tuple=True)[0]
            n_nodes = nodes_in_b.size(0)
            # 全连接
            r = nodes_in_b.repeat_interleave(n_nodes)
            c = nodes_in_b.repeat(n_nodes)
            # 移除自环
            mask = r != c
            row_list.append(r[mask])
            col_list.append(c[mask])

        edge_index = torch.stack([torch.cat(row_list), torch.cat(col_list)], dim=0)

        # 3. 消息传递
        for layer in self.layers:
            h, joint_pos = layer(h, joint_pos, edge_index)

        # 4. 提取配体的更新
        # 我们只关心配体的去噪，口袋是条件，保持不变
        l_h_out = h[:N_l]
        l_pos_out = joint_pos[:N_l]

        # 预测特征的噪声/得分
        l_feat_out = self.out_mlp(l_h_out)

        # 预测坐标的位移 (作为噪声)
        l_pos_noise = l_pos_out - l_pos

        return l_pos_noise, l_feat_out


# ==========================================
# 3. 扩散模型 (Conditional Diffusion Model)
# ==========================================
class TargetDiff(nn.Module):
    """
    简化的 TargetDiff 模型。
    基于结构的分子生成 (SBMG)。
    """

    def __init__(
        self, num_atom_types, pocket_feat_dim, hidden_dim=64, n_layers=3, timesteps=100
    ):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.timesteps = timesteps

        # 统一特征维度
        self.l_proj = nn.Linear(num_atom_types, hidden_dim)
        self.p_proj = nn.Linear(pocket_feat_dim, hidden_dim)

        # Score Network
        self.eps_net = ConditionalEGNN(
            in_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_node_dim=num_atom_types,
            n_layers=n_layers,
        )

        # 扩散过程参数 (线性 schedule)
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def remove_mean_batch(self, x, batch):
        """质心零化 (Zero Center of Mass)"""
        # 计算每个 batch 的质心
        batch_size = batch.max().item() + 1
        mean = torch.zeros(batch_size, 3, device=x.device)
        mean.index_add_(0, batch, x)
        count = torch.bincount(batch, minlength=batch_size).view(-1, 1).float()
        mean = mean / count.clamp(min=1)
        # 减去质心
        return x - mean[batch]

    def forward(self, batch_data):
        """
        训练前向传播，计算损失。
        """
        l_pos = batch_data["ligand_pos"]
        l_type = batch_data["ligand_type"]
        p_pos = batch_data["pocket_pos"]
        p_feat = batch_data["pocket_feat"]
        l_batch = batch_data["ligand_batch"]
        p_batch = batch_data["pocket_batch"]

        B = l_batch.max().item() + 1
        device = l_pos.device

        # 1. 随机采样时间步
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        t_node = t[l_batch]

        # 2. 添加噪声 (前向扩散)
        # 连续特征 (坐标) 的高斯扩散
        noise_pos = torch.randn_like(l_pos)
        noise_pos = self.remove_mean_batch(noise_pos, l_batch)  # 噪声也需要质心零化

        a_bar = self.alpha_bar[t_node].unsqueeze(1)  # type: ignore
        noisy_l_pos = torch.sqrt(a_bar) * l_pos + torch.sqrt(1 - a_bar) * noise_pos
        noisy_l_pos = self.remove_mean_batch(noisy_l_pos, l_batch)

        # 离散特征 (原子类型) 的扩散 (这里简化为连续高斯扩散，实际 EDM 中使用 Categorical Diffusion)
        noise_type = torch.randn_like(l_type)
        noisy_l_type = torch.sqrt(a_bar) * l_type + torch.sqrt(1 - a_bar) * noise_type

        # 3. 预测噪声
        # 投影特征
        l_feat_in = self.l_proj(noisy_l_type)
        p_feat_in = self.p_proj(p_feat)

        t_norm = t.float().unsqueeze(1) / self.timesteps

        pred_pos_noise, pred_type_noise = self.eps_net(
            noisy_l_pos, l_feat_in, p_pos, p_feat_in, t_norm, l_batch, p_batch
        )

        # 4. 计算损失
        # 坐标 MSE 损失
        loss_pos = F.mse_loss(pred_pos_noise, noise_pos)
        # 类型 MSE 损失 (简化版)
        loss_type = F.mse_loss(pred_type_noise, noise_type)

        loss = loss_pos + loss_type
        return loss, loss_pos, loss_type

    @torch.no_grad()
    def sample(self, p_pos, p_feat, n_ligand_atoms, p_batch, l_batch):
        """
        条件生成采样 (逆向去噪过程)。
        给定口袋条件，生成配体。
        """
        device = p_pos.device
        B = p_batch.max().item() + 1
        N_l = n_ligand_atoms.sum().item()

        # 1. 初始化纯噪声
        l_pos = torch.randn(N_l, 3, device=device)
        l_pos = self.remove_mean_batch(l_pos, l_batch)
        l_type = torch.randn(N_l, self.num_atom_types, device=device)

        p_feat_in = self.p_proj(p_feat)

        # 2. 逐步去噪
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = t.float().unsqueeze(1) / self.timesteps

            l_feat_in = self.l_proj(l_type)

            # 预测噪声
            pred_pos_noise, pred_type_noise = self.eps_net(
                l_pos, l_feat_in, p_pos, p_feat_in, t_norm, l_batch, p_batch
            )

            # 更新坐标和类型 (DDPM 采样公式)
            alpha = self.alpha[i]  # type: ignore
            alpha_bar = self.alpha_bar[i]  # type: ignore
            beta = self.beta[i]  # type: ignore

            if i > 0:
                z_pos = torch.randn_like(l_pos)
                z_pos = self.remove_mean_batch(z_pos, l_batch)
                z_type = torch.randn_like(l_type)
            else:
                z_pos = torch.zeros_like(l_pos)
                z_type = torch.zeros_like(l_type)

            # 坐标更新
            l_pos = (1 / torch.sqrt(alpha)) * (
                l_pos - (beta / torch.sqrt(1 - alpha_bar)) * pred_pos_noise
            ) + torch.sqrt(beta) * z_pos
            l_pos = self.remove_mean_batch(l_pos, l_batch)

            # 类型更新
            l_type = (1 / torch.sqrt(alpha)) * (
                l_type - (beta / torch.sqrt(1 - alpha_bar)) * pred_type_noise
            ) + torch.sqrt(beta) * z_type

        # 将连续的类型表示转换为离散的 one-hot
        l_type_discrete = torch.argmax(l_type, dim=-1)

        return l_pos, l_type_discrete


# ==========================================
# 3.5 数据集 (Dataset)
# ==========================================
# ==========================================
# 数据集 (Dataset)
# ==========================================
class SimplePocketMoleculeDataset(Dataset):
    """
    模拟的靶点-配体复合物数据集。
    提供蛋白质口袋 (Protein Pocket) 的上下文坐标和特征，以及配体 (Ligand) 的坐标和原子类型。
    """

    def __init__(
        self,
        num_samples=100,
        max_pocket_atoms=20,
        max_ligand_atoms=10,
        num_atom_types=4,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_pocket_atoms = max_pocket_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.num_atom_types = num_atom_types

        # 模拟数据生成
        self.data = []
        for _ in range(num_samples):
            # 随机生成口袋原子数量和配体原子数量
            n_p = np.random.randint(10, max_pocket_atoms + 1)
            n_l = np.random.randint(5, max_ligand_atoms + 1)

            # 口袋坐标 (固定在某个区域，例如半径为 10 的球壳上)
            theta = np.random.uniform(0, 2 * np.pi, n_p)
            phi = np.random.uniform(0, np.pi, n_p)
            r = np.random.uniform(8, 12, n_p)
            p_pos = np.stack(
                [
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi),
                ],
                axis=1,
            )
            p_pos = torch.tensor(p_pos, dtype=torch.float32)

            # 口袋特征 (简单起见，使用 one-hot 或随机特征)
            p_feat = torch.randn(n_p, 8)

            # 配体坐标 (在口袋中心附近，即原点附近)
            l_pos = torch.randn(n_l, 3) * 2.0
            # 质心零化 (Zero Center of Mass) - 仅对配体进行，因为口袋是固定的条件
            l_pos = l_pos - l_pos.mean(dim=0, keepdim=True)

            # 配体原子类型 (离散特征)
            l_type = torch.randint(0, num_atom_types, (n_l,))
            l_type_onehot = F.one_hot(l_type, num_classes=num_atom_types).float()

            self.data.append(
                {
                    "pocket_pos": p_pos,
                    "pocket_feat": p_feat,
                    "ligand_pos": l_pos,
                    "ligand_type": l_type_onehot,
                    "n_pocket": n_p,
                    "n_ligand": n_l,
                }
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    简单的批处理函数，将不同大小的图拼接成大图 (Batching by concatenation)。
    """
    batch_p_pos = []
    batch_p_feat = []
    batch_l_pos = []
    batch_l_type = []
    batch_p_batch = []
    batch_l_batch = []

    for i, item in enumerate(batch):
        batch_p_pos.append(item["pocket_pos"])
        batch_p_feat.append(item["pocket_feat"])
        batch_l_pos.append(item["ligand_pos"])
        batch_l_type.append(item["ligand_type"])
        batch_p_batch.append(torch.full((item["n_pocket"],), i, dtype=torch.long))
        batch_l_batch.append(torch.full((item["n_ligand"],), i, dtype=torch.long))

    return {
        "pocket_pos": torch.cat(batch_p_pos, dim=0),
        "pocket_feat": torch.cat(batch_p_feat, dim=0),
        "ligand_pos": torch.cat(batch_l_pos, dim=0),
        "ligand_type": torch.cat(batch_l_type, dim=0),
        "pocket_batch": torch.cat(batch_p_batch, dim=0),
        "ligand_batch": torch.cat(batch_l_batch, dim=0),
    }


# ==========================================
# 4. 训练与可视化 (Training & Visualization)
# ==========================================
def train_targetdiff(model, dataloader, num_epochs=50, lr=1e-3, device=None):
    """
    训练 TargetDiff 模型的主循环。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # 将数据移至设备
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss, loss_pos, loss_type = model(batch)
            loss.backward()
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # type: ignore
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return loss_history


def visualize_pocket_generation(model, dataset):
    """
    可视化条件生成结果：在给定的口袋中生成配体。
    """
    print("开始生成并可视化分子...")
    model.eval()
    device = next(model.parameters()).device

    # 取一个样本作为条件
    sample = dataset[0]
    p_pos = sample["pocket_pos"].to(device)
    p_feat = sample["pocket_feat"].to(device)
    n_p = sample["n_pocket"]

    # 假设我们要生成一个包含 8 个原子的配体
    n_l = 8
    n_ligand_atoms = torch.tensor([n_l], device=device)

    p_batch = torch.zeros(n_p, dtype=torch.long, device=device)
    l_batch = torch.zeros(n_l, dtype=torch.long, device=device)

    # 采样
    gen_l_pos, gen_l_type = model.sample(
        p_pos, p_feat, n_ligand_atoms, p_batch, l_batch
    )

    # 转换为 numpy
    p_pos_np = p_pos.cpu().numpy()
    gen_l_pos_np = gen_l_pos.cpu().numpy()
    gen_l_type_np = gen_l_type.cpu().numpy()

    # 绘制 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制口袋 (灰色半透明)
    ax.scatter(
        p_pos_np[:, 0],
        p_pos_np[:, 1],
        p_pos_np[:, 2],  # type: ignore
        c="gray",
        s=100,
        alpha=0.3,
        label="Protein Pocket",
    )

    # 绘制生成的配体 (根据原子类型着色)
    colors = ["red", "blue", "green", "orange"]
    for i in range(n_l):
        c = colors[gen_l_type_np[i] % len(colors)]
        ax.scatter(
            gen_l_pos_np[i, 0],
            gen_l_pos_np[i, 1],
            gen_l_pos_np[i, 2],  # type: ignore
            c=c,
            s=200,
            edgecolors="black",
            label="Ligand Atom" if i == 0 else "",
        )

    # 简单连线 (配体内部，基于距离)
    for i in range(n_l):
        for j in range(i + 1, n_l):
            dist = np.linalg.norm(gen_l_pos_np[i] - gen_l_pos_np[j])
            if dist < 2.0:  # 简单的成键阈值
                ax.plot(
                    [gen_l_pos_np[i, 0], gen_l_pos_np[j, 0]],  # type: ignore
                    [gen_l_pos_np[i, 1], gen_l_pos_np[j, 1]],
                    [gen_l_pos_np[i, 2], gen_l_pos_np[j, 2]],
                    c="black",
                    linewidth=2,
                )

    ax.set_title("Simulated Structure-Based Molecular Generation with TargetDiff:")  # type: ignore
    ax.legend()  # type: ignore
    plt.savefig("targetdiff_binding.png")
    plt.close()
    print("可视化完成，已保存 targetdiff_binding.png")


# ==========================================
# 5. 主程序 (Main)
# ==========================================
def main():
    print("=" * 60)
    print("TargetDiff 基于结构的条件分子生成")
    print("=" * 60)
    print()

    # 确保随机种子固定，以便复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 参数设置
    num_epochs = 50
    batch_size = 16
    lr = 1e-3
    num_atom_types = 4
    pocket_feat_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("配置:")
    print(f"  num_epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print(f"  num_atom_types: {num_atom_types}")
    print(f"  pocket_feat_dim: {pocket_feat_dim}")
    print()

    print("创建模型...")
    model = TargetDiff(
        num_atom_types=num_atom_types, pocket_feat_dim=pocket_feat_dim
    ).to(device)

    print("创建数据集...")
    dataset = SimplePocketMoleculeDataset(
        num_samples=200, num_atom_types=num_atom_types
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print()

    print("开始训练...")
    loss_history = train_targetdiff(
        model, dataloader, num_epochs=num_epochs, lr=lr, device=device
    )

    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TargetDiff Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("targetdiff_training_curve.png")
    plt.close()
    print()

    print("生成新分子并可视化...")
    visualize_pocket_generation(model, dataset)

    print()
    print("完成！结果已保存。")


if __name__ == "__main__":
    main()
