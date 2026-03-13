"""
GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation
基于几何扩散模型的 3D 分子构象生成
"""

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 基础类：等变图神经网络 (EGNN)
# ==========================================

def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """
    质心零化 (Zero Center of Mass)
    在处理 3D 坐标扩散时，必须确保坐标系统的质心被始终集中在零点，
    否则平移变化会导致扩散过程不稳定。
    x: [batch_size, num_atoms, 3] 或 [num_atoms, 3]
    """
    if x.dim() == 3:
        mean = torch.mean(x, dim=1, keepdim=True)
        return x - mean
    elif x.dim() == 2:
        mean = torch.mean(x, dim=0, keepdim=True)
        return x - mean
    return x

class EGNNLayer(nn.Module):
    """
    简化的等变图神经网络层 (EGNN Layer)
    保持 E(3) / SE(3) 等变性，用于更新节点特征和坐标
    """
    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super(EGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 消息网络：计算节点间的消息
        # 输入: h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 坐标更新网络：计算坐标的更新权重
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 节点更新网络：更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: 节点特征 [batch_size * num_atoms, hidden_dim]
        x: 节点坐标 [batch_size * num_atoms, 3]
        edge_index: 边索引 [2, num_edges]
        """
        row, col = edge_index
        
        # 计算相对距离平方
        coord_diff = x[row] - x[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        
        # 拼接特征计算消息
        if edge_attr is not None:
            edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=1)
        else:
            edge_input = torch.cat([h[row], h[col], radial], dim=1)
            
        m_ij = self.edge_mlp(edge_input)
        
        # 坐标更新 (等变性体现：通过相对坐标的线性组合更新)
        # x_i' = x_i + \sum_j (x_i - x_j) * phi(m_ij)
        coord_weight = self.coord_mlp(m_ij)
        coord_update = coord_diff * coord_weight
        
        # 聚合坐标更新
        x_new = x.clone()
        x_new.index_add_(0, row, coord_update)
        
        # 聚合消息更新节点特征
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, m_ij)
        
        node_input = torch.cat([h, m_i], dim=1)
        h_new = h + self.node_mlp(node_input)
        
        return h_new, x_new

class EGNN(nn.Module):
    """
    等变图神经网络模型
    用于在扩散过程中预测坐标的噪声
    """
    def __init__(self, in_node_dim: int, hidden_dim: int, out_node_dim: int, num_layers: int = 4):
        super(EGNN, self).__init__()
        self.node_embedding = nn.Linear(in_node_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.node_out = nn.Linear(hidden_dim, out_node_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        h: [N, in_node_dim]
        x: [N, 3]
        edge_index: [2, E]
        batch_index: [N]
        返回预测的坐标噪声 [N, 3]
        """
        h = self.node_embedding(h)
        
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
            
        # 预测的噪声即为坐标的更新量 (简化处理)
        # 实际上，EGNN 可以直接输出等变的向量作为噪声预测
        # 这里我们用最后一层的坐标变化作为噪声预测
        # 为了保证平移不变性，我们需要对预测的噪声进行质心零化
        
        # 重新计算一次坐标更新作为输出
        # 或者直接使用 x 作为输出，但需要减去输入 x
        # 这里我们简化，直接返回 x，在外部计算 x_pred - x_input
        return x

# ==========================================
# 模型：GeoDiff (几何扩散模型)
# ==========================================

class GeoDiff(nn.Module):
    """
    GeoDiff: 3D 分子构象生成的扩散模型
    给定分子拓扑 (图结构)，生成稳定的 3D 笛卡尔坐标
    """
    def __init__(self, num_atoms: int, node_feat_dim: int, hidden_dim: int = 64, num_timesteps: int = 100):
        super(GeoDiff, self).__init__()
        self.num_atoms = num_atoms
        self.num_timesteps = num_timesteps
        
        # 扩散过程参数 (DDPM)
        # beta_t 线性增加
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        # 噪声预测网络 (EGNN)
        # 输入节点特征包括原始特征和时间步嵌入
        self.time_emb_dim = 32
        self.egnn = EGNN(
            in_node_dim=node_feat_dim + self.time_emb_dim, 
            hidden_dim=hidden_dim, 
            out_node_dim=node_feat_dim
        )

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """正弦时间步嵌入"""
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程：向真实坐标添加高斯噪声
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
            # 质心零化
            noise = remove_mean(noise)
            
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        # 确保 xt 质心为 0
        xt = remove_mean(xt)
        return xt, noise

    def forward(self, h: torch.Tensor, x0: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        训练前向传播：计算噪声预测损失
        h: [batch_size, num_atoms, feat_dim]
        x0: [batch_size, num_atoms, 3]
        adjacency: [batch_size, num_atoms, num_atoms]
        """
        batch_size = x0.size(0)
        device = x0.device
        
        # 1. 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # 2. 前向扩散加噪
        xt, noise = self.forward_diffusion(x0, t)
        
        # 3. 准备 EGNN 输入 (展平 batch)
        # 构建 edge_index
        edge_indices = []
        for b in range(batch_size):
            adj = adjacency[b]
            edges = torch.nonzero(adj, as_tuple=False)
            edges = edges + b * self.num_atoms # 偏移节点索引
            edge_indices.append(edges)
        
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=0).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        # 展平节点特征和坐标
        h_flat = h.view(batch_size * self.num_atoms, -1)
        xt_flat = xt.view(batch_size * self.num_atoms, 3)
        
        # 添加时间嵌入
        t_emb = self.get_time_embedding(t) # [batch_size, time_emb_dim]
        t_emb_flat = t_emb.repeat_interleave(self.num_atoms, dim=0)
        h_input = torch.cat([h_flat, t_emb_flat], dim=-1)
        
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(self.num_atoms)
        
        # 4. 预测噪声
        # EGNN 输出更新后的坐标，我们用它减去输入坐标作为预测的噪声 (等变向量)
        xt_pred_flat = self.egnn(h_input, xt_flat, edge_index, batch_index)
        noise_pred_flat = xt_pred_flat - xt_flat
        
        noise_pred = noise_pred_flat.view(batch_size, self.num_atoms, 3)
        
        # 质心零化预测的噪声
        noise_pred = remove_mean(noise_pred)
        
        # 5. 计算 MSE 损失
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        逆向去噪采样：从纯噪声生成 3D 坐标
        """
        batch_size = h.size(0)
        device = h.device
        
        # 1. 初始化纯噪声坐标
        xt = torch.randn(batch_size, self.num_atoms, 3, device=device)
        xt = remove_mean(xt)
        
        # 构建 edge_index
        edge_indices = []
        for b in range(batch_size):
            adj = adjacency[b]
            edges = torch.nonzero(adj, as_tuple=False)
            edges = edges + b * self.num_atoms
            edge_indices.append(edges)
            
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=0).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        h_flat = h.view(batch_size * self.num_atoms, -1)
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(self.num_atoms)
        
        # 2. 逐步去噪
        for t_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            
            # 预测噪声
            xt_flat = xt.view(batch_size * self.num_atoms, 3)
            t_emb = self.get_time_embedding(t)
            t_emb_flat = t_emb.repeat_interleave(self.num_atoms, dim=0)
            h_input = torch.cat([h_flat, t_emb_flat], dim=-1)
            
            xt_pred_flat = self.egnn(h_input, xt_flat, edge_index, batch_index)
            noise_pred_flat = xt_pred_flat - xt_flat
            noise_pred = noise_pred_flat.view(batch_size, self.num_atoms, 3)
            noise_pred = remove_mean(noise_pred)
            
            # 计算去噪后的坐标 x_{t-1}
            alpha_t = (1.0 - self.betas[t_step]).view(-1, 1, 1)
            alpha_bar_t = self.alphas_cumprod[t_step].view(-1, 1, 1)
            
            # DDPM 采样公式
            if t_step > 0:
                noise = torch.randn_like(xt)
                noise = remove_mean(noise)
                sigma_t = torch.sqrt(self.betas[t_step]).view(-1, 1, 1)
            else:
                noise = torch.zeros_like(xt)
                sigma_t = 0.0
                
            xt = (1 / torch.sqrt(alpha_t)) * (xt - (self.betas[t_step] / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + sigma_t * noise
            xt = remove_mean(xt)
            
        return xt

# ==========================================
# 数据集：简单的 3D 构象数据集
# ==========================================

class Simple3DConformationDataset:
    """
    简单的 3D 构象数据集
    生成随机的分子图拓扑和对应的 3D 坐标
    """
    def __init__(self, num_samples: int = 1000, num_atoms: int = 10, feature_dim: int = 16):
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim
        
        # 节点特征 (例如原子类型的一热编码)
        self.node_features = torch.randn(num_samples, num_atoms, feature_dim)
        
        # 邻接矩阵 (图拓扑)
        self.adjacency = torch.zeros(num_samples, num_atoms, num_atoms)
        
        # 3D 坐标
        self.coordinates = torch.zeros(num_samples, num_atoms, 3)
        
        for i in range(num_samples):
            # 生成连通图
            for j in range(num_atoms - 1):
                self.adjacency[i, j, j+1] = 1.0
                self.adjacency[i, j+1, j] = 1.0
                
            # 随机添加一些边
            for j in range(num_atoms):
                for k in range(j + 2, num_atoms):
                    if torch.rand(1).item() < 0.2:
                        self.adjacency[i, j, k] = 1.0
                        self.adjacency[i, k, j] = 1.0
                        
            # 生成对应的 3D 坐标 (模拟某种稳定构象)
            # 这里我们用一个简单的弹簧模型模拟：相连的节点距离接近 1.0
            pos = torch.randn(num_atoms, 3)
            # 简单的迭代优化使其符合图结构
            for _ in range(50):
                for j in range(num_atoms):
                    for k in range(num_atoms):
                        if self.adjacency[i, j, k] == 1.0:
                            diff = pos[j] - pos[k]
                            dist = torch.norm(diff) + 1e-5
                            force = (dist - 1.0) * (diff / dist) * 0.1
                            pos[j] -= force
                            pos[k] += force
            
            # 质心零化
            pos = remove_mean(pos.unsqueeze(0)).squeeze(0)
            self.coordinates[i] = pos

    def __getitem__(self, idx: int):
        return self.node_features[idx], self.coordinates[idx], self.adjacency[idx]

    def __len__(self) -> int:
        return self.num_samples

# ==========================================
# 训练和可视化函数
# ==========================================

def train_geodiff(model: GeoDiff, dataloader, num_epochs: int = 50, lr: float = 1e-3):
    """训练 GeoDiff 模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (h, x0, adj) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 确保输入坐标质心为 0
            x0 = remove_mean(x0)
            
            loss = model(h, x0, adj)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
    # 绘制训练曲线
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("GeoDiff Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("geodiff_training_curve.png")
    plt.close()
    print("Training curve saved to geodiff_training_curve.png")
    
    return losses

def visualize_conformations(h: torch.Tensor, adj: torch.Tensor, coords: torch.Tensor, save_path: str = "geodiff_conformations.png"):
    """
    可视化生成的 3D 分子构象
    使用 matplotlib 的 3D 散点图和连线
    """
    num_samples = min(4, h.size(0))
    fig = plt.figure(figsize=(12, 12))
    
    for i in range(num_samples):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_title(f"Generated Conformation {i + 1}")
        
        pos = coords[i].detach().cpu().numpy()
        adjacency = adj[i].detach().cpu().numpy()
        num_atoms = pos.shape[0]
        
        # 绘制节点
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='lightblue', s=100, edgecolors='k')
        
        # 绘制边
        for j in range(num_atoms):
            for k in range(j + 1, num_atoms):
                if adjacency[j, k] > 0.5:
                    ax.plot(
                        [pos[j, 0], pos[k, 0]],
                        [pos[j, 1], pos[k, 1]],
                        [pos[j, 2], pos[k, 2]],
                        c='gray', alpha=0.7
                    )
                    
        # 设置坐标轴范围
        max_range = np.array([pos[:, 0].max()-pos[:, 0].min(), pos[:, 1].max()-pos[:, 1].min(), pos[:, 2].max()-pos[:, 2].min()]).max() / 2.0
        mid_x = (pos[:, 0].max()+pos[:, 0].min()) * 0.5
        mid_y = (pos[:, 1].max()+pos[:, 1].min()) * 0.5
        mid_z = (pos[:, 2].max()+pos[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")

# ==========================================
# 主程序
# ==========================================

def main():
    print("=" * 60)
    print("GeoDiff 3D 分子构象生成")
    print("=" * 60)
    
    # 参数设置
    num_atoms = 10
    feature_dim = 16
    hidden_dim = 64
    batch_size = 32
    num_epochs = 50
    num_timesteps = 100
    
    print("\n配置:")
    print(f"  原子数量: {num_atoms}")
    print(f"  特征维度: {feature_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  扩散步数: {num_timesteps}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    
    # 创建模型
    print("\n创建模型...")
    model = GeoDiff(num_atoms=num_atoms, node_feat_dim=feature_dim, hidden_dim=hidden_dim, num_timesteps=num_timesteps)
    
    # 创建数据集
    print("创建数据集...")
    dataset = Simple3DConformationDataset(num_samples=500, num_atoms=num_atoms, feature_dim=feature_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练模型
    print("\n开始训练...")
    train_geodiff(model, dataloader, num_epochs=num_epochs)
    
    # 生成新构象
    print("\n生成新构象...")
    model.eval()
    # 取一个 batch 的图拓扑作为条件
    h, _, adj = next(iter(dataloader))
    h = h[:4]
    adj = adj[:4]
    
    generated_coords = model.sample(h, adj)
    
    # 可视化
    visualize_conformations(h, adj, generated_coords, save_path="geodiff_conformations.png")
    
    print("\n完成！")
    print("生成的分子构象保存在 geodiff_conformations.png")

if __name__ == "__main__":
    main()
