"""
MolecularRNN 分子生成
基于循环神经网络 (RNN) 的自回归分子图生成模型，整合基类和示例

参考: Popova et al. (2019) "MolecularRNN: Generating realistic molecules with recurrent neural networks"
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 基础类：循环神经网络 (RNN)
# ==========================================


class BaseRNN(nn.Module, ABC):
    """循环神经网络基础抽象类"""

    def __init__(self):
        super(BaseRNN, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """单步或多步前向传播"""
        pass


# ==========================================
# 模型实现：MolecularRNN
# ==========================================


class MolecularRNN(BaseRNN):
    """
    分子 RNN 生成器
    包含两个层级：
    1. NodeRNN: 预测下一个原子的类型
    2. EdgeRNN: 预测新原子与已有原子之间的键
    """

    def __init__(
        self,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        hidden_dim: int = 128,
    ):
        super(MolecularRNN, self).__init__()

        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.hidden_dim = hidden_dim

        # 1. NodeRNN: 用于生成原子序列
        # 输入：上一个原子的类型 (One-hot)
        self.node_rnn = nn.GRUCell(input_size=num_atom_types, hidden_size=hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_atom_types),
        )

        # 2. EdgeRNN: 用于预测新原子 i 与已有原子 j (j < i) 之间的键
        # 输入：上一步预测的键类型
        self.edge_rnn = nn.GRUCell(input_size=num_bond_types, hidden_size=hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_bond_types),
        )

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """实现抽象类方法 (此处简化为 NodeRNN 的一步)"""
        # x: [Batch, AtomTypes]
        h_next = self.node_rnn(x, h)
        logits = self.node_mlp(h_next)
        return logits, h_next

    def generate(self, max_atoms: int = 20) -> Tuple[List[int], torch.Tensor]:
        """
        自回归生成分子
        1. 生成原子 -> 2. 为该原子预测所有可能的边 -> 3. 重复
        """
        device = next(self.parameters()).device
        
        generated_atoms = []
        # 邻接矩阵 (Long 类型存储键类型索引)
        adj = torch.zeros(max_atoms, max_atoms, dtype=torch.long).to(device)

        # 初始状态
        h_node = torch.zeros(1, self.hidden_dim).to(device)
        # 初始输入 (假设 0 是 Start Token)
        curr_atom_in = torch.zeros(1, self.num_atom_types).to(device)
        curr_atom_in[0, 0] = 1.0 

        for i in range(max_atoms):
            # --- 步骤 1: 生成第 i 个原子 ---
            h_node = self.node_rnn(curr_atom_in, h_node)
            node_logits = self.node_mlp(h_node)
            node_probs = F.softmax(node_logits, dim=-1)
            
            # 采样原子类型
            atom_type = torch.multinomial(node_probs, 1).item()
            generated_atoms.append(atom_type)

            # 如果生成 Stop Token (假设 0 在 i>0 时为 Stop)，则停止
            if atom_type == 0 and i > 0:
                break

            # --- 步骤 2: 为新原子 i 预测与已有原子 j 的键 ---
            if i > 0:
                h_edge = h_node.clone()
                curr_bond_in = torch.zeros(1, self.num_bond_types).to(device)
                
                for j in range(i):
                    h_edge = self.edge_rnn(curr_bond_in, h_edge)
                    edge_logits = self.edge_mlp(h_edge)
                    edge_probs = F.softmax(edge_logits, dim=-1)
                    
                    bond_type = torch.multinomial(edge_probs, 1).item()
                    adj[i, j] = bond_type
                    adj[j, i] = bond_type
                    
                    # 更新 EdgeRNN 输入
                    curr_bond_in = F.one_hot(torch.tensor([bond_type]), self.num_bond_types).float().to(device)

            # 更新 NodeRNN 输入
            curr_atom_in = F.one_hot(torch.tensor([atom_type]), self.num_atom_types).float().to(device)

        # 截断
        num_gen = len(generated_atoms)
        return generated_atoms, adj[:num_gen, :num_gen]


# ==========================================
# 数据集
# ==========================================


class MolecularDataset:
    """分子序列数据集 (模拟自回归训练数据)"""

    def __init__(self, num_samples: int = 1000, max_atoms: int = 20, num_atom_types: int = 5):
        self.num_samples = num_samples
        # 简化版数据：仅包含原子序列
        self.data = torch.randint(0, num_atom_types, (num_samples, max_atoms))

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self) -> int:
        return self.num_samples


# ==========================================
# 训练和可视化函数
# ==========================================


def train_mol_rnn(model, dataloader, num_epochs: int = 50, lr: float = 1e-3):
    """训练 MolecularRNN (Teacher Forcing)"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_seq in dataloader:
            # x_seq: [Batch, MaxAtoms]
            batch_size = x_seq.size(0)
            h = torch.zeros(batch_size, model.hidden_dim).to(x_seq.device)
            
            loss = 0
            # Teacher Forcing 训练 NodeRNN
            for t in range(x_seq.size(1) - 1):
                curr_in = F.one_hot(x_seq[:, t], model.num_atom_types).float()
                target = x_seq[:, t+1]
                
                logits, h = model(curr_in, h)
                loss += F.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / x_seq.size(1)

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
    return losses


def visualize_molecules(atoms, adj, save_path="mol_rnn_molecules.png"):
    """可视化生成的分子图"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        # 创建图
        G = nx.from_numpy_array((adj.numpy() > 0.5).astype(float))
        pos = nx.spring_layout(G)
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color="skyblue", node_size=400)
        ax.set_title(f"RNN Generated Molecule")
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
    print("MolecularRNN 分子生成模型训练与生成")
    print("=" * 60)

    # 参数设置
    num_atom_types = 5
    num_bond_types = 4
    hidden_dim = 128
    batch_size = 32
    num_epochs = 50
    lr = 1e-3

    # 1. 创建模型
    model = MolecularRNN(num_atom_types, num_bond_types, hidden_dim)

    # 2. 创建数据
    dataset = MolecularDataset(500, 20, num_atom_types)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 训练
    losses = train_mol_rnn(model, dataloader, num_epochs, lr)

    # 4. 生成与可视化
    with torch.no_grad():
        model.eval()
        gen_atoms, gen_adj = model.generate(max_atoms=15)
        print(f"Generated {len(gen_atoms)} atoms.")
        print(f"Atom sequence: {gen_atoms}")
        # 由于可视化需要 4 个样本，这里多次生成或简化展示
        visualize_molecules(gen_atoms, gen_adj)

    # 5. 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("MolecularRNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("mol_rnn_training_curve.png")
    plt.close()

    print("\n完成！结果已保存。")


if __name__ == "__main__":
    main()
