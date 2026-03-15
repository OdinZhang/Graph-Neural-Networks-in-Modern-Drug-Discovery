import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T


# ==========================================
# 1. 图数据的构建
# ==========================================
def load_or_build_graph(data_dir='./data', force_rebuild=False):
    graph_path = os.path.join(data_dir, 'macro_hetero_graph.pt')

    # 如果本地有保存好的图，直接极速加载
    if os.path.exists(graph_path) and not force_rebuild:
        print(f"从本地缓存加载异构图: {graph_path}")
        return torch.load(graph_path)

    print("本地无缓存，正在从 CSV 构建异构图...")
    df_combo = pd.read_csv(
        os.path.join(data_dir, 'demo_synergy_labels.csv'),
        dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str}
    )
    df_drug = pd.read_csv(
        os.path.join(data_dir, 'demo_drug_features.csv'),
        dtype={'ex_drug_id': str}
    )
    df_cell = pd.read_csv(
        os.path.join(data_dir, 'demo_cell_features.csv'),
        dtype={'ex_cell_id': str}
    )

    # 构建 ID 到 索引 的映射字典
    drug_id2idx = {str(d_id): i for i, d_id in enumerate(df_drug['ex_drug_id'])}
    cell_id2idx = {str(c_id): i for i, c_id in enumerate(df_cell['ex_cell_id'])}

    # 提取特征矩阵 (确保转换为 float32 张量)
    drug_str_cols = df_drug.select_dtypes(include=['object']).columns.tolist()
    cell_str_cols = df_cell.select_dtypes(include=['object']).columns.tolist()
    drug_feat = torch.tensor(
        df_drug.drop(columns=['ex_drug_id'] + drug_str_cols).values.astype(float),
        dtype=torch.float32
    )
    cell_feat = torch.tensor(
        df_cell.drop(columns=['ex_cell_id'] + cell_str_cols).values.astype(float),
        dtype=torch.float32
    )

    data = HeteroData()
    data['drug'].x = drug_feat
    data['cell'].x = cell_feat

    # 构建交互边 (为了简化，将药物组合的影响拆解为 Drug -> Cell 的两条边)
    src_nodes, dst_nodes, edge_weights = [], [], []

    for _, row in df_combo.iterrows():
        # 获取矩阵索引
        d_row_idx = drug_id2idx[str(row['drug_row'])]
        d_col_idx = drug_id2idx[str(row['drug_col'])]
        c_idx = cell_id2idx[str(row['cell_line_name'])]
        score = row['synergy_loewe']

        src_nodes.extend([d_row_idx, d_col_idx])
        dst_nodes.extend([c_idx, c_idx])
        edge_weights.extend([score, score])

    data['drug', 'interacts', 'cell'].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    data['drug', 'interacts', 'cell'].edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

    # 强制将图转为无向图，自动生成 ('cell', 'rev_interacts', 'drug') 反向边
    # 使得药物节点就能接收到来自细胞的信息，从而参与网络更新
    data = T.ToUndirected()(data)

    # 保存到本地
    torch.save(data, graph_path)
    print(f"图构建完成并已保存至: {graph_path}")
    return data


# ==========================================
# 2. PyTorch Dataset (用于 DataLoader)
# ==========================================
class SynergyDataset(Dataset):
    """用于喂给 DataLoader 的数据集类，只输出索引和标签"""

    def __init__(self, combo_csv_path, drug_id2idx, cell_id2idx):
        self.df = pd.read_csv(
            combo_csv_path,
            dtype={'drug_row': str, 'drug_col': str, 'cell_line_name': str}
        )
        self.drug_id2idx = drug_id2idx
        self.cell_id2idx = cell_id2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d_row_idx = self.drug_id2idx[str(row['drug_row'])]
        d_col_idx = self.drug_id2idx[str(row['drug_col'])]
        c_idx = self.cell_id2idx[str(row['cell_line_name'])]
        label = row['synergy_loewe']

        # 返回：药物A索引, 药物B索引, 细胞系索引, 真实的协同得分
        return torch.tensor([d_row_idx, d_col_idx, c_idx], dtype=torch.long), torch.tensor(label, dtype=torch.float32)


# ==========================================
# 3. 宏观异构图模型架构
# ==========================================
class MacroGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class SynergyPredictor(nn.Module):
    def __init__(self, drug_dim, cell_dim, hidden_dim, metadata):
        super().__init__()
        # 特征维度对齐层
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.cell_proj = nn.Linear(cell_dim, hidden_dim)

        # 异构图消息传递网络
        self.gnn = to_hetero(MacroGNN(hidden_dim), metadata=metadata)

        # 协同得分预测器 (Concat 后维度为 3 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_data, batch_queries):
        # 1. 维度对齐
        x_dict = {
            'drug': self.drug_proj(graph_data['drug'].x),
            'cell': self.cell_proj(graph_data['cell'].x)
        }

        # 2. 全图信息聚合，更新所有节点表征
        updated_nodes = self.gnn(x_dict, graph_data.edge_index_dict)

        # 3. 根据 DataLoader 传来的 Batch 索引，提取对应特征
        # batch_queries shape: (Batch_Size, 3) -> [drug_a_idx, drug_b_idx, cell_idx]
        d_a_embs = updated_nodes['drug'][batch_queries[:, 0]]
        d_b_embs = updated_nodes['drug'][batch_queries[:, 1]]
        c_embs = updated_nodes['cell'][batch_queries[:, 2]]

        # 4. 融合与预测
        combined = torch.cat([d_a_embs, d_b_embs, c_embs], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ==========================================
# 4. 训练与评估
# ==========================================
if __name__ == "__main__":
    DATA_DIR = './data'

    # 1. 加载或构建异构图
    graph = load_or_build_graph(data_dir=DATA_DIR)

    # 2. 数据集构建与划分 (8:2)
    # 重新加载 CSV 获取原始映射关系
    df_drug = pd.read_csv(os.path.join(DATA_DIR, 'demo_drug_features.csv'))
    df_cell = pd.read_csv(os.path.join(DATA_DIR, 'demo_cell_features.csv'))
    drug_id2idx = {str(d): i for i, d in enumerate(df_drug['ex_drug_id'])}
    cell_id2idx = {str(c): i for i, c in enumerate(df_cell['ex_cell_id'])}

    full_dataset = SynergyDataset(os.path.join(DATA_DIR, 'demo_synergy_labels.csv'), drug_id2idx, cell_id2idx)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 初始化模型
    DRUG_DIM = graph['drug'].x.shape[1]
    CELL_DIM = graph['cell'].x.shape[1]

    model = SynergyPredictor(drug_dim=DRUG_DIM, cell_dim=CELL_DIM, hidden_dim=64, metadata=graph.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # 4. 模型训练和测试
    print("\n 开始训练...")

    for epoch in range(1, 11):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        for batch_queries, labels in train_loader:
            optimizer.zero_grad()
            # 模型同时接收全图结构和当前批次的查询任务
            preds = model(graph, batch_queries)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(labels)

        avg_train_loss = total_train_loss / train_size

        # --- 测试阶段 ---
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_queries, labels in test_loader:
                preds = model(graph, batch_queries)
                loss = criterion(preds, labels)
                total_test_loss += loss.item() * len(labels)

        avg_test_loss = total_test_loss / test_size

        print(f"Epoch {epoch:02d} | Train Loss (MSE): {avg_train_loss:.4f} | Test Loss (MSE): {avg_test_loss:.4f}")

    # 5. 模型与预测结果保存
    print("\n 训练结束，正在保存模型和预测结果...")

    # 1. 保存模型权重 (State Dict)
    model_path = os.path.join(DATA_DIR, 'macro_hetero_model.pth')
    torch.save(model.state_dict(), model_path)

    # 2. 收集测试集上的最终预测值和真实值
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_queries, labels in test_loader:
            preds = model(graph, batch_queries)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())

    # 3. 将结果保存为 CSV 文件
    results_df = pd.DataFrame({
        'True_Synergy': all_trues,
        'Predicted_Synergy': all_preds
    })
    results_path = os.path.join(DATA_DIR, 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)

    print(f"模型权重已保存至: {model_path}")
    print(f"预测结果已保存至: {results_path}")