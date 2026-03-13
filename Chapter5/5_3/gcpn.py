import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# ============================================================
# 基础类与环境 (Environment & MDP Simulator)
# ============================================================

class SimpleGraphEnv:
    """
    简单的图生成环境 (MDP Simulator)
    状态 (State): 当前的图结构 (节点特征和邻接矩阵)
    动作 (Action): 添加新节点或在现有节点间添加边
    奖励 (Reward): 模拟奖励，例如最大化特定类型节点的数量或图的连通性
    """
    def __init__(self, max_nodes=10, num_node_types=3):
        self.max_nodes = max_nodes
        self.num_node_types = num_node_types
        self.reset()

    def reset(self):
        """重置环境，返回初始状态 (一个只包含一个随机类型节点的图)"""
        self.graph = nx.Graph()
        initial_node_type = random.randint(0, self.num_node_types - 1)
        self.graph.add_node(0, type=initial_node_type)
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """获取当前状态的张量表示"""
        num_nodes = self.graph.number_of_nodes()
        
        # 节点特征矩阵 (One-hot 编码)
        x = torch.zeros((self.max_nodes, self.num_node_types))
        for i in range(num_nodes):
            x[i, self.graph.nodes[i]['type']] = 1.0
            
        # 邻接矩阵
        adj = torch.zeros((self.max_nodes, self.max_nodes))
        for u, v in self.graph.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            
        # 掩码，指示哪些节点是真实存在的
        mask = torch.zeros(self.max_nodes)
        mask[:num_nodes] = 1.0
        
        return x, adj, mask

    def step(self, action):
        """
        执行动作并返回 (next_state, reward, done)
        动作空间简化为:
        - 0: 停止生成
        - 1: 添加一个新节点 (类型0) 并与当前最后一个节点连接
        - 2: 添加一个新节点 (类型1) 并与当前最后一个节点连接
        - 3: 添加一个新节点 (类型2) 并与当前最后一个节点连接
        """
        num_nodes = self.graph.number_of_nodes()
        done = False
        reward = 0.0
        
        if action == 0 or num_nodes >= self.max_nodes:
            done = True
        else:
            # 添加新节点
            new_node_type = action - 1
            new_node_id = num_nodes
            self.graph.add_node(new_node_id, type=new_node_type)
            
            # 简单起见，总是与前一个节点连接
            if new_node_id > 0:
                self.graph.add_edge(new_node_id - 1, new_node_id)
                
        self.current_step += 1
        if self.current_step >= self.max_nodes * 2:
            done = True
            
        # 计算奖励: 鼓励生成类型为 1 的节点 (模拟某种期望的化学性质)
        if done:
            reward = self._calculate_reward()
            
        return self._get_state(), reward, done

    def _calculate_reward(self):
        """计算最终图的奖励"""
        reward = 0.0
        for i in self.graph.nodes():
            if self.graph.nodes[i]['type'] == 1:
                reward += 1.0 # 鼓励类型1
            elif self.graph.nodes[i]['type'] == 2:
                reward -= 0.5 # 惩罚类型2
        
        # 鼓励连通性 (简单图总是连通的，这里只是示例)
        if nx.is_connected(self.graph):
            reward += 2.0
            
        return reward

# ============================================================
# 模型实现 (Policy Network)
# ============================================================

class GraphConvolution(nn.Module):
    """简单的图卷积层"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (N, in_features), adj: (N, N)
        # 添加自环
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        # 度矩阵
        degree = torch.sum(adj_hat, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # 归一化邻接矩阵: D^{-1/2} A D^{-1/2}
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_hat), d_mat_inv_sqrt)
        
        # 卷积操作
        support = self.linear(x)
        output = torch.mm(norm_adj, support)
        return output

class GCPNPolicy(nn.Module):
    """
    Graph Convolutional Policy Network (GCPN)
    使用 GCN 提取图状态特征，并输出动作的概率分布。
    
    数学公式 (Policy Gradient):
    \\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta} \\left[ \\sum_{t=0}^{T-1} \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) R(\\tau) \\right]
    其中 \\pi_\\theta 是策略网络，a_t 是动作，s_t 是状态，R(\\tau) 是轨迹的总回报。
    """
    def __init__(self, num_node_types, hidden_dim=32, action_dim=4):
        super(GCPNPolicy, self).__init__()
        self.gcn1 = GraphConvolution(num_node_types, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # 动作预测头 (基于图级别特征)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, adj, mask):
        """
        前向传播
        x: (max_nodes, num_node_types)
        adj: (max_nodes, max_nodes)
        mask: (max_nodes,)
        """
        # 图卷积提取节点特征
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        
        # 聚合节点特征得到图级别特征 (Readout)
        # 仅聚合真实存在的节点
        mask_expanded = mask.unsqueeze(1).expand_as(h)
        h_masked = h * mask_expanded
        graph_feat = torch.sum(h_masked, dim=0) / (torch.sum(mask) + 1e-8)
        
        # 预测动作概率 (Logits)
        action_logits = self.action_head(graph_feat)
        
        # 返回动作的概率分布
        return F.softmax(action_logits, dim=-1)

# ============================================================
# 训练与可视化 (Train & Viz)
# ============================================================

def train_gcpn(env, policy, optimizer, num_episodes=500, gamma=0.99):
    """
    使用 REINFORCE 算法训练 GCPN
    """
    policy.train()
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            x, adj, mask = state
            
            # 获取动作概率分布
            action_probs = policy(x, adj, mask)
            
            # 采样动作
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            
            # 记录 log probability
            log_probs.append(m.log_prob(action))
            
            # 执行动作
            next_state, reward, done = env.step(action.item())
            
            rewards.append(reward)
            state = next_state
            
        # 计算折扣回报 (Discounted Returns)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # 标准化回报以稳定训练
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # 计算 Policy Gradient 损失
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        # 反向传播与优化
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Epoch [{(episode + 1):>3}/{num_episodes}] Reward: {avg_reward:.4f} | Loss: {policy_loss.item():.4f}")
            
    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_history, alpha=0.6, color='blue', label='Episode Reward')
    # 平滑曲线
    window = 20
    if len(rewards_history) >= window:
        smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), smoothed, color='red', label='Smoothed Reward')
    plt.title('GCPN Training Curve (REINFORCE)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('gcpn_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存训练曲线至 gcpn_training_curve.png")
    
    return policy

def visualize_optimization(env, policy):
    """
    可视化优化前后的图结构
    """
    policy.eval()
    
    # 1. 随机生成的图 (未优化)
    random_env = SimpleGraphEnv(max_nodes=env.max_nodes, num_node_types=env.num_node_types)
    random_env.reset()
    done = False
    while not done:
        action = random.randint(0, 3)
        _, _, done = random_env.step(action)
    initial_graph = random_env.graph
    
    # 2. 策略生成的图 (已优化)
    state = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            x, adj, mask = state
            action_probs = policy(x, adj, mask)
            action = torch.argmax(action_probs).item() # 使用贪心策略生成
            state, _, done = env.step(action)
    optimized_graph = env.graph
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 节点颜色映射
    color_map = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
    
    # 绘制初始图
    ax = axes[0]
    pos1 = nx.spring_layout(initial_graph, seed=42)
    colors1 = [color_map[initial_graph.nodes[n]['type']] for n in initial_graph.nodes()]
    nx.draw(initial_graph, pos1, ax=ax, with_labels=True, node_color=colors1, 
            node_size=500, font_weight='bold', edge_color='gray')
    ax.set_title("Initial Random Graph\\n(Reward: {:.2f})".format(random_env._calculate_reward()))
    
    # 绘制优化图
    ax = axes[1]
    pos2 = nx.spring_layout(optimized_graph, seed=42)
    colors2 = [color_map[optimized_graph.nodes[n]['type']] for n in optimized_graph.nodes()]
    nx.draw(optimized_graph, pos2, ax=ax, with_labels=True, node_color=colors2, 
            node_size=500, font_weight='bold', edge_color='gray')
    ax.set_title("Optimized Graph by GCPN\\n(Reward: {:.2f})".format(env._calculate_reward()))
    
    # 添加图例
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='lightblue', label='Type 0 (Neutral)'),
        mpatches.Patch(color='lightgreen', label='Type 1 (Target, +1.0)'),
        mpatches.Patch(color='salmon', label='Type 2 (Penalty, -0.5)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig('gcpn_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存优化对比图至 gcpn_optimization.png")

# ============================================================
# 主程序 (Main)
# ============================================================

def main():
    print("============================================================")
    print("GCPN 强化学习分子优化")
    print("============================================================")
    
    # 配置参数
    MAX_NODES = 10
    NUM_NODE_TYPES = 3
    HIDDEN_DIM = 64
    ACTION_DIM = 4 # 0: Stop, 1: Add Type 0, 2: Add Type 1, 3: Add Type 2
    NUM_EPISODES = 500
    LR = 0.01
    
    print("配置:")
    print(f"  最大节点数: {MAX_NODES}")
    print(f"  节点类型数: {NUM_NODE_TYPES}")
    print(f"  隐藏层维度: {HIDDEN_DIM}")
    print(f"  训练回合数: {NUM_EPISODES}")
    print(f"  学习率: {LR}")
    
    print("创建模型...")
    env = SimpleGraphEnv(max_nodes=MAX_NODES, num_node_types=NUM_NODE_TYPES)
    policy = GCPNPolicy(num_node_types=NUM_NODE_TYPES, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    print("开始训练...")
    trained_policy = train_gcpn(env, policy, optimizer, num_episodes=NUM_EPISODES)
    
    print("生成可视化结果...")
    visualize_optimization(env, trained_policy)
    
    print("完成！")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
