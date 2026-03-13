# 第五章配套代码

## 项目结构

```text
GNN/
├── 5_1/                           # 第5.1节：无约束分子生成示例代码
│   ├── graph_flow.py             # Graph Normalizing Flow (基于仿射耦合层)
│   ├── graph_vae.py              # GraphVAE (基于 ELBO 优化)
│   ├── molgan.py                 # MolGAN (基于 WGAN-GP 和 Gumbel-Softmax)
│   └── molecular_rnn.py          # MolecularRNN (基于 NodeRNN 和 EdgeRNN 的自回归生成)
├── 5_2/                           # 第5.2节：基于结构的分子生成示例代码
│   ├── geodiff.py                # GeoDiff (3D构象生成，连续坐标扩散)
│   ├── edm.py                    # EDM (等变扩散模型，3D分子一次性生成)
│   └── targetdiff_sbmg.py        # TargetDiff简化版 (基于结构的分子条件生成)
├── 5_3/                           # 第5.3节：分子优化示例代码
│   ├── gcpn.py                   # GCPN (基于策略梯度的强化学习分子优化)
│   └── vtjnn.py                  # VTJNN (基于图风格转换的变分自编码优化)
├── AGENTS.md                       # 本开发指南
├── 5_1_生成_无约束分子生成方法.docx
├── 5_2_生成_基于结构的分子生成.docx
└── 5_3_生成_分子优化.docx
```

## 开发环境

- **依赖包**: `torch`, `rdkit`, `networkx`, `matplotlib`, `numpy`

## 已实现模型详解

### 第 5.1 节：无约束 2D/3D 分子图生成

1. **Graph Normalizing Flow**: 通过一系列可逆的仿射耦合层将简单分布映射到分子分布。
2. **GraphVAE**: 基于变分自编码器和重参数化技巧。
3. **MolGAN**: 使用 WGAN-GP 和 Gumbel-Softmax 解决离散图生成的梯度问题。
4. **MolecularRNN**: 利用 NodeRNN 和 EdgeRNN 结合 Teacher Forcing 进行自回归生成。

### 第 5.2 节：基于结构的 3D 分子生成

1. **GeoDiff (`geodiff.py`)**
   - **核心原理**: 扩散模型在三维空间的应用。给定分子拓扑，通过向真实坐标逐步添加高斯噪声，再利用等变网络 (EGNN) 预测噪声以进行逆向去噪采样，从而生成稳定的 3D 笛卡尔坐标。
   - **关键技术**: DDPM扩散过程、E(3) / SE(3) 等变神经网络设计、三维空间中的质心零化 (Zero Center of Mass)。

2. **EDM (`edm.py`)**
   - **核心原理**: 等变扩散模型 (Equivariant Diffusion Model)。实现了无约束条件下的 3D 分子结构与原子类型的“一次性”联合生成。
   - **关键技术**: 连续特征 (3D坐标) 的高斯扩散与离散特征 (原子类别) 的分类扩散 (Categorical Diffusion) 的联合分布建模。

3. **TargetDiff / DiffSBDD Simplified (`targetdiff_sbmg.py`)**
   - **核心原理**: 基于结构的分子生成 (SBMG)。在 EDM 的基础上，将靶点蛋白口袋的 3D 几何与特征作为扩散过程的条件约束 (Condition)。
   - **关键技术**: 条件扩散生成、交叉注意力 (Cross-Attention) 融合环境约束、复合物图 (Bipartite Graph / k-NN Graph) 环境表示。

### 第 5.3 节：分子优化

1. **GCPN (`gcpn.py`)**
   - **核心原理**: 图卷积策略网络 (Graph Convolutional Policy Network)。将分子优化建模为马尔可夫决策过程 (MDP)，通过图编辑动作（如添加原子、连接键）来逐步改造分子。
   - **关键技术**: 策略梯度 (Policy Gradient / REINFORCE)、多目标奖励函数 (Reward Function)、图编辑动作空间建模。

2. **VTJNN (`vtjnn.py`)**
   - **核心原理**: 变分联结树编码解码器 (Variational Junction Tree Encoder-Decoder)。利用隐空间中的风格转换向量 ($\Delta z = z_{target} - z_{source}$) 实现分子图的“风格迁移”或局部优化。
   - **关键技术**: 联结树近似 (Tree + Graph 编码)、隐空间风格转换向量预测、成对分子训练策略。
