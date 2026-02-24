"""
scYOU 模型定义模块
包含编码器、解码器、聚类层和双路径自编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from config.config import MODEL_CONFIG

# 细胞路径的MLP编码器（全连接网络）
class CellEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 activation=F.relu, num_layers: int = 3):  # 3层结构
        super(CellEncoder, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
        else:  # 极简架构
            self.layers.append(nn.Linear(in_channels, 256))  # 第一隐藏层256
            self.layers.append(nn.Linear(256, 64))          # 第二隐藏层64
            self.layers.append(nn.Linear(64, out_channels))   # 输出层32

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# 细胞路径的MLP解码器（全连接网络）
class CellDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 activation=F.relu, num_layers: int = 3):  # 3层结构
        super(CellDecoder, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
        else:  # 极简架构
            self.layers.append(nn.Linear(in_channels, 64))   # 第一隐藏层64
            self.layers.append(nn.Linear(64, 256))           # 第二隐藏层256
            self.layers.append(nn.Linear(256, out_channels)) # 输出层

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# 蛋白路径的编码器（全连接网络）
class ProteinEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 activation=F.relu, num_layers: int = 3):  # 3层结构
        super(ProteinEncoder, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
        else:  # 极简架构
            self.layers.append(nn.Linear(in_channels, 256))  # 第一隐藏层256
            self.layers.append(nn.Linear(256, 64))        # 第二隐藏层64
            self.layers.append(nn.Linear(64, out_channels))   # 输出层32

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# 蛋白路径的解码器（全连接网络）
class ProteinDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 activation=F.relu, num_layers: int = 3):  # 3层结构
        super(ProteinDecoder, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        elif num_layers == 2:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
        else:  # 极简架构
            self.layers.append(nn.Linear(in_channels, 64))   # 第一隐藏层64
            self.layers.append(nn.Linear(64, 256))           # 第二隐藏层256
            self.layers.append(nn.Linear(256, out_channels)) # 输出层

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# 聚类模块（DEC风格） - 添加目标分布计算和KL损失
class ClusterLayer(nn.Module):
    def __init__(self, n_clusters, z_dim, alpha=MODEL_CONFIG["alpha_cluster"]):
        super(ClusterLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))
        # 使用确定的初始化方法，避免随机波动
        nn.init.xavier_uniform_(self.mu, gain=nn.init.calculate_gain('relu'))
        
    def soft_assign(self, z):
        """计算软分配概率（学生t分布）"""
        # z: [n, d], mu: [k, d]
        dist = torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2)  # [n, k]
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()  # 归一化
        return q
    
    def target_distribution(self, q):
        """计算目标分布（增强高置信度分配）"""
        p = q**2 / q.sum(0)  # 增强高置信度的分配
        return (p.t() / p.sum(1)).t()  # 归一化
    
    def kl_loss(self, p, q):
        """计算KL散度损失"""
        return torch.mean(torch.sum(p * torch.log(p / (q + 1e-8)), dim=1))

# 双路径自编码器模型 - 添加聚类功能
class DualPathAE(nn.Module):
    def __init__(self, num_cells, num_proteins, 
                 high_var_protein_indices=None, 
                 go_protein_indices=None,
                 proteins_list=None):  # 新增蛋白名称列表
        super(DualPathAE, self).__init__()

        # 设置嵌入维度
        self.cell_embed_dim = MODEL_CONFIG["cell_embed_dim"]  # 细胞嵌入维度固定为32
        self.protein_embed_dim = MODEL_CONFIG["protein_embed_dim"]  # 蛋白嵌入维度固定为32
        
        # 新增：保存高方差蛋白索引
        self.high_var_protein_indices = high_var_protein_indices
        self.high_var_count = len(high_var_protein_indices) if high_var_protein_indices is not None else num_proteins

        # 保存蛋白名称列表
        self.proteins_list = proteins_list
        
        # 细胞路径：MLP自编码器 - 输入维度改为高方差蛋白的数量
        self.cell_encoder = CellEncoder(
            in_channels=self.high_var_count,  # 只使用高方差蛋白
            hidden_channels=256,  # 隐藏层大小（仅用于兼容）
            out_channels=self.cell_embed_dim,  # 输出维度32
            num_layers=3  # 3层结构
        )
        
        self.cell_decoder = CellDecoder(
            in_channels=self.cell_embed_dim,
            hidden_channels=256,  # 隐藏层大小（仅用于兼容）
            out_channels=self.high_var_count,  # 重建高方差蛋白
            num_layers=3  # 3层结构
        )
        
        # 蛋白路径：普通自编码器 - 使用完整的细胞特征
        self.protein_encoder = ProteinEncoder(
            in_channels=num_cells,
            hidden_channels=256,  # 隐藏层大小（仅用于兼容）
            out_channels=self.protein_embed_dim,  # 输出维度32
            num_layers=3  # 3层结构
        )
        
        self.protein_decoder = ProteinDecoder(
            in_channels=self.protein_embed_dim,
            hidden_channels=256,  # 隐藏层大小（仅用于兼容）
            out_channels=num_cells,
            num_layers=3  # 3层结构
        )

        # 聚类模块
        self.cluster_layer = ClusterLayer(
            n_clusters=10,  # 初始聚类数，训练时可调整
            z_dim=self.cell_embed_dim,  # 使用细胞嵌入维度
            alpha=MODEL_CONFIG["alpha_cluster"]
        )

        # GO相似性矩阵和蛋白索引
        self.go_similarity = None
        self.go_protein_indices = go_protein_indices
        
    def forward(self, x):
        # 新增：只选择高方差蛋白用于细胞路径
        if self.high_var_protein_indices is not None:
            x_high_var = x[:, self.high_var_protein_indices]  # (num_cells, high_var_count)
        else:
            x_high_var = x  # 如果没有指定高方差蛋白，使用全部
        
        # 细胞路径 - 只使用高方差蛋白
        cell_latent = self.cell_encoder(x_high_var)
        cell_recon = self.cell_decoder(cell_latent)
        
        # 蛋白路径 - 使用完整的表达矩阵
        protein_x = x.t()  # 转置为蛋白×细胞
        protein_latent = self.protein_encoder(protein_x)
        protein_recon = self.protein_decoder(protein_latent)
        
        # 新增：选择高方差蛋白对应的重建结果用于对比损失
        if self.high_var_protein_indices is not None:
            protein_recon_high_var = protein_recon[self.high_var_protein_indices, :]
        else:
            protein_recon_high_var = protein_recon
            
        return cell_latent, cell_recon, protein_latent, protein_recon, protein_recon_high_var.t()
    
    def get_cell_embedding(self, x):
        # 新增：在输入细胞编码器前选择高方差蛋白
        with torch.no_grad():
            if self.high_var_protein_indices is not None:
                x_high_var = x[:, self.high_var_protein_indices]
            else:
                x_high_var = x
            return self.cell_encoder(x_high_var)
    
    def get_protein_embedding(self, x):
        with torch.no_grad():
            protein_x = x.t()
            return self.protein_encoder(protein_x)
    
    def get_go_loss(self, protein_embed):
        """计算GO相似性损失，仅计算GO矩阵中存在的蛋白"""
        if self.go_similarity is None or self.go_protein_indices is None:
            return torch.tensor(0.0, device=protein_embed.device)
        
        try:
            # 确保有可计算的蛋白
            if len(self.go_protein_indices) == 0:
                return torch.tensor(0.0, device=protein_embed.device)
                
            # 获取GO蛋白的嵌入向量
            selected_protein_embed = protein_embed[self.go_protein_indices]
            n = len(self.go_protein_indices)
            
            # 获取这些蛋白的名称
            selected_protein_names = [self.proteins_list[i] for i in self.go_protein_indices]
            
            # 创建蛋白名到索引列表的映射
            name_to_indices = {}
            for idx, name in enumerate(selected_protein_names):
                if name not in name_to_indices:
                    name_to_indices[name] = []
                name_to_indices[name].append(idx)
            
            # 创建掩码矩阵，用于忽略相同蛋白名之间的比较
            mask = torch.ones((n, n), device=protein_embed.device)
            
            # 将相同蛋白名之间的位置设为0（忽略）
            for name, indices in name_to_indices.items():
                if len(indices) > 1:  # 只有多个同名的才需要处理
                    for i in indices:
                        for j in indices:
                            if i != j:  # 排除对角线（相同向量的比较）
                                mask[i, j] = 0
                                mask[j, i] = 0
            
            # 归一化嵌入
            selected_protein_embed_norm = F.normalize(selected_protein_embed, dim=1)
            
            # 计算嵌入相似度矩阵
            embed_similarity = torch.mm(selected_protein_embed_norm, selected_protein_embed_norm.t())
            
            # 确保GO矩阵与当前选择匹配
            if self.go_similarity.size(0) != n or self.go_similarity.size(1) != n:
                print(f"Adjusting GO matrix from {self.go_similarity.shape} to ({n}, {n})")
                # 如果GO矩阵大于需要的大小，截取前n行和前n列
                if self.go_similarity.size(0) > n:
                    go_sim_adjusted = self.go_similarity[:n, :n]
                # 如果小于，用0填充
                else:
                    go_sim_adjusted = torch.zeros((n, n), device=protein_embed.device)
                    min_dim = min(n, self.go_similarity.size(0))
                    go_sim_adjusted[:min_dim, :min_dim] = self.go_similarity[:min_dim, :min_dim]
            else:
                go_sim_adjusted = self.go_similarity
            
            # 应用掩码，忽略相同蛋白名之间的比较
            masked_embed_similarity = embed_similarity * mask
            masked_go_similarity = go_sim_adjusted * mask
            
            # 计算MSE损失（只考虑掩码后的部分）
            return F.mse_loss(masked_embed_similarity, masked_go_similarity)
            
        except Exception as e:
            print(f"GO loss calculation error: {str(e)}")
            return torch.tensor(0.0, device=protein_embed.device)
    
    def get_cluster_assignments(self, x):
        """获取聚类分配概率"""
        with torch.no_grad():
            cell_latent = self.get_cell_embedding(x)  # 使用更新后的get_cell_embedding
            q = self.cluster_layer.soft_assign(cell_latent)
        return q