"""
scYOU 工具函数模块
包含数据加载、随机种子设置、聚类评估等辅助功能
"""
import torch
import numpy as np
import random
import pandas as pd
import os
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from config.config import ENV_CONFIG

# 设置全局随机种子函数 - 增强版
def set_seed(seed):
    """设置所有可能的随机种子以保证最大程度的可复现性"""
    # 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA相关种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 确保CUDA确定性
    torch.backends.cudnn.deterministic = True  # 确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动优化（可能影响性能）
    torch.backends.cudnn.enabled = False       # 禁用cuDNN加速（进一步保证确定性）
    
    # 设置环境变量增强确定性（针对PyTorch）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 控制cuBLAS workspace
    torch.use_deterministic_algorithms(True)  # 启用确定性算法
    
    print(f"Set all random seeds to {seed} (enhanced determinism mode)")

def create_data_object(expression_matrix, device=ENV_CONFIG["device"]):
    """创建PyG数据对象（只有特征，无边）"""
    features = torch.tensor(expression_matrix, dtype=torch.float32).to(device)
    edge_index = torch.empty(2, 0, dtype=torch.long).to(device)
    return Data(x=features, edge_index=edge_index)

def myload_sc_proteomic_features(filename):
    """加载蛋白表达数据并处理NaN值"""
    # 1. 加载CSV文件
    df = pd.read_csv(filename, index_col=0)  # 读取为 [proteins × cells]
    
    # 记录原始NaN数量用于调试
    original_nan_count = df.isna().sum().sum()
    
    # 检查并替换NaN值为0
    df = df.fillna(0)
    
    # 转置到 [cells × proteins] 格式
    df = df.T
    
    # 获取表达矩阵、蛋白名称和细胞名称
    features = df.values.astype(float)
    proteins_list = df.columns.tolist()
    cell_list = df.index.tolist()
    
    # 计算处理后NaN数量
    final_nan_count = np.isnan(features).sum()
    
    print(f"Loaded data from {filename}")
    print(f"Original NaN count: {original_nan_count}")
    print(f"Final NaN count: {final_nan_count}")
    
    # 确保无NaN
    if final_nan_count > 0:
        features = np.nan_to_num(features, nan=0.0)
        print(f"Force-replaced remaining {final_nan_count} NaN values to 0")
    
    return proteins_list, cell_list, features

def select_top_var_proteins(data, n_top):
    """
    选择方差最大的前n个蛋白特征
    
    参数:
    data: 表达矩阵 (cells × proteins)
    n_top: 要选择的蛋白数量
    
    返回:
    high_var_indices: 高方差蛋白的索引
    """
    variances = np.var(data, axis=0)
    
    # 选择方差最大的前n个蛋白（排序是确定性的）
    top_var_indices = np.argsort(variances)[::-1][:n_top]
    high_var_indices = top_var_indices.tolist()
    
    print(f"Selected {len(high_var_indices)} high-variance proteins (top {n_top})")
    
    return high_var_indices

def load_go_similarity_matrix(go_file, data_proteins):
    """
    从CSV文件加载GO相似性矩阵并与数据蛋白质匹配
    返回:
    ◦ go_sim_matrix_tensor: 公共蛋白质的GO相似性矩阵的PyTorch张量
    ◦ go_protein_indices: 原始数据蛋白质列表中公共蛋白质的索引
    """
    # 从CSV加载GO矩阵
    go_df = pd.read_csv(go_file, index_col=0)
    
    # 获取GO蛋白质（行名）和相似性矩阵
    go_proteins = go_df.index.tolist()
    go_sim_matrix = go_df.values
    
    # 查找GO和数据之间的公共蛋白质
    common_proteins = set(data_proteins) & set(go_proteins)
    
    # 获取原始数据蛋白质列表中公共蛋白质的索引
    go_protein_indices = [i for i, prot in enumerate(data_proteins) if prot in common_proteins]
    
    # 获取GO矩阵中公共蛋白质的索引
    go_indices = [go_proteins.index(prot) for prot in common_proteins]
    
    # 提取公共蛋白质的子矩阵
    go_sim_matrix_sub = go_sim_matrix[go_indices, :]
    go_sim_matrix_sub = go_sim_matrix_sub[:, go_indices]
    
    # 转换为张量并标准化
    go_sim_matrix_tensor = torch.tensor(go_sim_matrix_sub, dtype=torch.float32)
    
    print(f"GO similarity matrix loaded: {len(go_proteins)} proteins in GO file")
    print(f"Common proteins: {len(common_proteins)}")
    print(f"GO matrix shape: {go_sim_matrix_sub.shape}")
    
    return go_sim_matrix_tensor, go_protein_indices

def evaluate_clustering(embeddings, true_labels, seed):
    """评估嵌入的聚类效果 - 增加seed参数控制随机性"""
    n_clusters = len(np.unique(true_labels))
    
    # KMeans聚类 - 固定随机状态
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 计算评估指标
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    return nmi, ari, cluster_labels