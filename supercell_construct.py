import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import networkx as nx
import random
import numpy as np
from collections import Counter

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
    
    # 应用分位数归一化
    # features = quantile_transform(features, n_quantiles=500, copy=True)
    # print("Applied quantile normalization")
    
    return proteins_list, cell_list, features

def construct_supercell_graph(features, a=5, b=3, random_seed=42):
    """
    基于细胞-蛋白矩阵构建超细胞划分
    
    参数:
        features: numpy数组, shape为[细胞数×蛋白数]的矩阵
        a: int, 每个细胞的最大相关邻居数
        b: int, 构建边的最小共同邻居数
        random_seed: int, 随机种子
        
    返回:
        supercell_labels: numpy数组, 每个细胞的超细胞标签
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 1. 计算细胞间的PCC矩阵
    n_cells = features.shape[0]
    print(f"Number of cells: {n_cells}")
    pcc_matrix = np.corrcoef(features)
    
    # 2. 对每个细胞找到a个最相关的邻居(不包括自身)
    neighbor_dict = {}
    for i in range(n_cells):
        # 获取相关性排序(降序),排除自身(i)
        sorted_indices = np.argsort(pcc_matrix[i])[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]
        top_a_neighbors = sorted_indices[:a]
        neighbor_dict[i] = set(top_a_neighbors)
    
    # 3. 构建邻接矩阵: 如果i和j的共同邻居数>b则连接
    adj_matrix = np.zeros((n_cells, n_cells), dtype=int)
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            common_neighbors = len(neighbor_dict[i] & neighbor_dict[j])
            if common_neighbors > b:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    # 4. 找出连通分量
    n_components, labels = connected_components(csgraph=csr_matrix(adj_matrix), directed=False)
    print(f"Found {n_components} connected components")
    
    # 5. 处理每个连通分量
    final_labels = np.zeros(n_cells, dtype=int)
    current_max_label = 0
    
    for comp_id in range(n_components):
        # 获取当前连通分量的所有细胞
        component_cells = np.where(labels == comp_id)[0]
        component_size = len(component_cells)
        
        if component_size < 3:
            # 小连通分量直接作为一个超细胞
            for cell in component_cells:
                final_labels[cell] = current_max_label
            current_max_label += 1
        else:
            # 对于大的连通分量(≥3个细胞),构建子图计算度中心性
            subgraph = nx.Graph()
            # 添加边
            for i in component_cells:
                for j in component_cells:
                    if adj_matrix[i, j] == 1 and i < j:
                        subgraph.add_edge(i, j)
            
            # 计算度中心性
            degrees = nx.degree_centrality(subgraph)
            cell_degrees = np.array([degrees[cell] for cell in component_cells])
            
            # 最小-最大归一化度中心性
            min_deg = np.min(cell_degrees)
            max_deg = np.max(cell_degrees)
            
            if max_deg > min_deg:  # 避免除以0
                normalized_degrees = (cell_degrees - min_deg) / (max_deg - min_deg)
            else:
                normalized_degrees = np.ones_like(cell_degrees)  # 所有节点度相同的情况
            
            # 转换为保留概率(归一化后的度中心性)
            keep_probs = normalized_degrees
            
            # 决定哪些细胞保留在当前超细胞,哪些单独成超细胞
            kept_cells = []
            separated_cells = []
            
            for idx, cell in enumerate(component_cells):
                if random.random() < keep_probs[idx]:
                    kept_cells.append(cell)
                else:
                    separated_cells.append(cell)
            
            # 分配标签
            # 首先处理保留的细胞(同一个超细胞)
            for cell in kept_cells:
                final_labels[cell] = current_max_label
            current_max_label += 1
            
            # 然后处理被剔除的细胞(各自单独成超细胞)
            for cell in separated_cells:
                final_labels[cell] = current_max_label
                current_max_label += 1
    
    print(f"Final supercell labels shape: {final_labels.shape}")
    return final_labels

def calculate_supercell_purity(supercell_labels, true_labels):
    """
    计算每个超细胞的纯度
    
    参数:
        supercell_labels: numpy数组, 每个细胞的超细胞标签
        true_labels: numpy数组, 每个细胞的真实标签
        
    返回:
        purity_dict: 字典, 键为超细胞标签, 值为该超细胞的纯度
        avg_purity: 所有超细胞的平均纯度
    """
    # 确保输入是numpy数组
    supercell_labels = np.asarray(supercell_labels)
    true_labels = np.asarray(true_labels)
    
    # 检查输入长度是否一致
    if len(supercell_labels) != len(true_labels):
        raise ValueError(f"supercell_labels和true_labels的长度必须相同: "
                         f"{len(supercell_labels)} vs {len(true_labels)}")
    
    # 获取所有唯一的超细胞标签
    unique_supercells = np.unique(supercell_labels)
    
    purity_dict = {}
    total_cells = 0
    total_purity = 0.0
    
    for sc_label in unique_supercells:
        # 获取当前超细胞的所有细胞索引
        mask = (supercell_labels == sc_label)
        sc_cells = true_labels[mask]
        sc_size = len(sc_cells)
        
        if sc_size == 0:
            purity = 0.0
        else:
            # 计算最常见的标签及其出现次数
            most_common = Counter(sc_cells).most_common(1)
            max_count = most_common[0][1] if most_common else 0
            purity = max_count / sc_size
        
        purity_dict[sc_label] = purity
        total_cells += sc_size
        total_purity += purity * sc_size
    
    # 计算加权平均纯度
    avg_purity = total_purity / total_cells if total_cells > 0 else 0.0
    
    return purity_dict, avg_purity

# 假设已有细胞×蛋白矩阵

proteins_list, cell_list, expression_matrix = myload_sc_proteomic_features('./data/expression/expression_Leduc-plexDIA.csv')

# 打印矩阵形状进行验证
print(f"Expression matrix shape: {expression_matrix.shape}")

# 加载真实标签
meta_df = pd.read_csv('./data/meta/meta_Leduc-plexDIA.csv', index_col=0)
true_labels = meta_df['sample'].values

print(f"True labels shape: {true_labels.shape}")

# 调用函数
supercell_labels = construct_supercell_graph(
    features=expression_matrix,
    a=5,          # 设置参数alpha
    b=2,          # 设置beta
    random_seed=42
)

# 保存结果
print(f"Saving supercell labels with shape: {supercell_labels.shape}")
pd.DataFrame(supercell_labels, columns=['supercell_label']).to_csv("./data/supercell/supercell_Leduc-plexDIA.csv", index=False)

# 计算纯度
purity_dict, avg_purity = calculate_supercell_purity(supercell_labels, true_labels)

print("各超细胞纯度:")
for sc_label, purity in purity_dict.items():
    print(f"超细胞 {sc_label}: 纯度 = {purity:.3f}")

print(f"\n平均纯度: {avg_purity:.3f}")

# # 保存纯度结果
# purity_df = pd.DataFrame(list(purity_dict.items()), columns=['Supercell', 'Purity'])
# purity_df.to_csv("./data/supercell/supercell_purity_Leduc-plexDIA.csv", index=False)
