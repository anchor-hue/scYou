"""
scYOU 项目配置文件
包含所有可配置参数和文件路径
"""
import torch

# ===================== 模型参数配置 =====================
MODEL_CONFIG = {
    "cell_embed_dim": 32,          # 细胞嵌入维度
    "protein_embed_dim": 32,       # 蛋白嵌入维度
    "weight_decay": 1e-5,          # 权重衰减
    "max_pretrain_epochs": 2000,   # 最大预训练轮数
    "max_train_epochs": 2000,      # 最大训练轮数
    "convergence_patience": 15,    # 收敛耐心值
    "convergence_threshold": 1e-4, # 收敛阈值
    "tau": 0.5,                    # 对比损失温度参数
    "alpha_cluster": 1.0           # 聚类层alpha参数
}

# ===================== 文件路径配置 =====================
FILE_PATHS = {
    # 选择要使用的数据集（注释/取消注释对应行）
    "expression_matrix": "./data/expression/expression_Montalvo.csv",
    # "expression_matrix": "./data/expression/expression_Specht.csv",
    # "expression_matrix": "./data/expression/expression_Khan.csv",
    # "expression_matrix": "./data/expression/expression_Huffman.csv",
    # "expression_matrix": "./data/expression/expression_Leduc-plexDIA.csv",
    # "expression_matrix": "./data/expression/expression_Leduc-TMT32.csv",
    
    # GO相似性文件路径
    "go_similarity": "./data/GO/GO_Montalvo.csv",
    # "go_similarity": "./data/GO/GO_Specht.csv",
    # "go_similarity": "./data/GO/GO_Khan.csv",
    # "go_similarity": "./data/GO/GO_Huffman.csv",
    # "go_similarity": "./data/GO/GO_Leduc-plexDIA.csv",
    # "go_similarity": "./data/GO/GO_Leduc-TMT32.csv",
    
    # 细胞标签文件路径
    "cell_labels": "./data/meta/meta_Montalvo.csv",
    # "cell_labels": "./data/meta/meta_Specht.csv",
    # "cell_labels": "./data/meta/meta_Khan.csv",
    # "cell_labels": "./data/meta/meta_Huffman.csv",
    # "cell_labels": "./data/meta/meta_Leduc-plexDIA.csv",
    # "cell_labels": "./data/meta/meta_Leduc-TMT32.csv",
    
    # 超细胞文件路径
    "supercell_labels": "./data/supercell/supercell_Montalvo.csv",
    # "supercell_labels": "./data/supercell/supercell_Specht.csv",
    # "supercell_labels": "./data/supercell/supercell_Khan.csv",
    # "supercell_labels": "./data/supercell/supercell_Huffman.csv",
    # "supercell_labels": "./data/supercell/meta_Leduc-plexDIA.csv",
    # "supercell_labels": "./data/supercell/GO_Leduc-TMT32.csv",
    
    # 结果输出目录
    "results_base_dir": "./grid_search_results/",
    "loss_plots_dir": "./loss_curves/",
    "embeddings_dir": "./embeddings/"
}

# ===================== 网格搜索参数配置 =====================
GRID_SEARCH_PARAMS = {
    # Montalvo 参数配置
    "n_top_var": [501], # 取全部蛋白维度
    "num_protos": [5], # 取真实聚类数
    'alpha': [1.0],
    'beta': [0.1],
    'learning_rate': [0.001],
    'tol': [0.005],

    # # Specht 参数配置（取消注释使用）
    # "n_top_var": [3042],
    # "num_protos": [2],
    # 'alpha': [1.0],
    # 'beta': [0.1],
    # 'learning_rate': [5e-4],
    # 'tol': [0.005],

    # # Khan 参数配置（取消注释使用）
    # "n_top_var": [3178],
    # "num_protos": [6],
    # 'alpha': [1.0],
    # 'beta': [0.1],
    # 'learning_rate': [5e-4],
    # 'tol': [0.005],

    # # Huffman 参数配置（取消注释使用）
    # "n_top_var": [1123],
    # "num_protos": [2],
    # 'alpha': [1.0],
    # 'beta': [0.1],
    # 'learning_rate': [0.001],
    # 'tol': [0.005],

    # # Leduc-plexDIA 参数配置（取消注释使用）
    # "n_top_var": [4619],
    # "num_protos": [3],
    # 'alpha': [1.0],
    # 'beta': [0.1],
    # 'learning_rate': [0.001],
    # 'tol': [0.005],

    # # Leduc-TMT32 参数配置（取消注释使用）
    # "n_top_var": [1775],
    # "num_protos": [3],
    # 'alpha': [0.5],
    # 'beta': [0.1],
    # 'learning_rate': [1e-4],
    # 'tol': [0.01],
    
    # 通用配置
    "gamma": [1.0],
    "delta": [1.0],
    "update_interval": [10],
    "seed": [9842]
}

# ===================== 环境配置 =====================
ENV_CONFIG = {
    "global_seed": 42,
    "device": torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
    "result_output_dir": "./grid_search_results/"
}

# ===================== 细胞标签列名配置 =====================
LABEL_COLUMNS = {
    # 根据使用的数据集选择对应的列名
    "cell_type_column": "Cell_type",  # Montalvo
    # "cell_type_column": "celltype",   # Specht/Huffman
    # "cell_type_column": "cellType",   # Khan
    # "cell_type_column": "sample",      # Leduc-plexDIA/Leduc-TMT32
    "supercell_column": "supercell_label"
}
