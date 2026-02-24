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
    # "expression_matrix": "/home/u2024001153/PythonProject/proteomics-data/Huffman2022v2-Figure456/processed_expression_matrix.csv",
    # "expression_matrix": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure5/processed_expression_matrix.csv",
    # "expression_matrix": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure7/processed_expression_matrix.csv",
    
    # GO相似性文件路径
    "go_similarity": "/home/u2024001153/PythonProject/scmodel3_1016/leduc/GO_CC_similarity_matrix.csv",
    # "go_similarity": "/home/u2024001153/PythonProject/scmodel3_1016/SCoPE2/GO_CC_similarity_matrix_Proteins-processed_processed.csv",
    # "go_similarity": "/home/u2024001153/PythonProject/scmodel3_1016/Khan2024/GO_CC_similarity_matrix_npop1_proteinMatrix_Imputed.NoBCByMSRun_processed.csv",
    # "go_similarity": "/home/u2024001153/PythonProject/proteomics-data/Huffman2022v2-Figure456/GO_CC_similarity_matrix_limmaCorrected_normed_prePCA.csv",
    # "go_similarity": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure5/GO_CC_similarity_matrix_plexDIA_SingleCellxProtein_fig5_processed.csv",
    # "go_similarity": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure7/GO_CC_similarity_matrix_TMT_ProteinxSingleCell_fig7_processed.csv",
    
    # 细胞标签文件路径
    "cell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/leduc/meta.csv",
    # "cell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/SCoPE2/Cells_T.csv",
    # "cell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/Khan2024/oldAlign_npop1_order_id.csv",
    # "cell_labels": "/home/u2024001153/PythonProject/proteomics-data/Huffman2022v2-Figure456/matched_cell_labels.csv",
    # "cell_labels": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure5/matched_cell_labels.csv",
    # "cell_labels": "/home/u2024001153/PythonProject/proteomics-data/Leduc2023-Figure7/matched_cell_labels.csv",
    
    # 超细胞文件路径
    "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_leduc.csv",
    # "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_SCoPE2.csv",
    # "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_Khan2024.csv",
    # "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_Huffman2022v2-Figure456.csv",
    # "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_Leduc2023-Figure5.csv",
    # "supercell_labels": "/home/u2024001153/PythonProject/scmodel3_1016/supercell_Leduc2023-Figure7.csv",
    
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
