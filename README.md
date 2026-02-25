# scYOU

## 1. 项目结构

```
 scYOU/
├── config/
│ └── config.py # 所有配置项集中管理
├── data/
│ ├── expression/ # 蛋白表达
│ ├── GO/ # GO相似性约束
│ ├── meta/ # 含聚类真实标签meta文件
│ └── supercell/ # 超细胞约束
├── src/
│ ├── init.py # 包初始化文件
│ ├── models.py # 模型定义
│ ├── trainer.py # 训练类
│ └── utils.py # 辅助函数
├── main.py # 主程序入口
└── requirements.txt # 环境依赖
└── README.md # 运行说明
```

这里data文件夹未给出，可通过zenodo下载完整数据：[Code for scYOU](https://zenodo.org/records/18756874)。



## 2. 快速开始

### 2.1. 安装环境依赖

- 安装依赖：

```
pip install -r requirements.txt
```

### 2.2. 配置参数和文件路径

在`config.py`中设置参数和文件路径：

- 模型参数配置

```python
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
```

- 数据集配置（以Montalvo数据集为例）

```python
# ===================== 文件路径配置 =====================
FILE_PATHS = {
    # 选择要使用的数据集
    "expression_matrix": "./data/expression/expression_Montalvo.csv",
    
    # GO相似性文件路径
    "go_similarity": "./data/GO/GO_Montalvo.csv",
    
    # 细胞标签文件路径
    "cell_labels": "./data/meta/meta_Montalvo.csv",
    
    # 超细胞文件路径
    "supercell_labels": "./data/supercell/supercell_Montalvo.csv",
    
    # 结果输出目录
    "results_base_dir": "./grid_search_results/",
    "loss_plots_dir": "./loss_curves/",
    "embeddings_dir": "./embeddings/"
}
```

```python
# ===================== 细胞标签列名配置 =====================
LABEL_COLUMNS = {
    # 根据使用的数据集选择对应的列名
    "cell_type_column": "Cell_type",  # Montalvo
    "supercell_column": "supercell_label"
}
```

- 其余模型超参数配置

```python
# ===================== 网格搜索参数配置 =====================
GRID_SEARCH_PARAMS = {
    # Montalvo 参数配置
    "n_top_var": [501], # 取全部蛋白维度
    "num_protos": [5], # 取真实聚类数
    'alpha': [1.0],
    'beta': [0.1],
    'learning_rate': [0.001],
    'tol': [0.005],
    
    # 通用配置
    "gamma": [1.0],
    "delta": [1.0],
    "update_interval": [10],
    "seed": [9842]
}
```

- 环境配置

```python
# ===================== 环境配置 =====================
ENV_CONFIG = {
    "global_seed": 42,
    "device": torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
    "result_output_dir": "./grid_search_results/"
}
```

### 2.3. 运行主代码

```
python main.py
```

### 2.4. 查看结果

- 运行完成后，最优参数和聚类指标会直接输出到控制台
- 损失曲线和聚类对比结果会保存到`./grid_search_results/`目录下


