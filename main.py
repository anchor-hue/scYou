"""
scYOU 主程序入口
执行网格搜索和模型训练
"""
import os
import itertools
import pandas as pd
import torch
from tqdm import tqdm
from src import (
    set_seed, myload_sc_proteomic_features, create_data_object,
    select_top_var_proteins, load_go_similarity_matrix,
    DualPathAE, scYOU
)
from config.config import (
    MODEL_CONFIG, FILE_PATHS, GRID_SEARCH_PARAMS, 
    ENV_CONFIG, LABEL_COLUMNS
)

def grid_search_main():
    # 设置全局随机种子
    set_seed(ENV_CONFIG["global_seed"])
    print(f"Global random seed set to {ENV_CONFIG['global_seed']}")

    # 创建结果输出目录
    os.makedirs(ENV_CONFIG["result_output_dir"], exist_ok=True)
    os.makedirs(FILE_PATHS["results_base_dir"], exist_ok=True)
    
    # 加载数据
    proteins_list, cell_list, expression_matrix = myload_sc_proteomic_features(
        FILE_PATHS["expression_matrix"]
    )
    
    num_cells = len(cell_list)
    num_proteins = len(proteins_list)
    
    # 加载GO相似性矩阵
    go_sim_matrix_tensor, go_protein_indices = load_go_similarity_matrix(
        FILE_PATHS["go_similarity"],
        proteins_list
    )
    
    # 创建数据对象
    data = create_data_object(expression_matrix, device=ENV_CONFIG["device"])
    
    # 加载元数据（真实标签）
    meta_df = pd.read_csv(FILE_PATHS["cell_labels"], index_col=0)
    true_labels = meta_df[LABEL_COLUMNS["cell_type_column"]].values
    
    # 加载超细胞标签
    supercell_df = pd.read_csv(FILE_PATHS["supercell_labels"])
    supercell_labels = torch.tensor(
        supercell_df[LABEL_COLUMNS["supercell_column"]].values, 
        dtype=torch.long
    )
    
    # 生成所有参数组合
    param_names = list(GRID_SEARCH_PARAMS.keys())
    param_combinations = list(itertools.product(*[GRID_SEARCH_PARAMS[name] for name in param_names]))
    
    print(f"Starting grid search with {len(param_combinations)} parameter combinations")
    
    # 存储结果
    results = []
    
    for i, params in enumerate(tqdm(param_combinations, desc="Grid Search")):
        # 解包参数
        param_dict = dict(zip(param_names, params))
        
        # 为当前参数创建唯一标识符
        param_str = "_".join(f"{k}={v}" for k, v in param_dict.items())
        result_dir = f"{FILE_PATHS['results_base_dir']}/run_{i+1}_{param_str}"
        os.makedirs(result_dir, exist_ok=True)
        
        print(f"\n\n=== Running combination {i+1}/{len(param_combinations)}: {param_dict} ===")
        
        try:
            # 设置当前参数组合的随机种子（关键：确保每次运行从相同状态开始）
            set_seed(param_dict['seed'])

            # 根据当前参数选择高方差蛋白（确定性操作）
            n_top_var = param_dict['n_top_var']
            high_var_protein_indices = select_top_var_proteins(expression_matrix, n_top_var)
            
            # 初始化模型 - 传入高方差蛋白索引
            model = DualPathAE(
                num_cells=num_cells,
                num_proteins=num_proteins,
                high_var_protein_indices=high_var_protein_indices,
                go_protein_indices=go_protein_indices,  # 基于全蛋白的GO索引
                proteins_list=proteins_list  # 全蛋白列表
            )
            
            # 设置设备
            device = ENV_CONFIG["device"]
            
            # 创建训练器 (添加聚类参数)
            trainer = scYOU(
                model=model,
                device=device,
                data=data.to(device),
                go_similarity_matrix=go_sim_matrix_tensor,
                go_protein_indices=go_protein_indices,
                learning_rate=param_dict['learning_rate'],
                weight_decay=MODEL_CONFIG["weight_decay"],
                num_protos=param_dict['num_protos'],
                max_pretrain_epochs=MODEL_CONFIG["max_pretrain_epochs"],
                max_train_epochs=MODEL_CONFIG["max_train_epochs"],
                alpha=param_dict['alpha'],
                beta=param_dict['beta'],
                gamma=param_dict['gamma'],
                delta=param_dict['delta'],
                seed=param_dict['seed'],  # 使用参数中的种子
                supercell_labels=supercell_labels,  # 传入超细胞标签
                true_labels=true_labels,  # 传入真实标签用于聚类评估
                convergence_patience=MODEL_CONFIG["convergence_patience"],
                convergence_threshold=MODEL_CONFIG["convergence_threshold"],
                update_interval=param_dict['update_interval'],
                tol=param_dict['tol']
            )
            
            # 训练模型
            actual_pretrain_epochs = trainer.pretrain()
            actual_train_epochs = trainer.train()

            # 绘制损失曲线
            trainer.plot_losses(result_dir)
            
            # # 保存聚类比较结果
            # trainer.save_clustering_comparison(result_dir)
            
            # 记录实际训练轮数
            param_dict['pretrain_epochs'] = actual_pretrain_epochs
            param_dict['train_epochs'] = actual_train_epochs
            
            # 获取嵌入 - 使用训练器保存的最终嵌入
            cell_embeddings = trainer.final_cell_embeddings
            protein_embeddings = trainer.final_protein_embeddings
            cluster_labels = trainer.final_cluster_labels
            
            # 保存嵌入结果
            cell_df = pd.DataFrame(cell_embeddings, index=cell_list)
            cell_df.index.name = "Cell"
            cell_df.to_csv(f"{result_dir}/cell_embeddings.csv")
            
            protein_df = pd.DataFrame(protein_embeddings, index=proteins_list)
            protein_df.index.name = "Protein"
            protein_df.to_csv(f"{result_dir}/protein_embeddings.csv")
            
            # 保存预训练后的聚类标签
            pretrain_cluster_df = pd.DataFrame(
                trainer.pretrain_cluster_labels, 
                index=cell_list, 
                columns=['Pretrain_Cluster']
            )
            pretrain_cluster_df.to_csv(f"{result_dir}/pretrain_cluster_labels.csv")
            
            # 保存最终聚类标签
            final_cluster_df = pd.DataFrame(
                cluster_labels, 
                index=cell_list, 
                columns=['Final_Cluster']
            )
            final_cluster_df.to_csv(f"{result_dir}/final_cluster_labels.csv")
            
            # 保存高方差蛋白列表
            high_var_proteins = [proteins_list[i] for i in high_var_protein_indices]
            with open(f"{result_dir}/high_var_proteins.txt", "w") as f:
                for protein in high_var_proteins:
                    f.write(f"{protein}\n")
            
            # 记录结果，包括预训练和最终的聚类指标
            result = param_dict.copy()
            result['Final_NMI'] = trainer.final_nmi
            result['Final_ARI'] = trainer.final_ari
            result['Pretrain_NMI'] = trainer.pretrain_nmi
            result['Pretrain_ARI'] = trainer.pretrain_ari
            result['NMI_Improvement'] = trainer.final_nmi - trainer.pretrain_nmi
            result['ARI_Improvement'] = trainer.final_ari - trainer.pretrain_ari
            result['n_top_var'] = n_top_var
            result['result_dir'] = result_dir
            results.append(result)
            
            # print(f"Completed: Pretrain epochs={actual_pretrain_epochs}, "
            #       f"Train epochs={actual_train_epochs}, "
            #       f"Top VAR={n_top_var}, "
            #       f"Pretrain NMI={trainer.pretrain_nmi:.4f}, Pretrain ARI={trainer.pretrain_ari:.4f}, "
            #       f"Final NMI={trainer.final_nmi:.4f}, Final ARI={trainer.final_ari:.4f}")
            
            # # 保存当前结果到总文件
            # pd.DataFrame(results).to_csv(f"{FILE_PATHS['results_base_dir']}/grid_search_progress.csv", index=False)
            
        except Exception as e:
            print(f"Error in combination {param_dict}: {str(e)}")
            # 记录失败情况
            result = param_dict.copy()
            result['error'] = str(e)
            results.append(result)
            # pd.DataFrame(results).to_csv(f"{FILE_PATHS['results_base_dir']}/grid_search_progress.csv", index=False)
    
    # 保存最终结果
    results_df = pd.DataFrame(results)
    # results_df.to_csv(f"{FILE_PATHS['results_base_dir']}/final_grid_search_results.csv", index=False)
    print("Grid search completed!")
    
    # 找到最优参数组合
    if 'Final_NMI' in results_df.columns:
        best_idx = results_df['Final_NMI'].idxmax()
        best_result = results_df.loc[best_idx]
        # print(f"\nBest parameters based on Final NMI: {best_result['Final_NMI']:.4f}")
        # print(f"Pretrain vs Final: NMI {best_result['Pretrain_NMI']:.4f} -> {best_result['Final_NMI']:.4f} (Improvement: {best_result['NMI_Improvement']:.4f})")
        # print(f"Pretrain vs Final: ARI {best_result['Pretrain_ARI']:.4f} -> {best_result['Final_ARI']:.4f} (Improvement: {best_result['ARI_Improvement']:.4f})")
        print(best_result.to_dict())
    
    return results_df

if __name__ == "__main__":
    # 强制在主程序入口再次设置种子
    set_seed(ENV_CONFIG["global_seed"])
    grid_search_main()