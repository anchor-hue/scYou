"""
scYOU 训练器模块
包含scYOU训练类，负责模型的预训练、训练和评估
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from src.utils import set_seed, evaluate_clustering
from config.config import MODEL_CONFIG

# scYOU模型训练类（添加聚类训练机制）
class scYOU:
    def __init__(self, model, device, data, go_similarity_matrix, go_protein_indices,
                 learning_rate, weight_decay, num_protos, 
                 max_pretrain_epochs, max_train_epochs,
                 alpha, beta, gamma, delta, seed,
                 supercell_labels, true_labels,  # 新增真实标签用于评估
                 convergence_patience=MODEL_CONFIG["convergence_patience"], 
                 convergence_threshold=MODEL_CONFIG["convergence_threshold"],
                 update_interval=5, tol=0.001):  # 添加聚类更新参数
        self.model = model
        self.data = data
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_protos = num_protos
        self.max_pretrain_epochs = max_pretrain_epochs
        self.max_train_epochs = max_train_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.seed = seed  # 保存种子用于内部随机操作
        self.convergence_patience = convergence_patience
        self.convergence_threshold = convergence_threshold
        self.supercell_labels = supercell_labels.to(device)  # 保存超细胞标签
        self.true_labels = true_labels  # 保存真实标签用于聚类评估
        
        # 新增：存储预训练后的聚类指标
        self.pretrain_nmi = None
        self.pretrain_ari = None
        self.pretrain_cluster_labels = None

        # 设置GO相似性矩阵
        if go_similarity_matrix is not None:
            self.model.go_similarity = go_similarity_matrix.to(device)
        self.model.go_protein_indices = go_protein_indices
        
        # 将模型和数据移动到设备
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        
        # 优化器 - 为两条路径分别创建优化器
        # 细胞路径参数：编码器、解码器和聚类层
        cell_parameters = list(self.model.cell_encoder.parameters()) + \
                         list(self.model.cell_decoder.parameters()) + \
                         list(self.model.cluster_layer.parameters())
        
        # 蛋白路径参数：编码器和解码器
        protein_parameters = list(self.model.protein_encoder.parameters()) + \
                            list(self.model.protein_decoder.parameters())
        
        self.cell_optimizer = torch.optim.Adam(
            cell_parameters, 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            amsgrad=True  # 使用amsgrad变体增加稳定性
        )
        
        self.protein_optimizer = torch.optim.Adam(
            protein_parameters, 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            amsgrad=True  # 使用amsgrad变体增加稳定性
        )
        
        # 设置随机种子 - 确保可复现性
        set_seed(self.seed)
        
        # 添加聚类训练参数
        self.update_interval = update_interval  # 聚类目标分布更新间隔
        self.tol = tol  # 聚类标签变化容忍度
        
        # 初始化聚类标签
        self.y_pred_last = None

        # 添加损失历史记录
        self.pretrain_loss_history = {
            'total': [],
            'cell_recon': [],
            'protein_recon': [],
            'contrastive': []
        }
        
        self.train_loss_history = {
            'cell_total': [],
            'cell_recon': [],
            'cluster': [],
            'protein_total': [],
            'protein_recon': [],
            'go': []
        }

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """计算两个向量集合的余弦相似度"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def contrastive_loss_with_supercell(self, view1: torch.Tensor, view2: torch.Tensor):
        """
        基于超细胞标签的对比损失（InfoNCE风格）
        """
        tau = MODEL_CONFIG["tau"]  # 温度参数固定，避免随机性
        batch_size = view1.size(0)
        
        # 归一化嵌入向量
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(view1, view2.t()) / tau  # [batch_size, batch_size]
        
        # 创建同一超细胞的掩码矩阵
        mask = (self.supercell_labels.unsqueeze(0) == self.supercell_labels.unsqueeze(1)).float()
        mask = mask.to(view1.device)
        
        # 排除自身对比（对角线）
        mask.fill_diagonal_(0)
        
        # 计算正例和负例
        positives = sim_matrix * mask
        negatives = sim_matrix * (1 - mask)
        
        # 对于每个样本，计算分子和分母
        # 分子：与同一超细胞的所有样本的指数相似度之和
        exp_positives = torch.exp(positives).sum(dim=1)
        
        # 分母：与所有其他样本的指数相似度之和（包括负例和自身）
        exp_all = torch.exp(sim_matrix).sum(dim=1) - torch.exp(torch.diag(sim_matrix))
        
        # 计算损失
        loss_per_sample = -torch.log(exp_positives / (exp_all + 1e-8))
        
        return loss_per_sample.mean()

    def pretrain(self):
        """预训练阶段，直到损失收敛或达到最大轮数"""
        self.model.train()
        
        # 初始化变量用于收敛判断
        best_loss = float('inf')
        no_improvement_count = 0
        pretrain_epochs = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=self.max_pretrain_epochs, desc="Pretraining", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        for epoch in range(1, self.max_pretrain_epochs + 1):
            # 每次迭代前重置随机状态（增强确定性）
            set_seed(self.seed + epoch)
            
            # 联合优化两个路径的参数
            self.cell_optimizer.zero_grad()
            self.protein_optimizer.zero_grad()
            
            # 前向传播 - 获取五个返回值
            cell_latent, cell_recon, protein_latent, protein_recon, protein_recon_high_var = self.model(self.data.x)
            
            # 获取高方差蛋白索引对应的输入数据
            if self.model.high_var_protein_indices is not None:
                x_high_var = self.data.x[:, self.model.high_var_protein_indices]
            else:
                x_high_var = self.data.x
            
            # 计算重建损失（MSE损失）
            loss_recon_cell = F.mse_loss(cell_recon, x_high_var)
            
            # 蛋白重建损失
            loss_recon_protein = F.mse_loss(
                protein_recon, 
                self.data.x.t(),  # 原始数据的转置
            )
            
            # 计算对比损失（细胞重建矩阵 vs 蛋白重建矩阵的高方差蛋白部分转置）
            contrastive_loss = self.contrastive_loss_with_supercell(cell_recon, protein_recon_high_var)
            
            # 总损失
            total_loss = self.alpha * (loss_recon_cell + loss_recon_protein) + self.beta * contrastive_loss

            # 记录损失历史
            self.pretrain_loss_history['total'].append(total_loss.item())
            self.pretrain_loss_history['cell_recon'].append(loss_recon_cell.item())
            self.pretrain_loss_history['protein_recon'].append(loss_recon_protein.item())
            self.pretrain_loss_history['contrastive'].append(contrastive_loss.item())
            
            # 反向传播
            total_loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸（固定裁剪值）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新两个优化器
            self.cell_optimizer.step()
            self.protein_optimizer.step()
            
            # 更新进度条
            pbar.update(1)
            pretrain_epochs = epoch
            
            # 记录损失历史
            if epoch % 10 == 0:
                pbar.set_description(f"Pretrain Epoch {epoch:03d}: Loss={total_loss.item():.4f}")
                print(f'Pretrain Epoch {epoch:03d}: '
                      f'Total Loss = {total_loss.item():.4f}, '
                      f'Cell Recon = {loss_recon_cell.item():.4f}, '
                      f'Protein Recon = {loss_recon_protein.item():.4f}, '
                      f'Contrastive = {contrastive_loss.item():.4f}')
            
            # 收敛判断（使用固定阈值，避免浮点数波动影响）
            if total_loss.item() < best_loss - self.convergence_threshold:
                best_loss = total_loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 如果连续多个epoch没有显著改进，则提前停止预训练
            if no_improvement_count >= self.convergence_patience:
                print(f"\nEarly stopping at pretrain epoch {epoch} - loss converged")
                break
        
        # 关闭进度条
        pbar.close()
        
        # 返回实际预训练的轮数
        return pretrain_epochs

    def evaluate_pretrain_clustering(self):
        """在预训练结束后评估聚类效果"""
        print("\nEvaluating clustering after pretraining...")
        
        # 获取预训练后的细胞嵌入
        cell_embeddings = self.get_cell_embeddings()
        
        # 检查并处理NaN值
        if np.isnan(cell_embeddings).any():
            print("Warning: Cell embeddings contain NaN values. Replacing with 0.")
            cell_embeddings = np.nan_to_num(cell_embeddings, nan=0.0)
        
        # 确定聚类数量（与真实标签类别数相同）
        n_clusters = len(np.unique(self.true_labels))
        
        # 使用KMeans聚类 - 固定随机状态
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=20,  # 固定n_init避免随机波动
            random_state=self.seed  # 使用类初始化时的种子
        ).fit(cell_embeddings)
        
        # 获取聚类标签
        cluster_labels = kmeans.labels_
        
        # 计算评估指标
        nmi = normalized_mutual_info_score(self.true_labels, cluster_labels)
        ari = adjusted_rand_score(self.true_labels, cluster_labels)
        
        print(f"Clustering results after pretraining: NMI={nmi:.4f}, ARI={ari:.4f}")
        
        # 保存结果
        self.pretrain_nmi = nmi
        self.pretrain_ari = ari
        self.pretrain_cluster_labels = cluster_labels
        
        return nmi, ari, cluster_labels

    def initialize_cluster_centers(self):
        """使用预训练嵌入初始化簇中心"""
        with torch.no_grad():
            cell_embeds = self.model.get_cell_embedding(self.data.x)
            embeds_np = cell_embeds.cpu().numpy()
        
        # 检查并处理NaN值
        if np.isnan(embeds_np).any():
            print("Warning: Cell embeddings contain NaN values. Replacing with 0.")
            embeds_np = np.nan_to_num(embeds_np, nan=0.0)
        
        # K-means初始化簇中心 - 固定随机状态
        kmeans = KMeans(
            n_clusters=self.num_protos, 
            n_init=20,  # 固定n_init
            random_state=self.seed  # 使用类初始化时的种子
        ).fit(embeds_np)
        
        # 设置模型簇中心参数
        self.model.cluster_layer.mu.data = torch.tensor(
            kmeans.cluster_centers_, 
            dtype=torch.float32
        ).to(self.device)
        
        # 初始化聚类标签
        self.y_pred_last = kmeans.labels_
        print(f"Cluster centers initialized with K-means (seed={self.seed})")

    def train(self):
        """训练过程：先预训练直到收敛，然后正式训练直到收敛"""
        # 进行预训练直到收敛
        actual_pretrain_epochs = self.pretrain()
        print(f"Pretraining completed in {actual_pretrain_epochs} epochs")
        
        # 预训练结束后评估聚类效果
        self.evaluate_pretrain_clustering()
        
        # 更新聚类中心数
        self.model.cluster_layer.n_clusters = self.num_protos
        
        # 初始化聚类中心
        self.initialize_cluster_centers()
        
        # 正式训练 - 使用收敛停止机制
        self.model.train()
        
        # 为两条路径分别设置收敛变量
        best_cell_loss = float('inf')
        best_protein_loss = float('inf')
        no_improvement_cell = 0
        no_improvement_protein = 0
        actual_train_epochs = 0
        
        # 初始化目标分布p
        p = None
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=self.max_train_epochs, desc="Training", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        # 添加最小训练轮数要求
        min_train_epochs = 2  # 至少训练2个epoch
        
        # 标记两条路径是否已停止训练
        cell_stopped = False
        protein_stopped = False
        
        for epoch in range(1, self.max_train_epochs + 1):
            # 每次迭代前重置随机状态（增强确定性）
            set_seed(self.seed + actual_pretrain_epochs + epoch)
            
            # 前向传播 - 获取五个返回值
            cell_latent, cell_recon, protein_latent, protein_recon, _ = self.model(self.data.x)
            
            # 获取高方差蛋白索引对应的输入数据
            if self.model.high_var_protein_indices is not None:
                x_high_var = self.data.x[:, self.model.high_var_protein_indices]
            else:
                x_high_var = self.data.x
            
            # === 聚类目标分布更新 ===
            if epoch % self.update_interval == 0:
                with torch.no_grad():
                    # 计算软分配q
                    q = self.model.cluster_layer.soft_assign(cell_latent)
                    # 计算目标分布p
                    p = self.model.cluster_layer.target_distribution(q).detach()
                    
                    # 获取当前聚类标签
                    current_labels = torch.argmax(q, dim=1).cpu().numpy()
                    
                    # 计算标签变化率
                    if self.y_pred_last is not None:
                        delta_label = np.sum(current_labels != self.y_pred_last) / len(current_labels)
                        print(f'Epoch {epoch}: delta label = {delta_label:.4f}')
                        
                        # 检查是否达到停止条件，但至少训练min_train_epochs个epoch
                        if epoch >= min_train_epochs and delta_label < self.tol:
                            print(f'Reached tolerance threshold (delta_label < {self.tol}). Stopping training.')
                            break
                    
                    # 更新聚类标签
                    self.y_pred_last = current_labels
            
            # === 损失计算 ===
            # 计算重建损失
            loss_recon_cell = F.mse_loss(cell_recon, x_high_var)
            loss_recon_protein = F.mse_loss(protein_recon, self.data.x.t())
            
            # 计算DEC聚类损失（使用目标分布p）
            if p is not None:
                q_pred = self.model.cluster_layer.soft_assign(cell_latent)
                cluster_loss = self.model.cluster_layer.kl_loss(p, q_pred)
            else:
                cluster_loss = torch.tensor(0.0, device=self.device)
            
            # 计算GO相似性损失
            go_loss = self.model.get_go_loss(protein_latent)
            
            # 动态调整聚类损失权重（固定调整策略）
            if epoch > 100:
                effective_gamma = self.gamma * max(0.5, 1 - (epoch-100)/100)
            else:
                effective_gamma = self.gamma
            
            # 细胞路径损失：重建损失 + 聚类损失
            cell_total_loss = loss_recon_cell + effective_gamma * cluster_loss
            
            # 蛋白路径损失：重建损失 + GO损失
            protein_total_loss = loss_recon_protein + self.delta * go_loss

            # 记录训练损失历史
            self.train_loss_history['cell_total'].append(cell_total_loss.item())
            self.train_loss_history['cell_recon'].append(loss_recon_cell.item())
            self.train_loss_history['cluster'].append(cluster_loss.item())
            self.train_loss_history['protein_total'].append(protein_total_loss.item())
            self.train_loss_history['protein_recon'].append(loss_recon_protein.item())
            self.train_loss_history['go'].append(go_loss.item())

            # 细胞路径反向传播和参数更新（如果未停止）
            if not cell_stopped:
                self.cell_optimizer.zero_grad()
                cell_total_loss.backward(retain_graph=True)  # 保留计算图供蛋白路径使用
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.cell_encoder.parameters()) + 
                    list(self.model.cell_decoder.parameters()) +
                    list(self.model.cluster_layer.parameters()), 
                    max_norm=1.0
                )
                self.cell_optimizer.step()
            
            # 蛋白路径反向传播和参数更新（如果未停止）
            if not protein_stopped:
                self.protein_optimizer.zero_grad()
                protein_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.protein_encoder.parameters()) + 
                    list(self.model.protein_decoder.parameters()), 
                    max_norm=1.0
                )
                self.protein_optimizer.step()
            
            # 更新进度条
            pbar.update(1)
            actual_train_epochs += 1
            
            # 打印进度
            if epoch % 10 == 0:
                pbar.set_description(f"Train Epoch {epoch:03d}: Cell Loss={cell_total_loss.item():.4f}, Protein Loss={protein_total_loss.item():.4f}")
                print(f'\nTrain Epoch {epoch:03d}: '
                      f'Cell Total Loss = {cell_total_loss.item():.4f}, '
                      f'Cell Recon = {loss_recon_cell.item():.4f}, '
                      f'Cluster = {cluster_loss.item():.4f}, '
                      f'Protein Total Loss = {protein_total_loss.item():.4f}, '
                      f'Protein Recon = {loss_recon_protein.item():.4f}, '
                      f'GO = {go_loss.item():.4f}, '
                      f'Gamma = {effective_gamma:.4f}')
                print(f'Stopping status - Cell: {cell_stopped}, Protein: {protein_stopped}')
            
            # 细胞路径收敛判断
            if not cell_stopped:
                if cell_total_loss.item() < best_cell_loss - self.convergence_threshold:
                    best_cell_loss = cell_total_loss.item()
                    no_improvement_cell = 0
                else:
                    no_improvement_cell += 1
                    if no_improvement_cell >= self.convergence_patience:
                        print(f"\nCell path early stopping at epoch {epoch} - loss converged")
                        cell_stopped = True
            
            # 蛋白路径收敛判断
            if not protein_stopped:
                if protein_total_loss.item() < best_protein_loss - self.convergence_threshold:
                    best_protein_loss = protein_total_loss.item()
                    no_improvement_protein = 0
                else:
                    no_improvement_protein += 1
                    if no_improvement_protein >= self.convergence_patience:
                        print(f"\nProtein path early stopping at epoch {epoch} - loss converged")
                        protein_stopped = True
            
            # 如果两条路径都已停止，结束训练
            if cell_stopped and protein_stopped:
                print(f"\nBoth paths stopped at epoch {epoch}")
                break
        
        # 关闭进度条
        pbar.close()
        print(f"Training completed in {actual_train_epochs} epochs")
        
        # 保存最终聚类结果
        self.final_cell_embeddings = self.get_cell_embeddings()
        self.final_protein_embeddings = self.get_protein_embeddings()
        self.final_cluster_labels = self.get_cluster_labels()
        
        # 计算最终聚类指标
        self.final_nmi = normalized_mutual_info_score(self.true_labels, self.final_cluster_labels)
        self.final_ari = adjusted_rand_score(self.true_labels, self.final_cluster_labels)
        
        # # 比较预训练后和训练后的聚类指标
        # print("\n=== Clustering Performance Comparison ===")
        # print(f"After Pretraining: NMI={self.pretrain_nmi:.4f}, ARI={self.pretrain_ari:.4f}")
        # print(f"After Full Training: NMI={self.final_nmi:.4f}, ARI={self.final_ari:.4f}")
        
        # # 计算差异
        # nmi_diff = self.final_nmi - self.pretrain_nmi
        # ari_diff = self.final_ari - self.pretrain_ari
        # print(f"Improvement: NMI={nmi_diff:.4f}, ARI={ari_diff:.4f}")

        # 输出聚类指标
        print("\n=== Clustering Performance ===")
        print(f"After Full Training: NMI={self.final_nmi:.4f}, ARI={self.final_ari:.4f}")
        
        return actual_train_epochs

    def get_cell_embeddings(self) -> np.ndarray:
        """获取细胞嵌入 - 使用高方差蛋白"""
        self.model.eval()
        with torch.no_grad():
            cell_embed = self.model.get_cell_embedding(self.data.x)
        return cell_embed.cpu().detach().numpy()
    
    def get_protein_embeddings(self) -> np.ndarray:
        """获取蛋白嵌入"""
        self.model.eval()
        with torch.no_grad():
            protein_embed = self.model.get_protein_embedding(self.data.x)
        return protein_embed.cpu().detach().numpy()
    
    def get_cluster_labels(self) -> np.ndarray:
        """获取细胞聚类标签（使用软分配）"""
        self.model.eval()
        with torch.no_grad():
            cell_embed = self.get_cell_embeddings()
            q = self.model.cluster_layer.soft_assign(
                torch.tensor(cell_embed, dtype=torch.float32).to(self.device)
            )
            return torch.argmax(q, dim=1).cpu().numpy()

    def plot_losses(self, save_dir):
        """绘制并保存损失曲线图"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 预训练损失曲线
        plt.figure(figsize=(15, 15))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.pretrain_loss_history['total'], label='Total Loss')
        plt.plot(self.pretrain_loss_history['cell_recon'], label='Cell Recon Loss')
        plt.plot(self.pretrain_loss_history['protein_recon'], label='Protein Recon Loss')
        plt.plot(self.pretrain_loss_history['contrastive'], label='Contrastive Loss')
        plt.title('Pretraining Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 细胞路径训练损失曲线
        plt.subplot(3, 1, 2)
        plt.plot(self.train_loss_history['cell_total'], label='Cell Total Loss')
        plt.plot(self.train_loss_history['cell_recon'], label='Cell Recon Loss')
        plt.plot(self.train_loss_history['cluster'], label='Cluster Loss')
        plt.title('Cell Path Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 蛋白路径训练损失曲线
        plt.subplot(3, 1, 3)
        plt.plot(self.train_loss_history['protein_total'], label='Protein Total Loss')
        plt.plot(self.train_loss_history['protein_recon'], label='Protein Recon Loss')
        plt.plot(self.train_loss_history['go'], label='GO Loss')
        plt.title('Protein Path Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()
        
        print(f"Loss curves saved to {save_dir}")

    # def save_clustering_comparison(self, save_dir):
    #     """保存预训练后与正式训练后的聚类指标比较"""
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 创建比较数据框
    #     comparison_df = pd.DataFrame({
    #         'Stage': ['After Pretraining', 'After Full Training'],
    #         'NMI': [self.pretrain_nmi, self.final_nmi],
    #         'ARI': [self.pretrain_ari, self.final_ari]
    #     })
        
    #     # 保存为CSV
    #     comparison_df.to_csv(os.path.join(save_dir, 'clustering_comparison.csv'), index=False)
        
    #     # 绘制比较条形图
    #     plt.figure(figsize=(12, 6))
        
    #     # NMI比较
    #     plt.subplot(1, 2, 1)
    #     sns.barplot(x='Stage', y='NMI', data=comparison_df)
    #     plt.title('Normalized Mutual Information Comparison')
    #     plt.ylim(0, 1)
    #     plt.xticks(rotation=45)
        
    #     # ARI比较
    #     plt.subplot(1, 2, 2)
    #     sns.barplot(x='Stage', y='ARI', data=comparison_df)
    #     plt.title('Adjusted Rand Index Comparison')
    #     plt.ylim(0, 1)
    #     plt.xticks(rotation=45)
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, 'clustering_comparison.png'))
    #     plt.close()
        
    #     print(f"Clustering comparison saved to {save_dir}")