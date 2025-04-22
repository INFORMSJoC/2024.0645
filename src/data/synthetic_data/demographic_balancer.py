# src/data/synthetic_data/demographic_balancer.py
import numpy as np
import torch
from torch import nn, optim
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple
from collections import defaultdict

class DemographicBalancer:
    """多策略人口统计平衡器，集成对抗训练与密度估计"""
    
    def __init__(self, target_dist: Dict[str, Dict], device: str = 'cuda'):
        self.target_dist = target_dist
        self.device = device
        
        # 初始化核密度估计器
        self.kde_models = self._init_kde_models(target_dist)
        
        # 对抗性重加权网络
        self.adversary = AdversarialWeightNetwork(
            feature_dims=self._get_feature_dims(),
            hidden_dim=64
        ).to(device)
        
        # 优化器配置
        self.optimizer = optim.AdamW(
            self.adversary.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def balance_batch(self, samples: List[Dict]) -> List[Dict]:
        """执行多阶段平衡处理"""
        # 阶段1：对抗性重加权
        weights = self.adversarial_reweight(samples)
        
        # 阶段2：基于密度的过采样
        augmented = self.density_based_oversample(samples, weights)
        
        # 阶段3：最近邻清洗
        return self.nn_cleanse(augmented)

    def adversarial_reweight(self, samples: List[Dict]) -> np.ndarray:
        """对抗性样本重加权"""
        feature_vectors = self._extract_features(samples)
        self.adversary.train()
        
        for _ in range(3):  # 快速对抗训练迭代
            self.optimizer.zero_grad()
            
            # 计算当前分布
            pred_weights = self.adversary(feature_vectors)
            current_dist = self._compute_distribution(feature_vectors, pred_weights)
            
            # 计算KL散度损失
            loss = self._compute_kl_loss(current_dist)
            loss.backward()
            
            # 梯度裁剪与更新
            torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), 1.0)
            self.optimizer.step()
        
        with torch.no_grad():
            return self.adversary(feature_vectors).cpu().numpy()

    def density_based_oversample(self, samples: List[Dict], weights: np.ndarray) -> List[Dict]:
        """基于核密度估计的智能过采样"""
        feature_vectors = self._extract_features(samples)
        kde = gaussian_kde(feature_vectors.T, weights=weights)
        
        # 生成合成样本
        synthetic = kde.resample(len(samples) // 2).T
        return samples + self._create_synthetic_samples(synthetic)

    def nn_cleanse(self, samples: List[Dict]) -> List[Dict]:
        """最近邻去噪清洗"""
        features = self._extract_features(samples)
        nbrs = NearestNeighbors(n_neighbors=5).fit(features)
        distances, _ = nbrs.kneighbors(features)
        
        # 移除离群样本
        keep_mask = distances.mean(axis=1) < np.quantile(distances, 0.9)
        return [s for s, m in zip(samples, keep_mask) if m]

    def _compute_kl_loss(self, current_dist: Dict) -> torch.Tensor:
        """计算多维度KL损失"""
        total_loss = 0
        for dim, dist in current_dist.items():
            target = torch.tensor(self.target_dist[dim], device=self.device)
            total_loss += self.criterion(torch.log(dist + 1e-8), target)
        return total_loss

class AdversarialWeightNetwork(nn.Module):
    """多任务对抗权重预测网络"""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.embed_layers = nn.ModuleDict({
            name: nn.Embedding(num_classes, 8)
            for name, num_classes in feature_dims.items()
        })
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Linear(sum(8 for _ in feature_dims), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 多任务预测头
        self.heads = nn.ModuleDict({
            'weight': nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'bias_detect': nn.Sequential(
                nn.Linear(hidden_dim, len(feature_dims)),
                nn.Softmax(dim=1)
            )
        })

    def forward(self, x: Dict) -> torch.Tensor:
        # 嵌入层处理
        embeddings = []
        for name, values in x.items():
            emb = self.embed_layers[name](values)
            embeddings.append(emb)
        
        # 特征融合
        fused = self.fusion(torch.cat(embeddings, dim=1))
        
        # 多任务输出
        return {
            'weight': self.heads['weight'](fused).squeeze(),
            'bias_probs': self.heads['bias_detect'](fused)
        }
