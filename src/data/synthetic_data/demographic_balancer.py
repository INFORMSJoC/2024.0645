# src/data/synthetic_data/demographic_balancer.py
import numpy as np
import torch
from torch import nn, optim
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple
from collections import defaultdict

class DemographicBalancer:
    """������˿�ͳ��ƽ���������ɶԿ�ѵ�����ܶȹ���"""
    
    def __init__(self, target_dist: Dict[str, Dict], device: str = 'cuda'):
        self.target_dist = target_dist
        self.device = device
        
        # ��ʼ�����ܶȹ�����
        self.kde_models = self._init_kde_models(target_dist)
        
        # �Կ����ؼ�Ȩ����
        self.adversary = AdversarialWeightNetwork(
            feature_dims=self._get_feature_dims(),
            hidden_dim=64
        ).to(device)
        
        # �Ż�������
        self.optimizer = optim.AdamW(
            self.adversary.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def balance_batch(self, samples: List[Dict]) -> List[Dict]:
        """ִ�ж�׶�ƽ�⴦��"""
        # �׶�1���Կ����ؼ�Ȩ
        weights = self.adversarial_reweight(samples)
        
        # �׶�2�������ܶȵĹ�����
        augmented = self.density_based_oversample(samples, weights)
        
        # �׶�3���������ϴ
        return self.nn_cleanse(augmented)

    def adversarial_reweight(self, samples: List[Dict]) -> np.ndarray:
        """�Կ��������ؼ�Ȩ"""
        feature_vectors = self._extract_features(samples)
        self.adversary.train()
        
        for _ in range(3):  # ���ٶԿ�ѵ������
            self.optimizer.zero_grad()
            
            # ���㵱ǰ�ֲ�
            pred_weights = self.adversary(feature_vectors)
            current_dist = self._compute_distribution(feature_vectors, pred_weights)
            
            # ����KLɢ����ʧ
            loss = self._compute_kl_loss(current_dist)
            loss.backward()
            
            # �ݶȲü������
            torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), 1.0)
            self.optimizer.step()
        
        with torch.no_grad():
            return self.adversary(feature_vectors).cpu().numpy()

    def density_based_oversample(self, samples: List[Dict], weights: np.ndarray) -> List[Dict]:
        """���ں��ܶȹ��Ƶ����ܹ�����"""
        feature_vectors = self._extract_features(samples)
        kde = gaussian_kde(feature_vectors.T, weights=weights)
        
        # ���ɺϳ�����
        synthetic = kde.resample(len(samples) // 2).T
        return samples + self._create_synthetic_samples(synthetic)

    def nn_cleanse(self, samples: List[Dict]) -> List[Dict]:
        """�����ȥ����ϴ"""
        features = self._extract_features(samples)
        nbrs = NearestNeighbors(n_neighbors=5).fit(features)
        distances, _ = nbrs.kneighbors(features)
        
        # �Ƴ���Ⱥ����
        keep_mask = distances.mean(axis=1) < np.quantile(distances, 0.9)
        return [s for s, m in zip(samples, keep_mask) if m]

    def _compute_kl_loss(self, current_dist: Dict) -> torch.Tensor:
        """�����ά��KL��ʧ"""
        total_loss = 0
        for dim, dist in current_dist.items():
            target = torch.tensor(self.target_dist[dim], device=self.device)
            total_loss += self.criterion(torch.log(dist + 1e-8), target)
        return total_loss

class AdversarialWeightNetwork(nn.Module):
    """������Կ�Ȩ��Ԥ������"""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.embed_layers = nn.ModuleDict({
            name: nn.Embedding(num_classes, 8)
            for name, num_classes in feature_dims.items()
        })
        
        # ��߶������ں�
        self.fusion = nn.Sequential(
            nn.Linear(sum(8 for _ in feature_dims), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # ������Ԥ��ͷ
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
        # Ƕ��㴦��
        embeddings = []
        for name, values in x.items():
            emb = self.embed_layers[name](values)
            embeddings.append(emb)
        
        # �����ں�
        fused = self.fusion(torch.cat(embeddings, dim=1))
        
        # ���������
        return {
            'weight': self.heads['weight'](fused).squeeze(),
            'bias_probs': self.heads['bias_detect'](fused)
        }
