# src/evaluation/fairness_metrics/counterfactual_fairness.py
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict
import numpy as np
from scipy.spatial.distance import mahalanobis

class CounterfactualFairnessEngine:
    """反事实公平性评估引擎"""
    def __init__(self, 
                 model: AutoModel,
                 tokenizer: AutoTokenizer,
                 age_sensitive_embeddings: Dict[str, np.ndarray]):
        self.model = model
        self.tokenizer = tokenizer
        self._setup_sensitivity_analysis(age_sensitive_embeddings)
    
    def _setup_sensitivity_analysis(self, embeddings: Dict[str, np.ndarray]):
        """构建年龄敏感词嵌入的协方差矩阵"""
        self.reference_embeddings = embeddings
        self.cov_matrix = np.cov(np.array(list(embeddings.values())).T)
        self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)
    
    def _compute_counterfactual_shift(self,
                                    original_embedding: np.ndarray,
                                    target_age_group: str) -> float:
        """计算反事实嵌入的距离"""
        target_embedding = self.reference_embeddings[target_age_group]
        return mahalanobis(original_embedding, target_embedding, self.inv_cov_matrix)
    
    def evaluate(self,
                text_samples: List[str],
                original_age_context: str,
                counterfactual_age: str) -> Dict[str, float]:
        """执行反事实公平性评估"""
        with torch.no_grad():
            inputs = self.tokenizer(text_samples, return_tensors='pt', padding=True)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        shifts = []
        for emb in embeddings:
            shift = self._compute_counterfactual_shift(emb, counterfactual_age)
            shifts.append(shift)
        
        return {
            'mean_shift': np.mean(shifts),
            'std_shift': np.std(shifts),
            'max_shift': np.max(shifts),
            'fairness_violation_rate': np.mean(np.array(shifts) > self._get_threshold())
        }
    
    def _get_threshold(self) -> float:
        """动态计算公平性阈值(与论文对应)"""
        return np.percentile(list(self.reference_embeddings.values()), self.config['percentile'])
