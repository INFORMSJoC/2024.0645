# src/evaluation/bias_probing/latent_bias_detector.py
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import DBSCAN
from typing import Dict, Any
import scipy.stats as stats

class LatentBiasVisualizer:
    """Ǳ��ƫ���ռ���ӻ�ϵͳ��֧����չ��ģ̬����������"""
    def __init__(self,
                 reduction_method: str = 'umap',
                 clustering_method: str = 'dbscan'):
        self.reduction_method = reduction_method
        self.clustering_method = clustering_method
        self._init_models()
    
    def _init_models(self):
        """��ʼ����ά�����ģ��"""
        self.reducer = UMAP(n_components=3) if self.reduction_method == 'umap' \
                      else PCA(n_components=3)
        self.clusterer = DBSCAN(eps=0.5) if self.clustering_method == 'dbscan' \
                        else KMeans(n_clusters=3)
    
    def analyze(self,
              embeddings: np.ndarray,
              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ִ��Ǳ�ڿռ�ƫ������������ͼ2-3��ʵ�֣�"""
        reduced = self.reducer.fit_transform(embeddings)
        clusters = self.clusterer.fit_predict(reduced)
        
        # ��������е�����ֲ�ƫ��
        cluster_bias = {}
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            age_dist = metadata['age'][mask].value_counts(normalize=True)
            cluster_bias[cluster_id] = {
                'entropy': stats.entropy(age_dist),
                'dominant_age_group': age_dist.idxmax()
            }
        
        return {
            'reduced_embeddings': reduced,
            'cluster_analysis': cluster_bias,
            'bias_indicator': self._compute_bias_index(cluster_bias)
        }
    
    def _compute_bias_index(self, cluster_data: Dict) -> float:
        """����Ǳ��ƫ���ۺ�ָ��"""
        bias_scores = []
        for cluster in cluster_data.values():
            entropy = cluster['entropy']
            bias_score = 1 - entropy / np.log(len(cluster['dominant_age_group']))
            bias_scores.append(bias_score)
        return np.mean(bias_scores)
