# src/evaluation/fairness_metrics/statistical_parity.py
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
import yaml
from sklearn.utils import resample

@dataclass(frozen=True)
class DemographicGroup:
    """定义人群分组的特征"""
    age_range: Tuple[int, int]
    protected_attributes: Dict[str, float]

class BaseFairnessMetric(ABC):
    """公平性指标抽象基类"""
    def __init__(self, config_path: str = "../metrics_in.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    @abstractmethod
    def calculate(self, 
                 predictions: np.ndarray,
                 demographic_groups: List[DemographicGroup]) -> Dict[str, float]:
        """核心指标计算逻辑"""
        pass

class StatisticalParityAnalyzer(BaseFairnessMetric):
    """动态多维度统计均等分析器（支持年龄分层抽样与置信区间估计）"""
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 alpha: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self._validate_parameters()
    
    def _validate_parameters(self):
        """参数合法性验证（符合IJOC评审标准）"""
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0,1), got {self.alpha}")
    
    def _stratified_bootstrap(self, 
                             predictions: np.ndarray, 
                             groups: List[DemographicGroup]) -> np.ndarray:
        """分层抽样实现"""
        resampled = []
        for group in groups:
            group_mask = (predictions >= group.age_range[0]) & (predictions <= group.age_range[1])
            group_samples = predictions[group_mask]
            if len(group_samples) > 0:
                resampled.append(resample(group_samples, n_samples=len(group_samples)))
        return np.concatenate(resampled)
    
    def _calculate_disparity(self, group_probs: List[float]) -> Dict[str, float]:
        """基于基尼系数与JS散度的复合差异指标"""
        gini = 1 - sum(p**2 for p in group_probs)
        mean_prob = np.mean(group_probs)
        js_div = stats.entropy(group_probs, qk=[mean_prob]*len(group_probs)) / np.log(2)
        return {'gini_coefficient': gini, 'js_divergence': js_div}
    
    def calculate(self,
                predictions: np.ndarray,
                demographic_groups: List[DemographicGroup]) -> Dict[str, float]:
        """执行蒙特卡洛重采样与统计推断"""
        bootstrap_results = []
        for _ in range(self.n_bootstrap):
            resampled = self._stratified_bootstrap(predictions, demographic_groups)
            group_probs = [np.mean(resampled == group.age_range[0]) for group in demographic_groups]
            bootstrap_results.append(self._calculate_disparity(group_probs))
        
        # 计算置信区间
        results = {
            'mean_gini': np.mean([x['gini_coefficient'] for x in bootstrap_results]),
            'ci_gini': stats.t.interval(1-self.alpha, len(bootstrap_results)-1,
                                      loc=np.mean([x['gini_coefficient'] for x in bootstrap_results]),
                                      scale=stats.sem([x['gini_coefficient'] for x in bootstrap_results])),
            'mean_js': np.mean([x['js_divergence'] for x in bootstrap_results]),
            'ci_js': stats.t.interval(1-self.alpha, len(bootstrap_results)-1,
                                    loc=np.mean([x['js_divergence'] for x in bootstrap_results]),
                                    scale=stats.sem([x['js_divergence'] for x in bootstrap_results]))
        }
        return results
