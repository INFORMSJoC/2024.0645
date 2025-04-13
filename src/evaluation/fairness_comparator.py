# src/evaluation/fairness_comparator.py
from typing import List, Dict
import numpy as np
from sklearn.metrics import pairwise_distances
from .base_evaluator import BaseEvaluator

class FairnessComparator(BaseEvaluator):
    """��ƽ��ָ��ȽϷ�����"""
    def __init__(self, 
                 metrics: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics or ['accuracy', 'fairness', 'bias_score']
    
    def compare(self, 
               models: List[Dict[str, Any]],
               dataset: Dict[str, Any]) -> Dict[str, Any]:
        """�Ƚ϶��ģ�͵Ĺ�ƽ�Ա���"""
        results = {}
        for model in models:
            model_name = model['name']
            outputs = model['outputs']
            ground_truth = dataset['ground_truth']
            metadata = dataset['metadata']
            
            eval_results = self.evaluate(outputs, ground_truth, metadata)
            results[model_name] = eval_results
        
        # ������Թ�ƽ������
        baseline = results[self.config['baseline_model']]
        comparisons = {}
        for model, result in results.items():
            if model == self.config['baseline_model']:
                continue
            comparisons[model] = {
                metric: (result[metric] - baseline[metric]) / baseline[metric]
                for metric in self.metrics
            }
        
        return {
            'model_results': results,
            'comparisons': comparisons,
            'overall_fairness_ranking': self._rank_fairness(results)
        }
    
    def _rank_fairness(self, results: Dict[str, Dict[str, float]]) -> List[str]:
        """���ڹ�ƽ��ָ���ģ�ͽ�������"""
        scores = []
        for model, metrics in results.items():
            score = np.mean([metrics[metric] for metric in self.metrics])
            scores.append((model, score))
        return [model for model, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
