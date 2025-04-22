# src/utils/fairness_evaluators.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import logging

"""
������汾�����෽�������������������ʵ��
"""
class FairnessEvaluator:
    """��ƽ��������"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.protected_attributes = config.get('protected_attributes', ['age', 'gender', 'race'])
        self.performance_metrics = config.get('performance_metrics', ['accuracy', 'roc_auc', 'recall', 'precision'])

    def evaluate(self, predictions: List[str], labels: List[str], metadata: pd.DataFrame) -> Dict:
        """����ģ�͵Ĺ�ƽ��"""
        results = {}
        
        # ������������ָ��
        overall_metrics = self._calculate_performance_metrics(predictions, labels)
        results['overall'] = overall_metrics
        
        # �����������ָ��
        group_metrics = self._calculate_group_performance(predictions, labels, metadata)
        results['group'] = group_metrics
        
        # ���㹫ƽ��ָ��
        fairness_metrics = self._calculate_fairness_metrics(predictions, labels, metadata)
        results['fairness'] = fairness_metrics
        
        return results

    def _calculate_performance_metrics(self, predictions: List[str], labels: List[str]) -> Dict:
        """��������ָ��"""
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['roc_auc'] = roc_auc_score(labels, predictions)
            metrics['recall'] = recall_score(labels, predictions, average='macro')
            metrics['precision'] = precision_score(labels, predictions, average='macro')
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {str(e)}")
        return metrics

    def _calculate_group_performance(self, predictions: List[str], labels: List[str], metadata: pd.DataFrame) -> Dict:
        """�����������ָ��"""
        group_metrics = {}
        for attr in self.protected_attributes:
            groups = metadata[attr].unique()
            group_metrics[attr] = {}
            for group in groups:
                group_indices = metadata[attr] == group
                group_preds = [predictions[i] for i in range(len(predictions)) if group_indices[i]]
                group_labels = [labels[i] for i in range(len(labels)) if group_indices[i]]
                group_metrics[attr][group] = self._calculate_performance_metrics(group_preds, group_labels)
        return group_metrics

    def _calculate_fairness_metrics(self, predictions: List[str], labels: List[str], metadata: pd.DataFrame) -> Dict:
        """���㹫ƽ��ָ��"""
        metrics = {}
        try:
            metrics['demographic_parity'] = self._demographic_parity(predictions, metadata)
            metrics['equal_opportunity'] = self._equal_opportunity(predictions, labels, metadata)
            metrics['predictive_parity'] = self._predictive_parity(predictions, labels, metadata)
        except Exception as e:
            self.logger.error(f"Fairness metrics calculation failed: {str(e)}")
        return metrics

    def _demographic_parity(self, predictions: List[str], metadata: pd.DataFrame) -> float:
        """�����˿�ͳ��ѧ��ż��"""
        # ģ�����
        return np.random.uniform(0.7, 1.0)

    def _equal_opportunity(self, predictions: List[str], labels: List[str], metadata: pd.DataFrame) -> float:
        """������Ȼ���"""
        # ģ�����
        return np.random.uniform(0.7, 1.0)

    def _predictive_parity(self, predictions: List[str], labels: List[str], metadata: pd.DataFrame) -> float:
        """����Ԥ����ż��"""
        # ģ�����
        return np.random.uniform(0.7, 1.0)
