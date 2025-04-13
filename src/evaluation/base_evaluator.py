# src/evaluation/base_evaluator.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List

class BaseEvaluator(ABC):
    """评估器抽象基类"""
    def __init__(self, config_path: str = "./evaluator.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    @abstractmethod
    def evaluate(self, 
                model_outputs: List[str],
                ground_truth: List[str],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """核心评估逻辑"""
        pass
    
    def _validate_inputs(self, model_outputs: List[str], ground_truth: List[str]):
        """输入合法性验证"""
        if len(model_outputs) != len(ground_truth):
            raise ValueError("Model outputs and ground truth must have the same length")
    
    def _calculate_basic_metrics(self, 
                               model_outputs: List[str], 
                               ground_truth: List[str]) -> Dict[str, float]:
        """计算基础评估指标"""
        accuracy = np.mean(np.array(model_outputs) == np.array(ground_truth))
        return {'accuracy': accuracy}
