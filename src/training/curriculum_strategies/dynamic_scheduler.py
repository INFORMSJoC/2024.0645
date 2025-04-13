# src/training/curriculum_strategies/dynamic_scheduler.py
import numpy as np
from scipy.special import expit
from typing import Dict, List
import torch

class DynamicCurriculumScheduler:
    """自适应课程学习调度器，基于模型表现动态调整训练难度"""
    
    def __init__(self, config: dict):
        self.difficulty_metric = config['difficulty_metric']
        self.history_window = config['history_window']
        self.momentum = config['momentum']
        self.current_level = 0.0
        self.history = []
        
        # 初始化难度评估器
        self.evaluator = DifficultyEvaluator(
            config['evaluator_config'],
            config['protected_attributes']
        )
        
        # 课程阶段配置
        self.curriculum_stages = [
            {
                'threshold': 0.3,
                'data_mix': {'easy': 0.7, 'medium': 0.2, 'hard': 0.1},
                'lr_multiplier': 0.8
            },
            {
                'threshold': 0.6,
                'data_mix': {'easy': 0.3, 'medium': 0.5, 'hard': 0.2},
                'lr_multiplier': 1.0
            },
            {
                'threshold': 0.9,
                'data_mix': {'easy': 0.1, 'medium': 0.3, 'hard': 0.6},
                'lr_multiplier': 1.2
            }
        ]

    def update_curriculum(self, batch_metrics: dict) -> dict:
        """根据当前批次指标更新课程进度"""
        # 计算当前难度评分
        difficulty_score = self.evaluator.evaluate(
            batch_metrics['predictions'],
            batch_metrics['targets']
        )
        
        # 更新历史记录
        self.history.append(difficulty_score)
        if len(self.history) > self.history_window:
            self.history.pop(0)
            
        # 指数移动平均
        avg_score = sum(
            score * (self.momentum ** i) 
            for i, score in enumerate(reversed(self.history))
        ) / sum(self.momentum ** i for i in range(len(self.history)))
        
        # 确定当前阶段
        current_stage = next(
            (s for s in reversed(self.curriculum_stages) if avg_score >= s['threshold']),
            self.curriculum_stages[0]
        )
        
        # 调整学习率
        self._adjust_learning_rate(current_stage['lr_multiplier'])
        
        return
