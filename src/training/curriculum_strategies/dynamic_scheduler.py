# src/training/curriculum_strategies/dynamic_scheduler.py
import numpy as np
from scipy.special import expit
from typing import Dict, List
import torch

class DynamicCurriculumScheduler:
    """����Ӧ�γ�ѧϰ������������ģ�ͱ��ֶ�̬����ѵ���Ѷ�"""
    
    def __init__(self, config: dict):
        self.difficulty_metric = config['difficulty_metric']
        self.history_window = config['history_window']
        self.momentum = config['momentum']
        self.current_level = 0.0
        self.history = []
        
        # ��ʼ���Ѷ�������
        self.evaluator = DifficultyEvaluator(
            config['evaluator_config'],
            config['protected_attributes']
        )
        
        # �γ̽׶�����
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
        """���ݵ�ǰ����ָ����¿γ̽���"""
        # ���㵱ǰ�Ѷ�����
        difficulty_score = self.evaluator.evaluate(
            batch_metrics['predictions'],
            batch_metrics['targets']
        )
        
        # ������ʷ��¼
        self.history.append(difficulty_score)
        if len(self.history) > self.history_window:
            self.history.pop(0)
            
        # ָ���ƶ�ƽ��
        avg_score = sum(
            score * (self.momentum ** i) 
            for i, score in enumerate(reversed(self.history))
        ) / sum(self.momentum ** i for i in range(len(self.history)))
        
        # ȷ����ǰ�׶�
        current_stage = next(
            (s for s in reversed(self.curriculum_stages) if avg_score >= s['threshold']),
            self.curriculum_stages[0]
        )
        
        # ����ѧϰ��
        self._adjust_learning_rate(current_stage['lr_multiplier'])
        
        return
