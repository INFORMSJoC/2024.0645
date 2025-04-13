# src/utils/empathy_utils.py
import numpy as np
import torch
from transformers import pipeline
import logging

class EmpathyAnalyzer:
    """共情分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.emotion_classifier = pipeline(
            'text-classification',
            model=config.get('model_name', 'joeddav/distilbert-base-uncased-go-emotions-student'),
            device=config.get('device', 0)
        )
        self.age_groups = config.get('age_groups', ['young', 'middle', 'senior'])

    def analyze(self, text: str, age_group: str) -> Dict:
        """分析文本的情感和共情水平"""
        results = {
            'emotions': self._detect_emotions(text),
            'empathy_score': self._calculate_empathy_score(text, age_group)
        }
        return results

    def _detect_emotions(self, text: str) -> List[Dict]:
        """检测文本中的情感"""
        try:
            emotions = self.emotion_classifier(text, return_all_scores=True)
            return emotions[0]
        except Exception as e:
            self.logger.error(f"Emotion detection failed: {str(e)}")
            return []

    def _calculate_empathy_score(self, text: str, age_group: str) -> float:
        """计算共情分数"""
        # 模拟计算
        base_score = np.random.uniform(0.5, 1.0)
        age_factor = {
            'young': 1.2,
            'middle': 1.0,
            'senior': 0.8
        }.get(age_group, 1.0)
        return base_score * age_factor

    def adapt_response(self, response: str, age_group: str) -> str:
        """根据年龄群体调整回复的情感基调"""
        empathy_score = self._calculate_empathy_score(response, age_group)
        if empathy_score < 0.6:
            return self._enhance_empathy(response, age_group)
        return response

    def _enhance_empathy(self, response: str, age_group: str) -> str:
        """增强回复的共情性"""
        # 模拟增强逻辑
        empathy_phrases = {
            'young': ["I understand how you feel.", "That must be tough.", "You're not alone."],
            'middle': ["I can see why you'd feel that way.", "Let's work through this together.", "Your perspective is valuable."],
            'senior': ["I appreciate your insight.", "Your experience is important.", "Let me help you with that."]
        }
        return f"{response} {np.random.choice(empathy_phrases.get(age_group, []))}"
