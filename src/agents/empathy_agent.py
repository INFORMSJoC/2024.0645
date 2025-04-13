# src/agents/empathy_agent.py
import re
from typing import Dict, List
import json
import numpy as np
from enum import Enum
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import torch
from torch import nn

class AgeGroup(Enum):
    YOUNG_ADULT = (18, 30)
    MIDDLE_AGED = (31, 55)
    SENIOR = (56, 100)

class PerspectiveShifter:
    """多维度共情引擎，集成语义改写和情感适配"""
    """共情视角转换器，用于将文本转换为不同年龄群体的视角"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.age_lexicon = self._load_age_lexicon()
        self.emotion_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.emotion_vectors = self._load_emotion_embeddings()
    
    def shift_perspective(self, text: str, target_age: AgeGroup) -> str:
        """将文本转换为目标年龄群体的视角"""
        doc = self.nlp(text)
        transformed = []
        
        for token in doc:
            if token.text.lower() in self.age_lexicon:
                transformed.append(self._map_age_terms(token.text, target_age))
            elif token.ent_type_ == 'AGE':
                transformed.append(self._adjust_numerical_age(token.text, target_age))
            else:
                transformed.append(self._adapt_emotional_tone(token.text, target_age))
        
        return self._postprocess(' '.join(transformed))

    def _map_age_terms(self, term: str, target_age: AgeGroup) -> str:
        """映射年龄相关术语"""
        term_mapping = {
            AgeGroup.YOUNG_ADULT: {'elderly': 'young adult', 'senior': 'early-career'},
            AgeGroup.SENIOR: {'young': 'experienced', 'junior': 'seasoned'}
        }
        return term_mapping.get(target_age, {}).get(term.lower(), term)

    def _adjust_numerical_age(self, age_str: str, target_group: AgeGroup) -> str:
        """调整数值年龄"""
        if age_str.isdigit():
            age = int(age_str)
            midpoint = (target_group.value[0] + target_group.value[1]) // 2
            return str(midpoint)
        return age_str

    def _adapt_emotional_tone(self, text: str, target_age: AgeGroup) -> str:
        """调整情感基调"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        adjustment_factor = {
            AgeGroup.YOUNG_ADULT: 0.1,
            AgeGroup.SENIOR: -0.05
        }.get(target_age, 0)
        
        adjusted = polarity + adjustment_factor
        return self._rephrase_with_sentiment(text, adjusted)

    def _rephrase_with_sentiment(self, text: str, target_polarity: float) -> str:
        """根据目标情感极性重述文本"""
        # 这里可以实现更复杂的情感重述逻辑
        return text

    def _postprocess(self, text: str) -> str:
        """后处理文本"""
        return text

class EmpathyEnhancer:
    """共情增强器，用于生成共情视角的回答"""
    
    def __init__(self):
        self.shifter = PerspectiveShifter()
        self.age_groups = [ag for ag in AgeGroup]
        self.emotion_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def enhance_response(self, context: str, response: str) -> str:
        """增强回答的共情性"""
        perspectives = {}
        
        for age_group in self.age_groups:
            modified_context = self.shifter.shift_perspective(context, age_group)
            modified_response = self.shifter.shift_perspective(response, age_group)
            perspectives[age_group] = {
                'modified_context': modified_context,
                'modified_response': modified_response,
                'empathy_score': self._calculate_empathy_score(
                    modified_context, modified_response
                )
            }
        
        best_perspective = max(
            perspectives.items(), 
            key=lambda x: x[1]['empathy_score']
        )
        return best_perspective[1]['modified_response']

    def _calculate_empathy_score(self, context: str, response: str) -> float:
        """计算共情分数"""
        context_embedding = self.emotion_model.encode(context)
        response_embedding = self.emotion_model.encode(response)
        similarity = np.dot(context_embedding, response_embedding) / (
            np.linalg.norm(context_embedding) * np.linalg.norm(response_embedding)
        )
        return similarity

    def _extract_emotional_profile(self, text: str) -> np.ndarray:
        """提取文本的情感特征"""
        return self.emotion_model.encode(text)
