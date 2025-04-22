# src/models/fairllm_core/bias_detectors/multidimensional_detector.py
import re
from functools import lru_cache
import numpy as np
import torch
from transformers import pipeline
import spacy
from scipy.stats import entropy

class BiasDetectorPipeline:
    """多维度偏见检测框架，集成规则+ML混合方法"""
    
    def __init__(self, config: Dict):
        self.nlp = spacy.load("en_core_web_lg")
        self.ml_detector = pipeline(
            "zero-shot-classification",
            model=config['bias_classifier_model'],
            device=config.get('device', -1)
        )
        
        # 加载自定义偏见词典
        with open(config['bias_lexicon_path']) as f:
            self.bias_lexicon = set(line.strip() for line in f)
            
        # 初始化公平性指标计算器
        self.metric_calculator = FairnessMetrics(
            protected_attributes=config['protected_attributes']
        )

    @lru_cache(maxsize=1000)
    def detect(self, text: str, context: Dict = None) -> Dict:
        """执行多层级偏见检测"""
        
        # 语言层面分析
        doc = self.nlp(text)
        rule_based = self._rule_based_analysis(doc)
        
        # 机器学习检测
        ml_result = self.ml_detector(
            text,
            candidate_labels=["age_bias", "gender_bias", "racial_bias"],
            hypothesis_template="This text contains {}."
        )
        
        # 上下文关联分析
        contextual = self._contextual_analysis(text, context)
        
        # 公平性指标计算
        fairness_metrics = self.metric_calculator.calculate(
            text, 
            context.get('demographic_info')
        )
        
        return {
            'rule_based': rule_based,
            'ml_scores': ml_result['scores'],
            'contextual_analysis': contextual,
            'fairness_metrics': fairness_metrics
        }

    def _rule_based_analysis(self, doc) -> Dict:
        """基于规则和语言学模式的检测"""
        results = {'lexical': [], 'syntactic': []}
        
        # 词汇层面检测
        for token in doc:
            if token.lemma_ in self.bias_lexicon:
                results['lexical'].append({
                    'token': token.text,
                    'lemma': token.lemma_,
                    'position': token.i
                })
                
        # 句法模式检测
        for sent in doc.sents:
            for noun_chunk in sent.noun_chunks:
                if any(t.dep_ == 'amod' for t in noun_chunk.root.children):
                    modifiers = [t.text for t in noun_chunk.root.children if t.dep_ == 'amod']
                    if any(m in self.bias_lexicon for m in modifiers):
                        results['syntactic'].append({
                            'phrase': noun_chunk.text,
                            'modifiers': modifiers,
                            'sent_idx': sent.start
                        })
        
        return results

    def _contextual_analysis(self, text: str, context: Dict) -> Dict:
        """上下文关联的偏见分析"""
        # 实现论文中的上下文关联检测算法
        pass
