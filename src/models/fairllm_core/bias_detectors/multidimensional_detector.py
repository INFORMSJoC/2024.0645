# src/models/fairllm_core/bias_detectors/multidimensional_detector.py
import re
from functools import lru_cache
import numpy as np
import torch
from transformers import pipeline
import spacy
from scipy.stats import entropy

class BiasDetectorPipeline:
    """��ά��ƫ������ܣ����ɹ���+ML��Ϸ���"""
    
    def __init__(self, config: Dict):
        self.nlp = spacy.load("en_core_web_lg")
        self.ml_detector = pipeline(
            "zero-shot-classification",
            model=config['bias_classifier_model'],
            device=config.get('device', -1)
        )
        
        # �����Զ���ƫ���ʵ�
        with open(config['bias_lexicon_path']) as f:
            self.bias_lexicon = set(line.strip() for line in f)
            
        # ��ʼ����ƽ��ָ�������
        self.metric_calculator = FairnessMetrics(
            protected_attributes=config['protected_attributes']
        )

    @lru_cache(maxsize=1000)
    def detect(self, text: str, context: Dict = None) -> Dict:
        """ִ�ж�㼶ƫ�����"""
        
        # ���Բ������
        doc = self.nlp(text)
        rule_based = self._rule_based_analysis(doc)
        
        # ����ѧϰ���
        ml_result = self.ml_detector(
            text,
            candidate_labels=["age_bias", "gender_bias", "racial_bias"],
            hypothesis_template="This text contains {}."
        )
        
        # �����Ĺ�������
        contextual = self._contextual_analysis(text, context)
        
        # ��ƽ��ָ�����
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
        """���ڹ��������ѧģʽ�ļ��"""
        results = {'lexical': [], 'syntactic': []}
        
        # �ʻ������
        for token in doc:
            if token.lemma_ in self.bias_lexicon:
                results['lexical'].append({
                    'token': token.text,
                    'lemma': token.lemma_,
                    'position': token.i
                })
                
        # �䷨ģʽ���
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
        """�����Ĺ�����ƫ������"""
        # ʵ�������е������Ĺ�������㷨
        pass
