# src/models/third_party/qwen_safety.py
import re
from typing import List, Dict, Optional
import torch
from transformers import pipeline
import ahocorasick
import numpy as np
from scipy.special import expit

class QwenSafetyGuard:
    """ͨ��ǧ�ʰ�ȫ����ϵͳ�����ɶ༶���ݹ���"""
    
    def __init__(self, config: dict):
        # ��ʼ�����д�AC�Զ���
        self.sensitive_trie = ahocorasick.Automaton()
        with open(config["sensitive_words_path"]) as f:
            for word in f.read().splitlines():
                self.sensitive_trie.add_word(word, word)
        self.sensitive_trie.make_automaton()
        
        # ��ʼ����ȫ����ģ��
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config["safety_classifier"],
            device=config.get("device", 0)
        )
        
        # ƫ�����ģ��
        self.bias_detector = torch.jit.load(config["bias_detector_jit"])
        
        # ��ȫ��ֵ����
        self.thresholds = {
            "violence": 0.85,
            "hate_speech": 0.92,
            "sexual": 0.9,
            "bias": 0.75
        }
        
    def secure_generate(self, prompt: str, generation_func: callable) -> str:
        """��ȫ��ǿ����������"""
        # ���뾻��
        clean_prompt = self._sanitize_input(prompt)
        
        # ʵʱ��ȫ���
        safety_check = self._analyze_safety(clean_prompt)
        if safety_check["max_score"] > self.thresholds[safety_check["label"]]:
            raise ContentViolationError(f"��⵽{safety_check['label']}����")
        
        # ��ȫ����
        output = generation_func(clean_prompt)
        
        # �������
        safe_output = self._postprocess_output(output)
        return safe_output
    
    def _sanitize_input(self, text: str) -> str:
        """���뾻������"""
        # ɾ�����д�
        text = self._filter_sensitive_words(text)
        # ��׼���ı�
        text = self._normalize_text(text)
        return text
    
    def _filter_sensitive_words(self, text: str) -> str:
        """ʹ��AC�Զ�����Ч�������д�"""
        found = set()
        for _, word in self.sensitive_trie.iter(text):
            found.add(word)
        for word in found:
            text = text.replace(word, "*" * len(word))
        return text
    
    def _analyze_safety(self, text: str) -> dict:
        """��ά�Ȱ�ȫ����"""
        # ��ȫ����
        safety_result = self.classifier(
            text,
            candidate_labels=list(self.thresholds.keys()),
            hypothesis_template="�����ݰ���{}"
        )
        
        # ƫ�����
        bias_score = self.bias_detector(
            torch.tensor(self._text_to_vector(text))
        ).item()
        
        return {
            "label": safety_result["labels"][0],
            "scores": safety_result["scores"],
            "max_score": max(safety_result["scores"]),
            "bias_score": bias_score
        }
    
    def _postprocess_output(self, text: str) -> str:
        """���������Ϲ���"""
        # ���дʶ��ι���
        clean_text = self._filter_sensitive_words(text)
        
        # ����һ���Լ��
        if self._check_semantic_integrity(text, clean_text):
            return clean_text
        return "[�����Ѹ��ݰ�ȫ�����Ƴ�]"
    
    def _check_semantic_integrity(self, orig: str, processed: str) -> bool:
        """����һ������֤"""
        orig_vec = self._text_to_vector(orig)
        proc_vec = self._text_to_vector(processed)
        similarity = np.dot(orig_vec, proc_vec) / (
            np.linalg.norm(orig_vec) * np.linalg.norm(proc_vec)
        )
        return similarity > 0.6

class ContentViolationError(Exception):
    """���ݰ�ȫΥ���쳣"""
    pass

