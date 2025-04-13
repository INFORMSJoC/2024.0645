# src/models/third_party/qwen_safety.py
import re
from typing import List, Dict, Optional
import torch
from transformers import pipeline
import ahocorasick
import numpy as np
from scipy.special import expit

class QwenSafetyGuard:
    """通义千问安全防护系统，集成多级内容过滤"""
    
    def __init__(self, config: dict):
        # 初始化敏感词AC自动机
        self.sensitive_trie = ahocorasick.Automaton()
        with open(config["sensitive_words_path"]) as f:
            for word in f.read().splitlines():
                self.sensitive_trie.add_word(word, word)
        self.sensitive_trie.make_automaton()
        
        # 初始化安全分类模型
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config["safety_classifier"],
            device=config.get("device", 0)
        )
        
        # 偏见检测模型
        self.bias_detector = torch.jit.load(config["bias_detector_jit"])
        
        # 安全阈值配置
        self.thresholds = {
            "violence": 0.85,
            "hate_speech": 0.92,
            "sexual": 0.9,
            "bias": 0.75
        }
        
    def secure_generate(self, prompt: str, generation_func: callable) -> str:
        """安全增强的生成流程"""
        # 输入净化
        clean_prompt = self._sanitize_input(prompt)
        
        # 实时安全监测
        safety_check = self._analyze_safety(clean_prompt)
        if safety_check["max_score"] > self.thresholds[safety_check["label"]]:
            raise ContentViolationError(f"检测到{safety_check['label']}内容")
        
        # 安全生成
        output = generation_func(clean_prompt)
        
        # 输出后处理
        safe_output = self._postprocess_output(output)
        return safe_output
    
    def _sanitize_input(self, text: str) -> str:
        """输入净化处理"""
        # 删除敏感词
        text = self._filter_sensitive_words(text)
        # 标准化文本
        text = self._normalize_text(text)
        return text
    
    def _filter_sensitive_words(self, text: str) -> str:
        """使用AC自动机高效过滤敏感词"""
        found = set()
        for _, word in self.sensitive_trie.iter(text):
            found.add(word)
        for word in found:
            text = text.replace(word, "*" * len(word))
        return text
    
    def _analyze_safety(self, text: str) -> dict:
        """多维度安全分析"""
        # 安全分类
        safety_result = self.classifier(
            text,
            candidate_labels=list(self.thresholds.keys()),
            hypothesis_template="该内容包含{}"
        )
        
        # 偏见检测
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
        """输出后处理与合规检查"""
        # 敏感词二次过滤
        clean_text = self._filter_sensitive_words(text)
        
        # 语义一致性检查
        if self._check_semantic_integrity(text, clean_text):
            return clean_text
        return "[内容已根据安全策略移除]"
    
    def _check_semantic_integrity(self, orig: str, processed: str) -> bool:
        """语义一致性验证"""
        orig_vec = self._text_to_vector(orig)
        proc_vec = self._text_to_vector(processed)
        similarity = np.dot(orig_vec, proc_vec) / (
            np.linalg.norm(orig_vec) * np.linalg.norm(proc_vec)
        )
        return similarity > 0.6

class ContentViolationError(Exception):
    """内容安全违规异常"""
    pass

