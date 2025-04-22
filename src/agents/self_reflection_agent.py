# src/agents/self_reflection_agent.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
import google.generativeai as genai
from openai import OpenAI
from accelerate import dispatch_model, infer_auto_device_map

class ModelManager:
    """自适应多模型路由管理器，支持动态负载均衡"""
    
    _instance = None
    
    def __new__(cls, config: Dict):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models(config)
        return cls._instance
    
    def _init_models(self, config: Dict):
        self.models = {
            'gpt-3.5-turbo': OpenAI(api_key=config['openai_key']),
            'gemini1.0-pro-001': genai.GenerativeAI(api_key=config['gemini_key']),
            'llama2-7B-instruct': self._load_hf_model("meta-llama/Llama-2-7b-chat-hf", 8),
            'meta-llama3-8B-instruct': self._load_hf_model("meta-llama/Meta-Llama-3-8B-Instruct", 16),
            'mistral-7B-instruct-v0.2': self._load_hf_model("mistralai/Mistral-7B-Instruct-v0.2", 4),
            'qwen2-7B-instruct': self._load_hf_model("Qwen/Qwen1.5-7B-Chat", 8)
        }
        self.active_adapters = {
            'bias_mitigation': torch.nn.ModuleDict({
                model_name: AdapterLayer(1024, 4096) 
                for model_name in ['llama', 'mistral', 'qwen']
            })
        }

    def _load_hf_model(self, model_id: str, num_bits: int):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if num_bits == 16 else torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        model = dispatch_model(model, device_map=infer_auto_device_map(model))
        return {'model': model, 'tokenizer': tokenizer}

    def get_model(self, model_id: str) -> torch.nn.Module:
        return self.models[model_id]

class ReflectionChain:
    """反思链处理器，用于实现多阶段自我反思"""
    
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.reflection_steps = [
            self._generate_initial_response,
            self._detect_bias,
            self._refine_response
        ]
        self.max_retries = 3
        self.prompt_templates = {
            'init': "作为公正的AI助手，请回答以下问题：{context}",
            'detect': "分析以下回答是否存在年龄偏见：{response}",
            'refine': "基于{feedback}，请重新生成无偏见的回答：{context}"
        }

    async def execute(self, context: str, model_id: str) -> Tuple[str, List[Dict]]:
        """执行反思链"""
        reflection_log = []
        current_response = None
        
        for step in self.reflection_steps:
            for attempt in range(self.max_retries):
                try:
                    result = await step(context, current_response, model_id)
                    current_response, metadata = result
                    reflection_log.append(metadata)
                    break
                except Exception as e:
                    logging.error(f"Step {step.__name__} attempt {attempt+1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Reflection chain failed after {self.max_retries} attempts")
        
        return current_response, reflection_log

    async def _generate_initial_response(self, context: str, _, model_id: str) -> Tuple[str, Dict]:
        """生成初始回答"""
        prompt = self.prompt_templates['init'].format(context=context)
        response = await self.mm.models[model_id].chat.completions.create(
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, {'stage': 'init', 'prompt': prompt, 'raw_response': response}

    async def _detect_bias(self, context: str, response: str, model_id: str) -> Tuple[Dict, Dict]:
        """检测偏见"""
        analysis_prompt = self.prompt_templates['detect'].format(response=response)
        detection_result = await self.mm.models[model_id].chat.completions.create(
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        detection_result = detection_result.choices[0].message.content
        
        bias_metrics = self._quantify_bias(response, detection_result)
        return {'bias_detected': bias_metrics}, {
            'stage': 'detect',
            'metrics': bias_metrics,
            'analysis': detection_result
        }

    async def _refine_response(self, context: str, feedback: Dict, model_id: str) -> Tuple[str, Dict]:
        """优化回答"""
        refine_prompt = self.prompt_templates['refine'].format(
            context=context,
            feedback=feedback['analysis']
        )
        refined = await self.mm.models[model_id].chat.completions.create(
            messages=[{"role": "user", "content": refine_prompt}]
        )
        return refined.choices[0].message.content, {'stage': 'refine', 'final_response': refined}

    def _quantify_bias(self, text: str, analysis: str) -> Dict:
        """量化偏见"""
        age_terms = ['elderly', 'young', 'senior', 'junior']
        count = sum(text.lower().count(term) for term in age_terms)
        sentiment = self._sentiment_analysis(text)
        return {
            'term_frequency': count / len(text.split()),
            'sentiment_bias': abs(sentiment['positive'] - sentiment['negative']),
            'contextual_bias_score': self._contextual_bias_assessment(analysis)
        }

    def _sentiment_analysis(self, text: str) -> Dict:
        """情感分析"""
        blob = TextBlob(text)
        return {
            'positive': blob.sentiment.polarity,
            'negative': -blob.sentiment.polarity
        }

    def _contextual_bias_assessment(self, analysis: str) -> float:
        """上下文偏见评估"""
        bias_keywords = ['biased', 'stereotypical', 'unfair']
        count = sum(analysis.lower().count(term) for term in bias_keywords)
        return count / len(analysis.split())

