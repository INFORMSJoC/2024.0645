# src/utils/reflection_optimizers.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Any

class ReflectionOptimizer:
    """自我反思优化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def optimize(self, response: str, feedback: str) -> str:
        """根据反馈优化回复"""
        try:
            # 生成反思提示
            prompt = self._generate_reflection_prompt(response, feedback)
            
            # 生成优化后的回复
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(**inputs, max_length=200)
            optimized_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return optimized_response
        except Exception as e:
            self.logger.error(f"Reflection optimization failed: {str(e)}")
            return response

    def _generate_reflection_prompt(self, response: str, feedback: str) -> str:
        """生成反思提示"""
        return f"Original response: {response}\nFeedback: {feedback}\nOptimized response:"

class CollaborativeReflectionOptimizer(ReflectionOptimizer):
    """协作反思优化器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.collaborative_models = config.get('collaborative_models', ['gpt-3.5-turbo', 'gpt-4'])

    def optimize(self, response: str, feedbacks: List[str]) -> str:
        """根据多方反馈优化回复"""
        try:
            # 生成协作反思提示
            prompt = self._generate_collaborative_prompt(response, feedbacks)
            
            # 使用协作模型生成优化后的回复
            optimized_response = ""
            for model_name in self.collaborative_models:
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                outputs = self.model.generate(**inputs, max_length=200)
                current_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                optimized_response += current_response + " "
            
            return optimized_response.strip()
        except Exception as e:
            self.logger.error(f"Collaborative reflection optimization failed: {str(e)}")
            return response

    def _generate_collaborative_prompt(self, response: str, feedbacks: List[str]) -> str:
        """生成协作反思提示"""
        feedback_str = "\n".join([f"Feedback {i+1}: {feedback}" for i, feedback in enumerate(feedbacks)])
        return f"Original response: {response}\n{feedback_str}\nOptimized response:"