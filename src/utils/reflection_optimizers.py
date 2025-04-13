# src/utils/reflection_optimizers.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Any

class ReflectionOptimizer:
    """���ҷ�˼�Ż���"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def optimize(self, response: str, feedback: str) -> str:
        """���ݷ����Ż��ظ�"""
        try:
            # ���ɷ�˼��ʾ
            prompt = self._generate_reflection_prompt(response, feedback)
            
            # �����Ż���Ļظ�
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(**inputs, max_length=200)
            optimized_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return optimized_response
        except Exception as e:
            self.logger.error(f"Reflection optimization failed: {str(e)}")
            return response

    def _generate_reflection_prompt(self, response: str, feedback: str) -> str:
        """���ɷ�˼��ʾ"""
        return f"Original response: {response}\nFeedback: {feedback}\nOptimized response:"

class CollaborativeReflectionOptimizer(ReflectionOptimizer):
    """Э����˼�Ż���"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.collaborative_models = config.get('collaborative_models', ['gpt-3.5-turbo', 'gpt-4'])

    def optimize(self, response: str, feedbacks: List[str]) -> str:
        """���ݶ෽�����Ż��ظ�"""
        try:
            # ����Э����˼��ʾ
            prompt = self._generate_collaborative_prompt(response, feedbacks)
            
            # ʹ��Э��ģ�������Ż���Ļظ�
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
        """����Э����˼��ʾ"""
        feedback_str = "\n".join([f"Feedback {i+1}: {feedback}" for i, feedback in enumerate(feedbacks)])
        return f"Original response: {response}\n{feedback_str}\nOptimized response:"