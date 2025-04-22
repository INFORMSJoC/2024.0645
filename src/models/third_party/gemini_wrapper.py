# src/models/third_party/gemini_wrapper.py
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any
import logging
import json

class GeminiProClient:
    """Gemini客户端，支持多模态输入和高级安全控制"""
    
    def __init__(self, api_key: str, project_id: str):
        self.client = genai.configure(
            api_key=api_key,
            transport='grpc',
            client_options={
                'api_endpoint': f'generativelanguage.googleapis.com',
                'project': project_id
            }
        )
        self.safety_settings = self._load_safety_config()
        self.cache = TTLCache(maxsize=1000, ttl=300)
        
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
    async def generate(
        self,
        prompt: str,
        media: List[bytes] = None,
        generation_config: Dict[str, Any] = None
    ) -> Dict:
        """处理多模态输入并生成响应，带自动重试机制"""
        
        cache_key = self._generate_cache_key(prompt, media)
        if cached := self.cache.get(cache_key):
            return cached
        
        request = self._build_request(prompt, media, generation_config)
        
        try:
            response = await self.client.generate_content_async(**request)
            result = self._parse_response(response)
            self.cache[cache_key] = result
            return result
        except Exception as e:
            logging.error(f"Gemini API调用失败: {str(e)}")
            raise
    
    def _build_request(self, prompt, media, config):
        """构造符合企业级要求的请求体"""
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        *[{"inline_data": {"mime_type": self._detect_mime_type(m), "data": m}} for m in media]
                    ]
                }
            ],
            "safety_settings": self.safety_settings,
            "generation_config": {
                "temperature": config.get('temperature', 0.7),
                "top_p": config.get('top_p', 0.95),
                "max_output_tokens": config.get('max_tokens', 2048),
                "candidate_count": config.get('n', 1)
            }
        }
    
    def _load_safety_config(self) -> Dict:
        """加载企业级安全配置"""
        return {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE"
        }

