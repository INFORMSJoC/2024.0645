# src/models/third_party/openai_client.py
import openai
from openai import AzureOpenAI
from cryptography.fernet import Fernet
import hashlib
import hmac

class EnterpriseOpenAIClient:
    """OpenAI企业级安全客户端，支持Azure和原生端点"""
    
    def __init__(self, config: Dict):
        self.mode = config.get('mode', 'azure')
        self.encryption_key = Fernet(config['encryption_key'])
        self.message_hasher = hmac.new(config['hmac_key'], digestmod=hashlib.sha256)
        
        if self.mode == 'azure':
            self.client = AzureOpenAI(
                api_key=self._decrypt(config['azure_key']),
                api_version="2024-02-01",
                azure_endpoint=config['endpoint']
            )
        else:
            self.client = openai.OpenAI(
                api_key=self._decrypt(config['openai_key']),
                base_url=config.get('base_url')
            )
            
        self.rate_limiter = TokenBucketLimiter(
            rate=config.get('rate_limit', 500),
            capacity=config.get('burst_capacity', 1000)
        )

    @audit_log(action="OPENAI_API_CALL")
    @validate_input(schema=OPENAI_SCHEMA)
    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """安全增强的聊天补全接口"""
        
        self.rate_limiter.consume(1)
        hashed = self._compute_hmac(str(messages))
        
        response = self.client.chat.completions.create(
            messages=self._sanitize_messages(messages),
            model=kwargs.get('model', 'gpt-4-turbo'),
            temperature=kwargs.get('temperature', 0.7),
            **self._apply_security_settings(kwargs)
        )
        
        return {
            "content": response.choices[0].message.content,
            "audit_hash": hashed,
            "compliance_check": self._check_compliance(response)
        }
    
    def _sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
        """输入净化处理"""
        return [
            {
                "role": msg["role"],
                "content": self._redact_pii(msg["content"])
            }
            for msg in messages
        ]
    
    def _apply_security_settings(self, kwargs: Dict) -> Dict:
        """应用企业安全策略"""
        return {
            **kwargs,
            "content_filter": True,
            "data_governance": {
                "customer_id": self.config["customer_id"],
                "data_retention": "30d"
            }
        }

