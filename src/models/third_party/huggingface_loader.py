# src/models/third_party/huggingface_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import infer_auto_device_map, dispatch_model
import torch
import logging

class OptimizedModelLoader:
    """高效本地模型加载器，支持多GPU优化和量化"""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    def load_model(self, model_name: str) -> tuple:
        """优化模型加载流程，支持大模型分片"""
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.get("trust_remote_code", False)
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.quant_config if self.config["quantize"] else None,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            
            if self.config.get("device_map_optimization"):
                device_map = infer_auto_device_map(
                    model,
                    max_memory={i: "24GiB" for i in range(torch.cuda.device_count())}
                )
                model = dispatch_model(model, device_map=device_map)
                
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise ModelLoadingError("Failed to load model") from e

class ModelLoadingError(Exception):
    """自定义模型加载异常"""
    pass

