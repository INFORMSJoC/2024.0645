# src/models/third_party/mistral_quant.py
import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from einops import rearrange
import logging
from typing import Optional, Tuple

class MistralQuantizer(nn.Module):
    """Mistral量化专家，集成4/8位混合精度量化与稀疏注意力"""
    
    def __init__(self, model_name: str, quant_config: dict):
        super().__init__()
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("4bit", True),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"]
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quant_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self._patch_attention()
        
        # 量化感知训练参数
        self.quant_scale = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name, _ in self.model.named_linear_layers()
        })
        
    def _patch_attention(self):
        """替换原始注意力层为量化优化版本"""
        from models.fairllm_core.attention_modifiers import SparseAttention
        
        for i, layer in enumerate(self.model.model.layers):
            orig_attn = layer.self_attn
            layer.self_attn = QuantAttentionWrapper(
                orig_attn.hidden_size,
                orig_attn.num_heads,
                sparse_config={
                    "window_size": 4096,
                    "global_tokens": 128,
                    "sparsity": 0.4
                }
            )
            
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        # 动态量化缩放
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
        # 收集量化指标
        quant_metrics = self._collect_quant_metrics(outputs.hidden_states)
        return outputs.logits, quant_metrics
    
    def _collect_quant_metrics(self, hidden_states: Tuple[Tensor]) -> dict:
        """计算各层的量化误差和激活稀疏度"""
        metrics = {}
        for i, hs in enumerate(hidden_states):
            metrics[f"layer_{i}_quant_err"] = torch.mean((hs - hs.dequantize())**2)
            metrics[f"layer_{i}_sparsity"] = torch.mean((hs == 0).float())
        return metrics

class QuantAttentionWrapper(nn.Module):
    """量化优化的滑动窗口注意力，集成稀疏模式"""
    
    def __init__(self, embed_dim: int, num_heads: int, sparse_config: dict):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_config = sparse_config
        
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads)
        
        # 稀疏注意力模式
        self.register_buffer(
            "attention_mask",
            self._create_sliding_window_mask(
                sparse_config["window_size"],
                sparse_config["global_tokens"]
            )
        )
        
    def _create_sliding_window_mask(self, window_size: int, global_tokens: int) -> Tensor:
        """生成混合稀疏注意力模式"""
        mask = torch.ones((window_size + global_tokens, window_size + global_tokens))
        # 局部窗口连接
        mask[:-global_tokens, :-global_tokens] = torch.tril(
            torch.ones(window_size, window_size), diagonal=window_size//2
        )
        # 全局token连接
        mask[-global_tokens:, :] = 1
        mask[:, -global_tokens:] = 1
        return mask.bool()
    
    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b l (t h d) -> t b h l d", t=3, h=self.num_heads)
        
        # 应用旋转位置编码
        q, k = self.rotary_emb(q, k)
        
        # 稀疏注意力计算
        attn_weights = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.embed_dim)
        attn_weights = attn_weights.masked_fill(~self.attention_mask, -torch.inf)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 混合精度计算
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
        
        return self.o_proj(rearrange(output, "b h l d -> b l (h d)"))

