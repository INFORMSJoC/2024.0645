# src/models/third_party/llama_decorators.py
from functools import wraps
import torch
import time

def tensor_parallelize(func):
    """张量并行装饰器，优化Llama模型计算"""
    
    @wraps(func)
    def wrapper(self, hidden_states, *args, **kwargs):
        if self.tensor_parallel_enabled:
            hidden_states = self._split_tensor(hidden_states)
            outputs = func(self, hidden_states, *args, **kwargs)
            return self._merge_tensors(outputs)
        else:
            return func(self, hidden_states, *args, **kwargs)
    
    return wrapper

def fused_attention(func):
    """融合注意力计算装饰器，集成FlashAttention"""
    
    @wraps(func)
    def wrapper(self, query, key, value, **kwargs):
        if self.use_flash_attention:
            return self._flash_attention_forward(query, key, value)
        else:
            return func(self, query, key, value, **kwargs)
    
    return wrapper

def benchmark_latency(func):
    """延迟基准测试装饰器，收集性能指标"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        latency = time.perf_counter() - start_time
        
        if not hasattr(wrapper, "latency_stats"):
            wrapper.latency_stats = []
        wrapper.latency_stats.append(latency)
        
        return result
    
    return wrapper

