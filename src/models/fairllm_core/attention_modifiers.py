# src/models/fairllm_core/attention_modifiers.py
import math
from typing import Optional, Tuple
import torch
from torch import nn, einsum
from einops import rearrange, repeat

class SparseAttention(nn.Module):
    """混合稀疏注意力机制，集成局部窗口与全局token的注意力计算。"""
    
    def __init__(self, config: Dict):
        """
        初始化混合稀疏注意力机制。

        Args:
            config (Dict): 配置字典，包含以下键：
                'dim' (int): 输入特征的维度。
                'heads' (int): 注意力头的数量。
                'window_size' (int, optional): 局部窗口的大小。默认为 256。
                'num_global_tokens' (int, optional): 全局 token 的数量。默认为 32。
                'sparsity' (float, optional): 稀疏目标比例。默认为 0.3。
        """
        super().__init__()
        self.dim = config['dim']
        self.heads = config['heads']
        self.window_size = config.get('window_size', 256)
        self.global_tokens = config.get('num_global_tokens', 32)
        
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.to_out = nn.Linear(self.dim, self.dim)
        
        # 局部注意力参数
        self.local_attn = nn.Parameter(torch.randn(self.heads, self.window_size, self.window_size))
        # 全局注意力参数  
        self.global_attn = nn.Parameter(torch.randn(self.heads, self.global_tokens, self.window_size))
        
        # 动态稀疏掩码生成器
        self.sparsity_controller = SparsityController(
            dim=self.dim,
            num_heads=self.heads,
            sparsity_target=config.get('sparsity', 0.3)
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, _ = x.shape
        
        # 生成QKV投影
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 局部窗口划分
        q = rearrange(q, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        k = rearrange(k, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        v = rearrange(v, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        
        # 局部注意力计算
        local_attn = torch.einsum('bhwnd,bhwmd->bhwnm', q, k) * (self.dim ** -0.5)
        local_attn += self.local_attn.unsqueeze(0).unsqueeze(2)
        
        # 全局注意力注入
        global_q = self.global_attn[:, :, None] @ q.mean(dim=2, keepdim=True)
        global_attn = torch.einsum('bhgnd,bhgnc->bhgnc', global_q, k.mean(dim=2))
        
        # 动态稀疏化
        attn = self.sparsity_controller(local_attn, global_attn)
        attn = attn.softmax(dim=-1)
        
        # 上下文聚合
        out = torch.einsum('bhwnm,bhwmd->bhwnd', attn, v)
        out = rearrange(out, 'b h w n d -> b (w n) (h d)')
        
        return self.to_out(out)

class AttentionModulatorBank(nn.Module):
    """ 该模块的设计旨在增强模型对多尺度特征的关注能力，通过不同层级的调制和跨层交互，
        使模型能够更好地捕捉数据中的复杂模式。在实际应用中，需要根据具体任务和数据特点，
        调整配置参数（如隐藏层大小、注意力头数等）以获得最佳性能。
    See Also:
        LayerWiseModulator: 用于对单个网络层的特征进行调制的模块。
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.modulators = nn.ModuleList([
            LayerWiseModulator(config, layer_id=i)
            for i in range(config['num_layers'])
        ])
        self.cross_layer_controller = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_attention_heads']
            ),
            num_layers=3
        )
        
    def forward(self, x: Tensor, context: Dict) -> List[Tensor]:
        device = x.device
        batch_size = x.size(0)
        
        # 生成层间协调信号
        coordination_signal = self.cross_layer_controller(x)
        
        # 并行计算各层调制参数
        mod_params = []
        for i, modulator in enumerate(self.modulators):
            layer_params = modulator(
                x, 
                context.get(f'layer_{i}_metrics', {}),
                coordination_signal[:, i, :]
            )
            mod_params.append(layer_params)
        
        # 应用动态注意力掩码
        attention_masks = []
        for params in mod_params:
            mask = self._generate_dynamic_mask(
                params['suppression_matrix'],
                params['enhancement_factor'],
                batch_size,
                device
            )
            attention_masks.append(mask)
            
        return attention_masks

    def _generate_dynamic_mask(self, suppression, enhancement, batch_size, device):
        base_mask = torch.ones((batch_size, self.num_heads, self.seq_len), device=device)
        suppressed = base_mask * (1 - suppression.unsqueeze(-1))
        enhanced = base_mask * enhancement.unsqueeze(-1)
        return torch.clamp(suppressed + enhanced, min=0, max=1)
