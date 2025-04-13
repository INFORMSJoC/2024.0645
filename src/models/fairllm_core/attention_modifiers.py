# src/models/fairllm_core/attention_modifiers.py
import math
from typing import Optional, Tuple
import torch
from torch import nn, einsum
from einops import rearrange, repeat

class SparseAttention(nn.Module):
    """���ϡ��ע�������ƣ����ɾֲ�������ȫ��token"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.dim = config['dim']
        self.heads = config['heads']
        self.window_size = config.get('window_size', 256)
        self.global_tokens = config.get('num_global_tokens', 32)
        
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.to_out = nn.Linear(self.dim, self.dim)
        
        # �ֲ�ע��������
        self.local_attn = nn.Parameter(torch.randn(self.heads, self.window_size, self.window_size))
        # ȫ��ע��������  
        self.global_attn = nn.Parameter(torch.randn(self.heads, self.global_tokens, self.window_size))
        
        # ��̬ϡ������������
        self.sparsity_controller = SparsityController(
            dim=self.dim,
            num_heads=self.heads,
            sparsity_target=config.get('sparsity', 0.3)
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, _ = x.shape
        
        # ����QKVͶӰ
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # �ֲ����ڻ���
        q = rearrange(q, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        k = rearrange(k, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        v = rearrange(v, 'b h (w n) d -> b h w n d', w=n // self.window_size)
        
        # �ֲ�ע��������
        local_attn = torch.einsum('bhwnd,bhwmd->bhwnm', q, k) * (self.dim ** -0.5)
        local_attn += self.local_attn.unsqueeze(0).unsqueeze(2)
        
        # ȫ��ע����ע��
        global_q = self.global_attn[:, :, None] @ q.mean(dim=2, keepdim=True)
        global_attn = torch.einsum('bhgnd,bhgnc->bhgnc', global_q, k.mean(dim=2))
        
        # ��̬ϡ�軯
        attn = self.sparsity_controller(local_attn, global_attn)
        attn = attn.softmax(dim=-1)
        
        # �����ľۺ�
        out = torch.einsum('bhwnm,bhwmd->bhwnd', attn, v)
        out = rearrange(out, 'b h w n d -> b (w n) (h d)')
        
        return self.to_out(out)

class AttentionModulatorBank(nn.Module):
    """������ע������������ʵ�������еĹ�ʽ(12)-(15)"""
    
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
        
        # ���ɲ��Э���ź�
        coordination_signal = self.cross_layer_controller(x)
        
        # ���м��������Ʋ���
        mod_params = []
        for i, modulator in enumerate(self.modulators):
            layer_params = modulator(
                x, 
                context.get(f'layer_{i}_metrics', {}),
                coordination_signal[:, i, :]
            )
            mod_params.append(layer_params)
        
        # Ӧ�ö�̬ע��������
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
