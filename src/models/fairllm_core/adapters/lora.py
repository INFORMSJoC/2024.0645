# src/models/fairllm_core/adapters/lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

class DynamicLoraAdapter(nn.Module):
    """��̬LoRA��������֧������ʱ����·��"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.r = config['lora_rank']
        self.alpha = config['lora_alpha']
        self.target_modules = config['target_modules']
        self.task_embeddings = nn.Embedding(
            config['num_tasks'], 
            config['task_embed_dim']
        )
        self.router = nn.Sequential(
            nn.Linear(config['task_embed_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, len(self.target_modules)*2)
        )
        
        # ��ʼ��������������
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for module in self.target_modules:
            self.lora_A[module] = nn.Parameter(
                torch.randn(config['hidden_size'], self.r)
            )
            self.lora_B[module] = nn.Parameter(
                torch.zeros(self.r, config['hidden_size'])
            )
        
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        task_emb = self.task_embeddings(task_id)
        gates = torch.sigmoid(self.router(task_emb))
        
        # ��̬���������
        adapter_output = 0
        for i, module in enumerate(self.target_modules):
            A = self.lora_A[module]
            B = self.lora_B[module]
            scale = self.alpha / self.r
            
            # ����ſ�ֵ
            gate_a, gate_b = gates[:, 2*i], gates[:, 2*i+1]
            
            # ��������������
            delta_weight = (gate_a.unsqueeze(-1) * A) @ (gate_b.unsqueeze(-1) * B)
            adapter_output += torch.einsum('bd,dr->br', x, delta_weight) * scale
            
        return x + adapter_output

class AdapterRouter(nn.Module):
    """��̬������·�����磬��Ӧ�����еĹ�ʽʵ��"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.adapters = nn.ModuleDict({
            name: DynamicLoraAdapter(config)
            for name in config['adapter_types']
        })
        self.router_network = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
               
