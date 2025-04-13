# src/training/rlhf_trainer.py
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from accelerate import Accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from models.fairllm_core.attention_modifiers import SparseAttention
from agents.self_reflection_agent import ReflectionChain

class RLHFTrainer:
    """我们自定义一个rlhf训练，集成ppo算法与反射式优化"""
    
    def __init__(self, config: dict, model, ref_model, reward_model, tokenizer):
        self.accelerator = Accelerator()
        self.config = config
        self.step_counter = 0
        
        # 初始化分布式训练环境
        self.model, self.ref_model, self.reward_model, self.optimizer = self._setup_distributed(
            model, ref_model, reward_model
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['max_steps']
        )
        
        # 反射优化组件
        self.reflection = ReflectionChain(config['reflection_config'])
        self.kl_ctl = AdaptiveKLController(
            init_kl=config['init_kl'],
            target=config['target_kl'],
            horizon=config['kl_horizon']
        )
        
        # 奖励塑形模块
        self.reward_shaper = RewardShaper(
            kl_coeff=config['kl_coeff'],
            length_penalty=config['length_penalty'],
            diversity_bonus=config['diversity_bonus']
        )

    def _setup_distributed(self, model, ref_model, reward_model):
        """配置分布式训练环境"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        model, optimizer = self.accelerator.prepare(model, optimizer)
        ref_model = self.accelerator.prepare(ref_model)
        reward_model = self.accelerator.prepare(reward_model)
        
        return model, ref_model, reward_model, optimizer

    def train_step(self, batch: dict) -> dict:
        """执行完整训练步骤，包含PPO三个阶段"""
        # 生成阶段
        with torch.no_grad():
            rollout = self._generate_rollout(batch)
        
        # 评估阶段
        rewards = self._calculate_rewards(rollout)
        
        # 反射优化阶段
        rollout = self.reflection.apply(rollout)
        
        # 学习阶段
        stats = self._ppo_update(rollout, rewards)
        
        # 分布式同步
        self.accelerator.wait_for_everyone()
        return stats

    def _generate_rollout(self, batch: dict) -> dict:
        """使用混合采样策略生成训练数据"""
        self.model.eval()
        generate_kwargs = {
            "max_length": self.config['max_length'],
            "temperature": self.config['temperature'],
            "top_p": self.config['top_p'],
            "do_sample": True,
            "use_cache": True
        }
        
        with self.accelerator.autocast():
            outputs = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                **generate_kwargs
            )
        
        # 反射式优化生成结果
        outputs = self.reflection.refine(outputs)
        return self._process_rollout(outputs)

    def _calculate_rewards(self, rollout: dict) -> torch.Tensor:
        """计算多维度奖励信号"""
        # 基础奖励
        with torch.no_grad():
            reward_outputs = self.reward_model(
                input_ids=rollout['input_ids'],
                attention_mask=rollout['attention_mask']
            )
            base_rewards = reward_outputs.logits.squeeze(-1)
        
        # KL散度惩罚
        logprobs = rollout['logprobs']
        ref_logprobs = rollout['ref_logprobs']
        kl_penalty = self.kl_ctl.get_penalty(logprobs, ref_logprobs)
        
        # 多样性奖励
        diversity_bonus = self._calculate_diversity(rollout['sequences'])
        
        return self.reward_shaper(
            base_rewards,
            kl_penalty,
            diversity_bonus,
            rollout['response_lengths']
        )

    def _ppo_update(self, rollout: dict, rewards: torch.Tensor) -> dict:
        """PPO优化阶段，集成混合精度训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 计算优势
        values = rollout['values']
        advantages = self._compute_advantages(rewards, values)
        
        # 多阶段优化
        stats = {}
        for _ in range(self.config['ppo_epochs']):
            for mini_batch in self._create_mini_batches(rollout, advantages):
                loss, sub_stats = self._compute_loss(mini_batch)
                self.accelerator.backward(loss)
                
                # 梯度裁剪与更新
                if self.accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 合并统计信息
                stats = self._merge_stats(stats, sub_stats)
        
        return stats

    def _compute_loss(self, batch: dict) -> tuple:
        """计算PPO损失函数，包含价值函数损失"""
        # 策略损失
        logits = self.model(**batch['model_inputs']).logits
        logprobs = self._get_logprobs(logits, batch['response_ids'])
        ratios = torch.exp(logprobs - batch['old_logprobs'])
        
        # CLIP损失
        surr1 = ratios * batch['advantages']
        surr2 = torch.clamp(ratios, 1 - self.config['clip_eps'], 1 + self.config['clip_eps']) * batch['advantages']
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        values = self.model.value_head(batch['hidden_states'])
        value_loss = F.mse_loss(values, batch['returns'])
        
        # 熵正则化
        entropy = self._compute_entropy(logits)
        
        total_loss = (
            policy_loss 
            + self.config['value_coeff'] * value_loss
            - self.config['entropy_coeff'] * entropy
        )
        
        return total_loss, {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "metrics/entropy": entropy.item(),
            "metrics/kl_divergence": (logprobs - batch['old_logprobs']).mean().item()
        }

class AdaptiveKLController:
    """动态KL散度控制器"""
    
    def __init__(self, init_kl: float, target: float, horizon: int):
        self.kl_coeff = init_kl
        self.target = target
        self.horizon = horizon
        
    def get_penalty(self, logprobs: Tensor, ref_logprobs: Tensor) -> Tensor:
        kl = logprobs - ref_logprobs
        kl_mean = kl.mean()
        self._update_coeff(kl_mean)
        return self.kl_coeff * kl
    
    def _update_coeff(self, kl: float):
        target = self.target
        proportional_error = (kl - target) / target
        adjustment = proportional_error / self.horizon
        self.kl_coeff = max(self.kl_coeff * (1 + adjustment), 0.0)
