# src/training/adversarial_trainer.py
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim.lr_scheduler import CyclicLR
from models.fairllm_core.attention_modifiers import AttentionModulatorBank
from agents.debate_agent import DebateCoordinator

class AdversarialTrainer:
    """多智能体对抗训练系统，集成梯度扰动与对抗样本生成"""
    
    def __init__(self, model, adversaries, config: dict):
        self.model = model
        self.adversaries = adversaries
        self.config = config
        
        # 对抗优化器
        self.adv_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, adversaries.parameters()),
            lr=config['adv_lr'],
            momentum=0.9,
            nesterov=True
        )
        
        # 主模型优化器
        self.main_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['main_lr'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CyclicLR(
            self.main_optimizer,
            base_lr=config['base_lr'],
            max_lr=config['max_lr'],
            step_size_up=config['cycle_steps']
        )
        
        # 对抗参数
        self.attack_scheduler = AttackScheduler(
            max_steps=config['max_steps'],
            attack_types=config['attack_types']
        )
        
        # 辩论协调器
        self.debate_coordinator = DebateCoordinator(config['debate_config'])

    def train_step(self, batch: dict) -> dict:
        """对抗训练三重奏：生成、攻击、防御"""
        # 生成对抗样本
        adv_batch = self._generate_adversarial_examples(batch)
        
        # 执行对抗训练
        debate_result = self.debate_coordinator.conduct_debate(adv_batch)
        final_batch = self._merge_debate_results(adv_batch, debate_result)
        
        # 主模型优化
        main_loss, stats = self._update_main_model(final_batch)
        
        # 对抗器优化
        adv_loss = self._update_adversaries(batch, final_batch)
        
        return {**stats, "loss/adv": adv_loss}

    def _generate_adversarial_examples(self, batch: dict) -> dict:
        """生成多类型对抗样本"""
        self.model.eval()
        self.adversaries.train()
        
        adversarial_samples = []
        for attack in self.attack_scheduler.get_current_attacks():
            adv_sample = self._apply_attack(
                batch, 
                attack_type=attack,
                epsilon=self.config['epsilon']
            )
            adversarial_samples.append(adv_sample)
        
        return self._combine_adv_samples(adversarial_samples)

    def _apply_attack(self, batch: dict, attack_type: str, epsilon: float) -> dict:
        """执行指定类型的对抗攻击"""
        if attack_type == "PGD":
            return self._pgd_attack(batch, epsilon)
        elif attack_type == "FGSM":
            return self._fgsm_attack(batch, epsilon)
        elif attack_type == "CLARE":
            return self._clare_attack(batch)
        else:
            raise ValueError(f"未知攻击类型: {attack_type}")

    def _pgd_attack(self, batch: dict, epsilon: float, num_steps: int = 3) -> dict:
        """投影梯度下降攻击"""
        orig_embeds = self.model.get_input_embeddings()(batch['input_ids'])
        delta = torch.zeros_like(orig_embeds).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for _ in range(num_steps):
            perturbed = orig_embeds + delta
            outputs = self.model(inputs_embeds=perturbed)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            grad = torch.autograd.grad(loss, [delta])[0]
            
            # 更新扰动
            delta = delta + self.config['alpha'] * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = delta.detach().requires_grad_()
        
        return self._project_perturbation(delta, orig_embeds, epsilon)

class AttackScheduler:
    """动态对抗攻击调度器，实现课程对抗训练"""
    
    def __init__(self, max_steps: int, attack_types: list):
        self.step = 0
        self.max_steps = max_steps
        self.attack_types = attack_types
        self.current_strategy = self._init_strategy()
        
    def get_current_attacks(self) -> list:
        """根据训练进度选择攻击组合"""
        progress = self.step / self.max_steps
        if progress < 0.3:
            return [self.attack_types[0]]
        elif progress < 0.6:
            return self.attack_types[:2]
        else:
            return self.attack_types
        
    def _init_strategy(self) -> dict:
        """初始化攻击参数调度策略"""
        return {
            'epsilon': lambda t: 0.1 + 0.4 * (t / self.max_steps),
            'alpha': lambda t: 0.01 * (1 - t / self.max_steps),
            'probability': lambda t: 0.5 * (t / self.max_steps)
        }
