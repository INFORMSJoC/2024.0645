import time
import torch
from transformers import AutoModelForCausalLM, get_scheduler
from torch.distributed import get_rank, get_world_size

from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.model.model_utils import create_hf_model, create_reward_model
from utils.module.lora import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from utils.utils import get_optimizer_grouped_parameters, print_rank_0


def log_init(model_name, start_time=None):
    """
    记录模型初始化的开始和结束时间。
    """
    rank = get_rank()
    if rank == 0:
        tag = "start" if start_time is None else "end"
        suffix = "ing" if start_time is None else "ed"
        duration = ""
        if start_time is not None:
            duration = f"(duration: {time.time() - start_time:.2f}s)"
        msg = f"[{tag}] Initializing {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class CustomTrainingEngine:
    """
    自定义训练引擎，用于初始化和管理不同的模型。
    """
    def __init__(self, actor_model_path, reward_model_path, tokenizer, args, total_iters):
        self.args = args
        self.total_iters = total_iters
        self.tokenizer = tokenizer

        # 初始化演员模型
        self.actor = self._initialize_actor(actor_model_path)
        # 初始化参考模型
        self.ref = self._initialize_ref(actor_model_path)
        # 初始化EMA模型（可选）
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._initialize_ema(actor_model_path)
        # 初始化奖励模型
        self.critic = None
        self.reward = self._initialize_reward(reward_model_path)

    def _initialize_actor(self, model_path):
        """
        初始化演员模型。
        """
        start_time = log_init("Actor")

        # 配置DeepSpeed
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=not self.args.unpin_actor_parameters,
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len + self.args.max_answer_seq_len,
            enable_tensorboard=self.args.enable_tensorboard,
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            bf16=self.args.actor_bf16,
            memory_efficient_linear=self.args.memory_efficient_linear,
            tb_path=self.args.tensorboard_path,
            tb_name=""
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * get_world_size()
            * self.args.gradient_accumulation_steps_actor
        )

        # 创建模型
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=model_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.args.disable_actor_dropout
        )

        # LoRA配置
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name, self.args.actor_lora_dim
            )
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(actor_model)

        # 优化器配置
        optimizer_class = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model,
            self.args.actor_weight_decay,
            self.args.actor_lora_learning_rate
        )
        optimizer = optimizer_class(
            optim_params, lr=self.args.actor_learning_rate, betas=(0.9, 0.95)
        )

        # 学习率调度器
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.total_iters
        )

        # 初始化DeepSpeed引擎
        actor_engine, *_ = deepspeed.initialize(
            model=actor_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config
        )

        log_init("Actor", start_time)
        return actor_engine

    def _initialize_ref(self, model_path):
        """
        初始化参考模型。
        """
        start_time = log_init("Reference")

        # 配置DeepSpeed
        zero_stage = self.args.reference_zero_stage
        if zero_stage not in (0, 3):
            zero_stage = 0
            print_rank_0(
                f"Setting stage = {zero_stage} for the reference model (as it does not have optimizer and gradients)."
            )

        ds_config = get_eval_ds_config(
            offload=self.args.offload_reference_model,
            stage=zero_stage,
            bf16=self.args.actor_bf16
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * get_world_size()
            * self.args.gradient_accumulation_steps_actor
        )

        # 创建模型
        ref_model = create_hf_model(
            AutoModelForCausalLM, model_path, self.tokenizer, ds_config
        )

        # 初始化DeepSpeed引擎
        ref_engine, *_ = deepspeed.initialize(model=ref_model, config=ds_config)
        log_init("Reference", start_time)
        return ref_engine

    def _initialize_ema(self, model_path):
        """
        初始化EMA模型。
        """
        start_time = log_init("EMA")

        # 配置DeepSpeed
        zero_stage = self.args.reference_zero_stage
        if zero_stage not in (0, 3):
            zero_stage = 0
            print_rank_0(
                f"Setting stage = {zero_stage} for the EMA model (as it does not have optimizer and gradients)."
            )

        ds_config = get_eval_ds_config(
            offload=self.args.offload_reference_model,
            stage=zero_stage,
            bf16=self.args.actor_bf16
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * get_world_size()
            * self.args.gradient_accumulation_steps_actor
        )

        # 创建模型
        ema_model = create_hf_model(
            AutoModelForCausalLM, model_path, self.tokenizer, ds_config
        )
        if self.args.actor_lora_dim > 0:
            ema_model = convert_linear_layer_to_lora(
                ema_model, self.args.actor_lora_module_name, self.args.actor_lora_dim
            )

        # 初始化DeepSpeed引擎
        ema_engine, *_ = deepspeed.initialize(model=ema_model, config=ds_config)
        log_init("EMA", start_time)
        return ema_engine

    def _initialize_reward(self, model_path):
        """
        初始化奖励模型。
        """
        start_time = log_init("Reward")

        # 配置DeepSpeed
        zero_stage = self.args.reward_zero_stage
        if zero_stage != 3:
            zero_stage = 0
            print_rank_0(
                f"Setting stage = {zero_stage} for the reward model (as it does not have optimizer and gradients)."
            )

        ds_config = get_eval_ds_config(
            offload=self.args.offload_reward_model,
            stage=zero_stage,
            bf16=self.args.reward_bf16
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * get_world_size()
            * self.args.gradient_accumulation_steps
        )

        # 创建模型
        reward_model = create_reward_model(
            model_name_or_path=model_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_reward_dropout,
            zero_stage=zero_stage
        )

        # 初始化DeepSpeed引擎
        reward_engine, *_ = deepspeed.initialize(model=reward_model, config=ds_config)
        log_init("Reward", start_time)
        return reward_engine


if __name__ == "__main__":
    # 假设 args 是一个包含所有必要参数的对象
    # 创建训练引擎
    engine = CustomTrainingEngine(
        actor_model_path="actor_model_path",
        reward_model_path="reward_model_path",
        tokenizer="tokenizer",
        args="args",
        total_iters=1000
    )
    # 使用 engine.actor, engine.ref, engine.actor_ema, engine.reward 进行训练
    