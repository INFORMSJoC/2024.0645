#!/bin/bash
# scripts/train_selfbmil.sh

set -euo pipefail

# 环境配置
export ACCELERATE_LOG_LEVEL=debug
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=16
export TF32_OVERRIDE=0
export TORCH_EXTENSIONS_DIR=/tmp/fairllm/torch_extensions

# 默认参数
CONFIG_PATH="${1:-configs/training_configs/rlhf_selfbmil_train.yaml}"
LOG_DIR="${2:-logs/$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS=10000
SAVE_INTERVAL=500
EVAL_INTERVAL=100
LOGGING_INTERVAL=10
MIXED_PRECISION="bf16"
DYNAMO_BACKEND="inductor"
FS_DP_CONFIG="fsdp_config.json"

# 环境检测
NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_NODES=$(kubectl get nodes -l role=training --no-headers | wc -l)
if [ -z "$NUM_NODES" ]; then
    NUM_NODES=1
fi

# 初始化训练环境
init_training() {
    echo "初始化训练环境..."
    mkdir -p "$LOG_DIR/checkpoints"
    mkdir -p "$LOG_DIR/tensorboard"
    cp "$CONFIG_PATH" "$LOG_DIR/config.yaml"
    
    # 配置加速器
    accelerate config default
    accelerate config --num_machines "$NUM_NODES"
    accelerate config --num_processes "$((NUM_GPUS * NUM_NODES))"
    accelerate config --mixed_precision "$MIXED_PRECISION"
    accelerate config --dynamo_backend "$DYNAMO_BACKEND"
    accelerate config --fsdp_config "$FS_DP_CONFIG"
    
    # 配置日志
    touch "$LOG_DIR/train.log"
    touch "$LOG_DIR/eval.log"
    touch "$LOG_DIR/metrics.log"
    
    # 初始化 Kubernetes
    kubectl create namespace fairllm 2>/dev/null || true
}

# 执行多阶段训练
run_training() {
    echo "开始多阶段训练..."
    
    accelerate launch \
        --num_processes "$((NUM_GPUS * NUM_NODES))" \
        --num_machines "$NUM_NODES" \
        --mixed_precision "$MIXED_PRECISION" \
        --dynamo_backend "$DYNAMO_BACKEND" \
        --fsdp_config "$FS_DP_CONFIG" \
        src/training/rlhf_trainer.py \
        --config "$CONFIG_PATH" \
        --log_dir "$LOG_DIR" \
        --max_steps "$MAX_STEPS" \
        --save_interval "$SAVE_INTERVAL" \
        --eval_interval "$EVAL_INTERVAL" \
        --logging_interval "$LOGGING_INTERVAL" \
        --project_name "fairllm" \
        --wandb_key "$(vault get wandb::access-key)" \
        --tensorboard_dir "$LOG_DIR/tensorboard" \
        --debug_mode "${DEBUG_MODE:-false}" \
        --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-null}" \
        2>&1 | tee "$LOG_DIR/train.log"
}

# 训练后优化
post_train() {
    echo "训练完成，开始模型优化..."
    
    # 模型量化、剪枝和编译
    python scripts/optimize_model.py \
        --input "$LOG_DIR/checkpoints/final" \
        --output "$LOG_DIR/optimized_model" \
        --quantize \
        --prune \
        --compile \
        --target_device "cuda" \
        --performance_report "$LOG_DIR/optimization_report.json"
    
    # 同步到云存储
    aws s3 sync "$LOG_DIR" "s3://fairllm-training-logs/$(basename "$LOG_DIR")"
    
    # 生成模型验证报告
    python scripts/validate_model.py \
        --model_path "$LOG_DIR/optimized_model" \
        --test_data_path "data/test_set.json" \
        --report_path "$LOG_DIR/validation_report.json"
    
    # 上传验证报告
    aws s3 cp "$LOG_DIR/validation_report.json" "s3://fairllm-training-reports/$(basename "$LOG_DIR")_validation.json"
    
    # 推送指标到 Prometheus
    python scripts/push_to_prometheus.py \
        --report "$LOG_DIR/validation_report.json" \
        --endpoint "http://prometheus-push-gateway:9091"
}

# 异常处理
handle_interrupt() {
    echo "训练中断，保存进度..."
    accelerate checkpoint "$LOG_DIR/checkpoints/interrupt"
    
    # 发送告警通知
    python scripts/send_alert.py \
        --message "训练中断" \
        --severity "warning"
    
    exit 1
}

# 主程序
main() {
    # 初始化
    init_training
    
    # 捕获中断信号
    trap handle_interrupt SIGINT SIGTERM
    
    # 执行训练
    if ! run_training; then
        echo "训练失败，错误代码: $?"
        exit 1
    fi
    
    # 后处理
    post_train
}

# 执行主程序
main

