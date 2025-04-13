#!/bin/bash
# scripts/train_selfbmil.sh

set -euo pipefail

# ��������
export ACCELERATE_LOG_LEVEL=debug
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=16
export TF32_OVERRIDE=0
export TORCH_EXTENSIONS_DIR=/tmp/fairllm/torch_extensions

# Ĭ�ϲ���
CONFIG_PATH="${1:-configs/training_configs/rlhf_selfbmil_train.yaml}"
LOG_DIR="${2:-logs/$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS=10000
SAVE_INTERVAL=500
EVAL_INTERVAL=100
LOGGING_INTERVAL=10
MIXED_PRECISION="bf16"
DYNAMO_BACKEND="inductor"
FS_DP_CONFIG="fsdp_config.json"

# �������
NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_NODES=$(kubectl get nodes -l role=training --no-headers | wc -l)
if [ -z "$NUM_NODES" ]; then
    NUM_NODES=1
fi

# ��ʼ��ѵ������
init_training() {
    echo "��ʼ��ѵ������..."
    mkdir -p "$LOG_DIR/checkpoints"
    mkdir -p "$LOG_DIR/tensorboard"
    cp "$CONFIG_PATH" "$LOG_DIR/config.yaml"
    
    # ���ü�����
    accelerate config default
    accelerate config --num_machines "$NUM_NODES"
    accelerate config --num_processes "$((NUM_GPUS * NUM_NODES))"
    accelerate config --mixed_precision "$MIXED_PRECISION"
    accelerate config --dynamo_backend "$DYNAMO_BACKEND"
    accelerate config --fsdp_config "$FS_DP_CONFIG"
    
    # ������־
    touch "$LOG_DIR/train.log"
    touch "$LOG_DIR/eval.log"
    touch "$LOG_DIR/metrics.log"
    
    # ��ʼ�� Kubernetes
    kubectl create namespace fairllm 2>/dev/null || true
}

# ִ�ж�׶�ѵ��
run_training() {
    echo "��ʼ��׶�ѵ��..."
    
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

# ѵ�����Ż�
post_train() {
    echo "ѵ����ɣ���ʼģ���Ż�..."
    
    # ģ����������֦�ͱ���
    python scripts/optimize_model.py \
        --input "$LOG_DIR/checkpoints/final" \
        --output "$LOG_DIR/optimized_model" \
        --quantize \
        --prune \
        --compile \
        --target_device "cuda" \
        --performance_report "$LOG_DIR/optimization_report.json"
    
    # ͬ�����ƴ洢
    aws s3 sync "$LOG_DIR" "s3://fairllm-training-logs/$(basename "$LOG_DIR")"
    
    # ����ģ����֤����
    python scripts/validate_model.py \
        --model_path "$LOG_DIR/optimized_model" \
        --test_data_path "data/test_set.json" \
        --report_path "$LOG_DIR/validation_report.json"
    
    # �ϴ���֤����
    aws s3 cp "$LOG_DIR/validation_report.json" "s3://fairllm-training-reports/$(basename "$LOG_DIR")_validation.json"
    
    # ����ָ�굽 Prometheus
    python scripts/push_to_prometheus.py \
        --report "$LOG_DIR/validation_report.json" \
        --endpoint "http://prometheus-push-gateway:9091"
}

# �쳣����
handle_interrupt() {
    echo "ѵ���жϣ��������..."
    accelerate checkpoint "$LOG_DIR/checkpoints/interrupt"
    
    # ���͸澯֪ͨ
    python scripts/send_alert.py \
        --message "ѵ���ж�" \
        --severity "warning"
    
    exit 1
}

# ������
main() {
    # ��ʼ��
    init_training
    
    # �����ж��ź�
    trap handle_interrupt SIGINT SIGTERM
    
    # ִ��ѵ��
    if ! run_training; then
        echo "ѵ��ʧ�ܣ��������: $?"
        exit 1
    fi
    
    # ����
    post_train
}

# ִ��������
main

