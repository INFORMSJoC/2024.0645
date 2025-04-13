#!/bin/bash
# scripts/model_service.sh

set -euo pipefail

MODEL_PATH="${1:-/models/fairllm}"
CONFIG_PATH="${2:-configs/experiment_configs/realworld_deployment.yaml}"
DEPLOY_ENV="${3:-prod}"
REPLICAS="${4:-3}"
NAMESPACE="${5:-fairllm}"
KUBECONFIG="${KUBECONFIG:-/etc/kubernetes/admin.conf}"

# 全局变量
GIT_SHA=$(git rev-parse --short HEAD)
BUILD_TAG="fairllm-service:${GIT_SHA}"
REGISTRY="registry.internal/fairllm"
IMAGE_TAG="${REGISTRY}/service:${GIT_SHA}"

# 模型验证
validate_model() {
    echo "开始模型验证..."
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "错误：模型路径不存在"
        exit 1
    fi
    
    # 模型完整性检查
    if ! python scripts/verify_model.py \
        --model "$MODEL_PATH" \
        --config "$CONFIG_PATH" \
        --integrity_check \
        --performance_check; then
        echo "模型验证失败"
        exit 2
    fi
    
    # 模型性能测试
    if ! python scripts/test_model_performance.py \
        --model "$MODEL_PATH" \
        --config "$CONFIG_PATH" \
        --output "$MODEL_PATH/performance_report.json"; then
        echo "模型性能测试失败"
        exit 3
    fi
}

# 构建服务镜像
build_service() {
    echo "开始构建服务镜像..."
    
    # 构建镜像
    docker build -t "$BUILD_TAG" \
        --build-arg MODEL_PATH="$MODEL_PATH" \
        --build-arg CONFIG_PATH="$CONFIG_PATH" \
        --build-arg GIT_SHA="$GIT_SHA" \
        -f docker/Dockerfile.api .
    
    # 打标签
    docker tag "$BUILD_TAG" "$IMAGE_TAG"
    
    # 推送镜像
    if ! docker push "$IMAGE_TAG"; then
        echo "镜像推送失败"
        exit 4
    fi
}

# 部署到 Kubernetes
deploy_k8s() {
    echo "开始部署到 Kubernetes..."
    
    # 替换环境变量
    envsubst < k8s/deployment.yaml | kubectl apply -f - -n "$NAMESPACE"
    kubectl rollout status deploy/fairllm-service -n "$NAMESPACE" --timeout=300s
    
    # 根据部署环境选择策略
    case "$DEPLOY_ENV" in
        "prod")
            # 金丝雀发布
            kubectl apply -f k8s/canary.yaml -n "$NAMESPACE"
            sleep 300  # 监控阶段
            kubectl apply -f k8s/full_deploy.yaml -n "$NAMESPACE"
            
            # 设置 HPA
            kubectl apply -f k8s/hpa.yaml -n "$NAMESPACE"
            ;;
        "staging")
            # 蓝绿部署
            kubectl apply -f k8s/blue_green.yaml -n "$NAMESPACE"
            ;;
        *)
            echo "未知环境: $DEPLOY_ENV"
            exit 5
            ;;
    esac
}

# 设置监控
setup_monitoring() {
    echo "设置监控系统..."
    
    # 部署监控组件
    kubectl apply -f k8s/monitoring/ -n "$NAMESPACE"
    
    # 升级 Prometheus
    helm upgrade prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$NAMESPACE" \
        --values k8s/monitoring/values.yaml
    
    # 初始化告警规则
    kubectl apply -f k8s/monitoring/alerts/ -n "$NAMESPACE"
    
    # 配置日志收集
    kubectl apply -f k8s/logging/ -n "$NAMESPACE"
}

# 验证部署
validate_deployment() {
    echo "验证部署状态..."
    
    # 检查 Pod 状态
    if ! kubectl wait --for=condition=Ready pod \
        -l app=fairllm-service \
        -n "$NAMESPACE" \
        --timeout=300s; then
        echo "Pod 未就绪"
        exit 6
    fi
    
    # 运行冒烟测试
    if ! python scripts/smoke_test.py \
        --endpoint "http://fairllm-service.$NAMESPACE.svc.cluster.local/api/v1/predict" \
        --config "$CONFIG_PATH"; then
        echo "冒烟测试失败"
        exit 7
    fi
}

# 异常处理
handle_error() {
    local exit_code=$?
    echo "部署失败，错误代码: $exit_code"
    
    # 回滚部署
    kubectl rollout undo deploy/fairllm-service -n "$NAMESPACE"
    
    # 发送告警通知
    python scripts/send_alert.py \
        --message "部署失败 (Exit Code: $exit_code)" \
        --severity "critical"
    
    exit $exit_code
}

# 主程序
main() {
    # 捕获错误
    trap handle_error ERR
    
    # 初始化
    validate_model
    build_service
    
    # 部署
    deploy_k8s
    setup_monitoring
    
    # 验证
    validate_deployment
    
    # 清理
    docker rmi "$BUILD_TAG"
    docker rmi "$IMAGE_TAG"
    
    echo "部署成功"
}

# 执行主程序
main
