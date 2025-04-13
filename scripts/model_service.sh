#!/bin/bash
# scripts/model_service.sh

set -euo pipefail

MODEL_PATH="${1:-/models/fairllm}"
CONFIG_PATH="${2:-configs/experiment_configs/realworld_deployment.yaml}"
DEPLOY_ENV="${3:-prod}"
REPLICAS="${4:-3}"
NAMESPACE="${5:-fairllm}"
KUBECONFIG="${KUBECONFIG:-/etc/kubernetes/admin.conf}"

# ȫ�ֱ���
GIT_SHA=$(git rev-parse --short HEAD)
BUILD_TAG="fairllm-service:${GIT_SHA}"
REGISTRY="registry.internal/fairllm"
IMAGE_TAG="${REGISTRY}/service:${GIT_SHA}"

# ģ����֤
validate_model() {
    echo "��ʼģ����֤..."
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "����ģ��·��������"
        exit 1
    fi
    
    # ģ�������Լ��
    if ! python scripts/verify_model.py \
        --model "$MODEL_PATH" \
        --config "$CONFIG_PATH" \
        --integrity_check \
        --performance_check; then
        echo "ģ����֤ʧ��"
        exit 2
    fi
    
    # ģ�����ܲ���
    if ! python scripts/test_model_performance.py \
        --model "$MODEL_PATH" \
        --config "$CONFIG_PATH" \
        --output "$MODEL_PATH/performance_report.json"; then
        echo "ģ�����ܲ���ʧ��"
        exit 3
    fi
}

# ����������
build_service() {
    echo "��ʼ����������..."
    
    # ��������
    docker build -t "$BUILD_TAG" \
        --build-arg MODEL_PATH="$MODEL_PATH" \
        --build-arg CONFIG_PATH="$CONFIG_PATH" \
        --build-arg GIT_SHA="$GIT_SHA" \
        -f docker/Dockerfile.api .
    
    # ���ǩ
    docker tag "$BUILD_TAG" "$IMAGE_TAG"
    
    # ���;���
    if ! docker push "$IMAGE_TAG"; then
        echo "��������ʧ��"
        exit 4
    fi
}

# ���� Kubernetes
deploy_k8s() {
    echo "��ʼ���� Kubernetes..."
    
    # �滻��������
    envsubst < k8s/deployment.yaml | kubectl apply -f - -n "$NAMESPACE"
    kubectl rollout status deploy/fairllm-service -n "$NAMESPACE" --timeout=300s
    
    # ���ݲ��𻷾�ѡ�����
    case "$DEPLOY_ENV" in
        "prod")
            # ��˿ȸ����
            kubectl apply -f k8s/canary.yaml -n "$NAMESPACE"
            sleep 300  # ��ؽ׶�
            kubectl apply -f k8s/full_deploy.yaml -n "$NAMESPACE"
            
            # ���� HPA
            kubectl apply -f k8s/hpa.yaml -n "$NAMESPACE"
            ;;
        "staging")
            # ���̲���
            kubectl apply -f k8s/blue_green.yaml -n "$NAMESPACE"
            ;;
        *)
            echo "δ֪����: $DEPLOY_ENV"
            exit 5
            ;;
    esac
}

# ���ü��
setup_monitoring() {
    echo "���ü��ϵͳ..."
    
    # ���������
    kubectl apply -f k8s/monitoring/ -n "$NAMESPACE"
    
    # ���� Prometheus
    helm upgrade prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$NAMESPACE" \
        --values k8s/monitoring/values.yaml
    
    # ��ʼ���澯����
    kubectl apply -f k8s/monitoring/alerts/ -n "$NAMESPACE"
    
    # ������־�ռ�
    kubectl apply -f k8s/logging/ -n "$NAMESPACE"
}

# ��֤����
validate_deployment() {
    echo "��֤����״̬..."
    
    # ��� Pod ״̬
    if ! kubectl wait --for=condition=Ready pod \
        -l app=fairllm-service \
        -n "$NAMESPACE" \
        --timeout=300s; then
        echo "Pod δ����"
        exit 6
    fi
    
    # ����ð�̲���
    if ! python scripts/smoke_test.py \
        --endpoint "http://fairllm-service.$NAMESPACE.svc.cluster.local/api/v1/predict" \
        --config "$CONFIG_PATH"; then
        echo "ð�̲���ʧ��"
        exit 7
    fi
}

# �쳣����
handle_error() {
    local exit_code=$?
    echo "����ʧ�ܣ��������: $exit_code"
    
    # �ع�����
    kubectl rollout undo deploy/fairllm-service -n "$NAMESPACE"
    
    # ���͸澯֪ͨ
    python scripts/send_alert.py \
        --message "����ʧ�� (Exit Code: $exit_code)" \
        --severity "critical"
    
    exit $exit_code
}

# ������
main() {
    # �������
    trap handle_error ERR
    
    # ��ʼ��
    validate_model
    build_service
    
    # ����
    deploy_k8s
    setup_monitoring
    
    # ��֤
    validate_deployment
    
    # ����
    docker rmi "$BUILD_TAG"
    docker rmi "$IMAGE_TAG"
    
    echo "����ɹ�"
}

# ִ��������
main
