#!/bin/bash
# requirements/docker/k8s/recovery_strategy.sh

set -euo pipefail

# ��ȡ������״̬��Pod
kubectl get pods -l app=fairllm-service -o json | \
jq -r '.items[] | select(.status.phase != "Running") | .metadata.name' | \
while read -r pod_name; do
    echo "Pod $pod_name is not running, attempting recovery..."
    
    # ɾ��Pod�Դ����ؽ�
    kubectl delete pod "$pod_name"
    
    # ����ؽ�״̬
    if ! kubectl wait --for=condition=Ready pod "$pod_name" --timeout=300s; then
        echo "Recovery of $pod_name failed"
        exit 1
    fi
    
    echo "Pod $pod_name recovered successfully"
done

# ���������彡��״̬
if ! kubectl get deployment fairllm-service -o jsonpath='{.status.availableReplicas}' | grep -q '[1-9]'; then
    echo "Service is not healthy"
    exit 1
fi

echo "Recovery completed successfully"
