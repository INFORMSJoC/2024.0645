#!/bin/bash
# requirements/docker/k8s/recovery_strategy.sh

set -euo pipefail

# 获取非运行状态的Pod
kubectl get pods -l app=fairllm-service -o json | \
jq -r '.items[] | select(.status.phase != "Running") | .metadata.name' | \
while read -r pod_name; do
    echo "Pod $pod_name is not running, attempting recovery..."
    
    # 删除Pod以触发重建
    kubectl delete pod "$pod_name"
    
    # 检查重建状态
    if ! kubectl wait --for=condition=Ready pod "$pod_name" --timeout=300s; then
        echo "Recovery of $pod_name failed"
        exit 1
    fi
    
    echo "Pod $pod_name recovered successfully"
done

# 检查服务整体健康状态
if ! kubectl get deployment fairllm-service -o jsonpath='{.status.availableReplicas}' | grep -q '[1-9]'; then
    echo "Service is not healthy"
    exit 1
fi

echo "Recovery completed successfully"
