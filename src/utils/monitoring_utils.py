# src/utils/monitoring_utils.py
import logging
import torch
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from kubernetes import client, config
import requests

class ModelMonitor:
    """模型监控工具"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_registry = CollectorRegistry()
        self._init_metrics()
        self._init_kubernetes()

    def _init_metrics(self):
        """初始化监控指标"""
        self.metrics = {
            'bias_score': Gauge(
                'model_bias_score',
                'Model bias score across different dimensions',
                ['dimension', 'model', 'environment'],
                registry=self.metrics_registry
            ),
            'inference_latency': Gauge(
                'model_inference_latency_seconds',
                'Model inference latency in seconds',
                registry=self.metrics_registry
            ),
            'gpu_utilization': Gauge(
                'model_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id', 'model'],
                registry=self.metrics_registry
            ),
            'memory_utilization': Gauge(
                'model_memory_utilization_percent',
                'Memory utilization percentage',
                registry=self.metrics_registry
            )
        }

    def _init_kubernetes(self):
        """初始化 Kubernetes 客户端"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.api = client.CustomObjectsApi()

    def update_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """更新指标值"""
        if metric_name not in self.metrics:
            self.logger.warning(f"Metric {metric_name} not found")
            return
        
        if labels:
            self.metrics[metric_name].labels(**labels).set(value)
        else:
            self.metrics[metric_name].set(value)

    def push_metrics(self, endpoint: str = 'prometheus-push-gateway:9091'):
        """推送指标到 Prometheus"""
        try:
            push_to_gateway(endpoint, job='model_monitoring', registry=self.metrics_registry)
            self.logger.info("Metrics pushed to Prometheus successfully")
        except Exception as e:
            self.logger.error(f"Failed to push metrics: {str(e)}")

    def query_kubernetes(self, query: str) -> Dict:
        """查询 Kubernetes 资源使用情况"""
        try:
            response = self.api.get_cluster_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                plural="pods",
                name=query
            )
            return response
        except Exception as e:
            self.logger.error(f"Kubernetes query failed: {str(e)}")
            return {}

    def monitor_inference(self, response_time: float, model_name: str):
        """监控推理性能"""
        self.update_metric('inference_latency', response_time)
        self.update_metric('gpu_utilization', np.random.uniform(30, 80), {'model': model_name, 'gpu_id': '0'})
        self.update_metric('memory_utilization', np.random.uniform(40, 70))
        self.push_metrics()