# src/utils/prometheus_utils.py
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_http_client import Prometheus
import logging
from typing import Dict, Any

class PrometheusExporter:
    """Prometheus 指标导出工具类"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.prometheus_url = self.config.get('prometheus_url', 'http://prometheus:9090')
        self.push_gateway_url = self.config.get('push_gateway_url', 'http://pushgateway:9091')
        self.logger = logging.getLogger(__name__)
        
        # 初始化指标
        self._init_metrics()

    def _init_metrics(self):
        """初始化指标"""
        for metric_name, metric_config in self.config['metrics'].items():
            self.metrics[metric_name] = Gauge(
                metric_name,
                metric_config['description'],
                metric_config.get('labels', []),
                registry=self.registry
            )

    def update_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """更新指标值"""
        if metric_name not in self.metrics:
            self.logger.warning(f"Metric {metric_name} not found")
            return
        
        if labels:
            self.metrics[metric_name].labels(**labels).set(value)
        else:
            self.metrics[metric_name].set(value)

    def push_metrics(self, job_name: str = 'fairllm'):
        """推送指标到 Prometheus"""
        try:
            push_to_gateway(
                self.push_gateway_url,
                job=job_name,
                registry=self.registry
            )
            self.logger.info("Metrics pushed to Prometheus successfully")
        except Exception as e:
            self.logger.error(f"Failed to push metrics: {str(e)}")

    def query_prometheus(self, query: str) -> Dict:
        """查询 Prometheus"""
        try:
            prom = Prometheus(self.prometheus_url)
            result = prom.query(query)
            return result
        except Exception as e:
            self.logger.error(f"Prometheus query failed: {str(e)}")
            return {}
