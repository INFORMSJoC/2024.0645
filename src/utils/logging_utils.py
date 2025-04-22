# src/utils/logging_utils.py
import logging
import os
import sys
from typing import Dict, Any
import yaml
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class Logger:
    """统一日志管理工具"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loggers = {}
        self._init_logging()
        self._init_prometheus()

    def _init_logging(self):
        """初始化日志系统"""
        logging.basicConfig(
            level=self.config.get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.get('log_file', 'app.log'))
            ]
        )
        
        # 创建自定义日志器
        for logger_name in self.config.get('loggers', []):
            self.loggers[logger_name] = logging.getLogger(logger_name)

    def _init_prometheus(self):
        """初始化 Prometheus 客户端"""
        self.prometheus_registry = CollectorRegistry()
        self.metrics = {
            'bias_score': Gauge(
                'fairllm_bias_score',
                'Multidimensional bias metrics',
                ['dimension', 'environment'],
                registry=self.prometheus_registry
            ),
            'inference_latency': Gauge(
                'fairllm_inference_latency_seconds',
                'Inference latency distribution',
                registry=self.prometheus_registry
            ),
            'gpu_util': Gauge(
                'fairllm_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id'],
                registry=self.prometheus_registry
            )
        }

    def get_logger(self, name: str) -> logging.Logger:
        """获取日志器实例"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def log(self, name: str, level: str, message: str):
        """记录日志"""
        logger = self.get_logger(name)
        if level == 'debug':
            logger.debug(message)
        elif level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'critical':
            logger.critical(message)

    def push_to_prometheus(self, endpoint: str = 'prometheus-push-gateway:9091'):
        """将日志推送到 Prometheus"""
        try:
            push_to_gateway(endpoint, job='fairllm-logging', registry=self.prometheus_registry)
        except Exception as e:
            self.log('prometheus', 'error', f"Push to Prometheus failed: {str(e)}")

    def log_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标"""
        if name in self.metrics:
            if labels:
                self.metrics[name].labels(**labels).set(value)
            else:
                self.metrics[name].set(value)
