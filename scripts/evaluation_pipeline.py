# scripts/evaluation_pipeline.py
import os
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import pandas as pd
import wandb
from tqdm import tqdm
import kubernetes.client
from kubernetes.client.rest import ApiException
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class EvaluationPipeline:
    """多维度评估流水线，支持并行化评估与自动报告生成"""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config['output_root']) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_wandb()
        self._init_prometheus()
        self._load_components()
        self._init_k8s()

    def _init_wandb(self):
        """初始化实验跟踪"""
        wandb.init(
            project="fairllm-eval",
            config=self.config,
            dir=str(self.output_dir),
            tags=["automated", "bias_audit"]
        )

    def _init_prometheus(self):
        """初始化 Prometheus 客户端"""
        self.prometheus_registry = CollectorRegistry()
        self.bias_score = Gauge(
            'fairllm_bias_score',
            'Multidimensional bias metrics',
            ['dimension', 'dataset', 'environment'],
            registry=self.prometheus_registry
        )
        self.inference_latency = Gauge(
            'fairllm_inference_latency_seconds',
            'Inference latency distribution',
            registry=self.prometheus_registry
        )
        self.gpu_util = Gauge(
            'fairllm_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.prometheus_registry
        )

    def _init_k8s(self):
        """初始化 Kubernetes 客户端"""
        try:
            kubernetes.config.load_incluster_config()
        except:
            kubernetes.config.load_kube_config()
        self.k8s_api = kubernetes.client.BatchV1Api()

    def _load_components(self):
        """动态加载评估组件"""
        self.datasets = [
            self._import_component(d['type'], d['path']) 
            for d in self.config['datasets']
        ]
        self.metrics = [
            self._import_component(m['type'], m['path'])
            for m in self.config['metrics']
        ]

    def _import_component(self, comp_type: str, path: str):
        """动态导入评估组件"""
        module_path = Path(path)
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, comp_type)(self.config)

    def run(self):
        """执行多阶段评估"""
        with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
            # 第一阶段：基础指标评估
            futures = [
                executor.submit(self._evaluate_dataset, dataset)
                for dataset in self.datasets
            ]
            results = [f.result() for f in tqdm(futures, desc="数据集评估")]
            
            # 第二阶段：交叉分析
            cross_results = self._cross_analysis(results)
            
            # 第三阶段：生成报告
            report = self._generate_report(results + cross_results)
            
        wandb.log(report)
        self._upload_results(report)
        self._push_to_prometheus(report)

    def _evaluate_dataset(self, dataset):
        """单数据集评估流程"""
        try:
            loader = dataset.get_loader(batch_size=self.config['batch_size'])
            scores = defaultdict(list)
            
            for batch in tqdm(loader, desc=f"评估 {dataset.name}", leave=False):
                batch_results = self.model.evaluate(batch)
                for metric in self.metrics:
                    scores[metric.name].extend(metric.calculate(batch_results))
                
            # 将结果发送到 Prometheus
            for metric_name, values in scores.items():
                self.bias_score.labels(
                    dimension=metric_name,
                    dataset=dataset.name,
                    environment=os.getenv('ENV', 'dev')
                ).set(np.mean(values))
            
            return {
                'dataset': dataset.name,
                'metrics': {k: np.mean(v) for k, v in scores.items()},
                'raw_data': scores
            }
        except Exception as e:
            wandb.alert(
                title="评估错误",
                text=f"数据集 {dataset.name} 评估失败: {str(e)}"
            )
            return None

    def _cross_analysis(self, results):
        """跨数据集分析"""
        # 实现论文中的交叉分析方法
        pass

    def _generate_report(self, results):
        """生成评估报告"""
        report = {
            'summary': {},
            'detailed': {}
        }
        for result in results:
            if result:
                report['summary'][result['dataset']] = result['metrics']
                report['detailed'][result['dataset']] = result['raw_data']
        return report

    def _upload_results(self, report):
        """上传结果到云存储"""
        # 上传到 S3
        aws s3 sync "$LOG_DIR" "s3://fairllm-evaluation-results/$(basename "$LOG_DIR")"
        
        # 上传到 Kubernetes ConfigMap
        try:
            self.k8s_api.create_namespaced_config_map(
                namespace="fairllm",
                body={
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": f"evaluation-report-{self.run_id}"
                    },
                    "data": {
                        "report.json": json.dumps(report)
                    }
                }
            )
        except ApiException as e:
            wandb.alert(
                title="Kubernetes API 错误",
                text=f"上传报告到 Kubernetes 失败: {str(e)}"
            )

    def _push_to_prometheus(self, report):
        """将结果推送到 Prometheus"""
        try:
            push_to_gateway(
                'prometheus-push-gateway:9091',
                job='fairllm-evaluation',
                registry=self.prometheus_registry
            )
        except requests.exceptions.RequestException as e:
            wandb.alert(
                title="Prometheus 推送错误",
                text=f"推送结果到 Prometheus 失败: {str(e)}"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估流水线")
    parser.add_argument("--config", default="configs/experiment_configs/exp_bias_mitigation.yaml")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline(args.config)
    pipeline.run()
    
    