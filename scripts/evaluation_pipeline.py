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
    """��ά��������ˮ�ߣ�֧�ֲ��л��������Զ���������"""
    
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
        """��ʼ��ʵ�����"""
        wandb.init(
            project="fairllm-eval",
            config=self.config,
            dir=str(self.output_dir),
            tags=["automated", "bias_audit"]
        )

    def _init_prometheus(self):
        """��ʼ�� Prometheus �ͻ���"""
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
        """��ʼ�� Kubernetes �ͻ���"""
        try:
            kubernetes.config.load_incluster_config()
        except:
            kubernetes.config.load_kube_config()
        self.k8s_api = kubernetes.client.BatchV1Api()

    def _load_components(self):
        """��̬�����������"""
        self.datasets = [
            self._import_component(d['type'], d['path']) 
            for d in self.config['datasets']
        ]
        self.metrics = [
            self._import_component(m['type'], m['path'])
            for m in self.config['metrics']
        ]

    def _import_component(self, comp_type: str, path: str):
        """��̬�����������"""
        module_path = Path(path)
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, comp_type)(self.config)

    def run(self):
        """ִ�ж�׶�����"""
        with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
            # ��һ�׶Σ�����ָ������
            futures = [
                executor.submit(self._evaluate_dataset, dataset)
                for dataset in self.datasets
            ]
            results = [f.result() for f in tqdm(futures, desc="���ݼ�����")]
            
            # �ڶ��׶Σ��������
            cross_results = self._cross_analysis(results)
            
            # �����׶Σ����ɱ���
            report = self._generate_report(results + cross_results)
            
        wandb.log(report)
        self._upload_results(report)
        self._push_to_prometheus(report)

    def _evaluate_dataset(self, dataset):
        """�����ݼ���������"""
        try:
            loader = dataset.get_loader(batch_size=self.config['batch_size'])
            scores = defaultdict(list)
            
            for batch in tqdm(loader, desc=f"���� {dataset.name}", leave=False):
                batch_results = self.model.evaluate(batch)
                for metric in self.metrics:
                    scores[metric.name].extend(metric.calculate(batch_results))
                
            # ��������͵� Prometheus
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
                title="��������",
                text=f"���ݼ� {dataset.name} ����ʧ��: {str(e)}"
            )
            return None

    def _cross_analysis(self, results):
        """�����ݼ�����"""
        # ʵ�������еĽ����������
        pass

    def _generate_report(self, results):
        """������������"""
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
        """�ϴ�������ƴ洢"""
        # �ϴ��� S3
        aws s3 sync "$LOG_DIR" "s3://fairllm-evaluation-results/$(basename "$LOG_DIR")"
        
        # �ϴ��� Kubernetes ConfigMap
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
                title="Kubernetes API ����",
                text=f"�ϴ����浽 Kubernetes ʧ��: {str(e)}"
            )

    def _push_to_prometheus(self, report):
        """��������͵� Prometheus"""
        try:
            push_to_gateway(
                'prometheus-push-gateway:9091',
                job='fairllm-evaluation',
                registry=self.prometheus_registry
            )
        except requests.exceptions.RequestException as e:
            wandb.alert(
                title="Prometheus ���ʹ���",
                text=f"���ͽ���� Prometheus ʧ��: {str(e)}"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="������ˮ��")
    parser.add_argument("--config", default="configs/experiment_configs/exp_bias_mitigation.yaml")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline(args.config)
    pipeline.run()
    
    