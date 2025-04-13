# src/utils/kubernetes_utils.py
import kubernetes.client
import kubernetes.config
import logging
from typing import Dict, Any

class KubernetesManager:
    """Kubernetes 管理工具类"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            kubernetes.config.load_kube_config(config_file=config_path)
        else:
            try:
                kubernetes.config.load_incluster_config()
            except:
                kubernetes.config.load_kube_config()
        
        self.api = kubernetes.client.AppsV1Api()
        self.core_api = kubernetes.client.CoreV1Api()
        self.batch_api = kubernetes.client.BatchV1Api()
        self.logger = logging.getLogger(__name__)

    def deploy_model(self, deployment_spec: Dict):
        """部署模型到 Kubernetes"""
        try:
            deployment = kubernetes.client.V1Deployment(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=deployment_spec['name'],
                    labels=deployment_spec.get('labels', {})
                ),
                spec=kubernetes.client.V1DeploymentSpec(
                    replicas=deployment_spec['replicas'],
                    selector=kubernetes.client.V1LabelSelector(
                        match_labels=deployment_spec['labels']
                    ),
                    template=kubernetes.client.V1PodTemplateSpec(
                        metadata=kubernetes.client.V1ObjectMeta(
                            labels=deployment_spec['labels']
                        ),
                        spec=kubernetes.client.V1PodSpec(
                            containers=[
                                kubernetes.client.V1Container(
                                    name=deployment_spec['container_name'],
                                    image=deployment_spec['image'],
                                    ports=[kubernetes.client.V1ContainerPort(
                                        container_port=deployment_spec['port']
                                    )],
                                    resources=kubernetes.client.V1ResourceRequirements(
                                        requests=deployment_spec.get('requests', {}),
                                        limits=deployment_spec.get('limits', {})
                                    ),
                                    env=deployment_spec.get('env', [])
                                )
                            ],
                            node_selector=deployment_spec.get('node_selector', {})
                        )
                    )
                )
            )
            
            self.api.create_namespaced_deployment(
                namespace=deployment_spec['namespace'],
                body=deployment
            )
            self.logger.info(f"Deployment {deployment_spec['name']} created successfully")
        except ApiException as e:
            self.logger.error(f"Failed to create deployment: {str(e)}")
            raise

    def scale_deployment(self, name: str, namespace: str, replicas: int):
        """调整部署的副本数"""
        try:
            self.api.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body=kubernetes.client.V1Scale(
                    spec=kubernetes.client.V1ScaleSpec(replicas=replicas)
                )
            )
            self.logger.info(f"Deployment {name} scaled to {replicas} replicas")
        except ApiException as e:
            self.logger.error(f"Failed to scale deployment: {str(e)}")
            raise

    def get_pod_logs(self, name: str, namespace: str) -> str:
        """获取 Pod 日志"""
        try:
            logs = self.core_api.read_namespaced_pod_log(
                name=name,
                namespace=namespace
            )
            return logs
        except ApiException as e:
            self.logger.error(f"Failed to get pod logs: {str(e)}")
            return ""

    def create_job(self, job_spec: Dict):
        """创建 Kubernetes Job"""
        try:
            job = kubernetes.client.V1Job(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=job_spec['name'],
                    labels=job_spec.get('labels', {})
                ),
                spec=kubernetes.client.V1JobSpec(
                    template=kubernetes.client.V1PodTemplateSpec(
                        metadata=kubernetes.client.V1ObjectMeta(
                            labels=job_spec.get('labels', {})
                        ),
                        spec=kubernetes.client.V1PodSpec(
                            containers=[
                                kubernetes.client.V1Container(
                                    name=job_spec['container_name'],
                                    image=job_spec['image'],
                                    command=job_spec.get('command', []),
                                    args=job_spec.get('args', []),
                                    env=job_spec.get('env', [])
                                )
                            ],
                            restart_policy='Never'
                        )
                    ),
                    backoff_limit=job_spec.get('backoff_limit', 3)
                )
            )
            
            self.batch_api.create_namespaced_job(
                namespace=job_spec['namespace'],
                body=job
            )
            self.logger.info(f"Job {job_spec['name']} created successfully")
        except ApiException as e:
            self.logger.error(f"Failed to create job: {str(e)}")
            raise
