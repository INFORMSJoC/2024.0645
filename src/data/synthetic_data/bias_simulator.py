# src/data/synthetic_data/bias_simulator.py
import random
from typing import Dict, List
import numpy as np
import torch
from transformers import pipeline
from faker import Faker
from models.fairllm_core.bias_detectors import BiasDetectorPipeline

class BiasSimulator:
    """��ά��ƫ��ģ������������������ģ�������ע��"""
    
    def __init__(self, config: Dict):
        self.generator = pipeline(
            "text-generation",
            model=config['generator_model'],
            device=config.get('device', 0),
            torch_dtype=torch.bfloat16
        )
        self.faker = Faker()
        self.bias_types = config['bias_types']
        self.detector = BiasDetectorPipeline(config['detector_config'])
        
        # ����ƫ��ģ��
        with open(config['bias_templates_path']) as f:
            self.templates = json.load(f)
            
        # ��ʼ���˿ڷֲ�ģ��
        self.demographic_dist = {
            'age': lambda: int(np.random.normal(35, 15)),
            'gender': lambda: random.choice(['M','F','NB']),
            'ethnicity': lambda: random.choices(
                ['White', 'Black', 'Asian', 'Hispanic'],
                weights=[0.6, 0.15, 0.15, 0.1]
            )[0]
        }

    def generate_batch(self, batch_size: int, target_bias: Dict) -> List[Dict]:
        """���ɾ���ָ��ƫ��������������"""
        samples = []
        while len(samples) < batch_size:
            template = self._select_template(target_bias)
            prompt = self._fill_template(template)
            
            # ����ģ������
            generated = self.generator(
                prompt,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2
            )[0]['generated_text']
            
            # ��������֤
            cleaned = self._postprocess(generated)
            if self._validate_bias(cleaned, target_bias):
                samples.append({
                    'text': cleaned,
                    'bias_label': target_bias,
                    'demographics': self._generate_demographics()
                })
                
        return samples

    def _fill_template(self, template: Dict) -> str:
        """��䶯̬ģ�����"""
        params = {
            'name': self.faker.name(),
            'age': self.demographic_dist['age'](),
            'gender': self._map_gender_pronoun(self.demographic_dist['gender']()),
            'ethnicity': self.demographic_dist['ethnicity'](),
            'occupation': self.faker.job()
        }
        return template['text'].format(**params)

    def _validate_bias(self, text: str, target: Dict) -> bool:
        """��֤���ɵ��ı�����Ŀ��ƫ������"""
        report = self.detector.detect(text)
        return all(
            report['ml_scores'][bias_type] >= target[bias_type]
            for bias_type in target.keys()
        )

class DemographicBalancer:
    """�˿�ͳ��ƽ������ʵ�ֶԿ����ؼ�Ȩ�붯̬����"""
    
    def __init__(self, demographic_config: Dict):
        self.feature_weights = demographic_config['feature_weights']
        self.history = defaultdict(list)
        self.adversarial_network = self._build_adversarial_network(
            input_dim=len(self.feature_weights),
            hidden_dim=64
        )
        
    def reweight_samples(self, samples: List[Dict]) -> List[float]:
        """����Կ���ƽ��Ȩ��"""
        # ��ȡ�˿���������
        feature_vectors = np.array([
            [s['demographics'][f] for f in self.feature_weights.keys()]
            for s in samples
        ])
        
        # �Կ�����Ԥ��Ȩ��
        with torch.no_grad():
            weights = self.adversarial_network(
                torch.tensor(feature_vectors, dtype=torch.float32)
            ).sigmoid().numpy()
            
        # ������ʷ�ֲ�
        self._update_distribution(feature_vectors)
        return weights.squeeze()

    def _build_adversarial_network(self, input_dim: int, hidden_dim: int) -> torch.nn.Module:
        """�����Կ�ƽ������"""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
