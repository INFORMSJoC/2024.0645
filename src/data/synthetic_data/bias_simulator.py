# src/data/synthetic_data/bias_simulator.py
import random
from typing import Dict, List
import numpy as np
import torch
from transformers import pipeline
from faker import Faker
from models.fairllm_core.bias_detectors import BiasDetectorPipeline

class BiasSimulator:
    """多维度偏见模拟生成器，集成语言模型与规则注入"""
    
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
        
        # 加载偏见模板
        with open(config['bias_templates_path']) as f:
            self.templates = json.load(f)
            
        # 初始化人口分布模型
        self.demographic_dist = {
            'age': lambda: int(np.random.normal(35, 15)),
            'gender': lambda: random.choice(['M','F','NB']),
            'ethnicity': lambda: random.choices(
                ['White', 'Black', 'Asian', 'Hispanic'],
                weights=[0.6, 0.15, 0.15, 0.1]
            )[0]
        }

    def generate_batch(self, batch_size: int, target_bias: Dict) -> List[Dict]:
        """生成具有指定偏见特征的数据批"""
        samples = []
        while len(samples) < batch_size:
            template = self._select_template(target_bias)
            prompt = self._fill_template(template)
            
            # 语言模型生成
            generated = self.generator(
                prompt,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2
            )[0]['generated_text']
            
            # 后处理与验证
            cleaned = self._postprocess(generated)
            if self._validate_bias(cleaned, target_bias):
                samples.append({
                    'text': cleaned,
                    'bias_label': target_bias,
                    'demographics': self._generate_demographics()
                })
                
        return samples

    def _fill_template(self, template: Dict) -> str:
        """填充动态模板参数"""
        params = {
            'name': self.faker.name(),
            'age': self.demographic_dist['age'](),
            'gender': self._map_gender_pronoun(self.demographic_dist['gender']()),
            'ethnicity': self.demographic_dist['ethnicity'](),
            'occupation': self.faker.job()
        }
        return template['text'].format(**params)

    def _validate_bias(self, text: str, target: Dict) -> bool:
        """验证生成的文本符合目标偏见特征"""
        report = self.detector.detect(text)
        return all(
            report['ml_scores'][bias_type] >= target[bias_type]
            for bias_type in target.keys()
        )

class DemographicBalancer:
    """人口统计平衡器，实现对抗性重加权与动态采样"""
    
    def __init__(self, demographic_config: Dict):
        self.feature_weights = demographic_config['feature_weights']
        self.history = defaultdict(list)
        self.adversarial_network = self._build_adversarial_network(
            input_dim=len(self.feature_weights),
            hidden_dim=64
        )
        
    def reweight_samples(self, samples: List[Dict]) -> List[float]:
        """计算对抗性平衡权重"""
        # 提取人口特征向量
        feature_vectors = np.array([
            [s['demographics'][f] for f in self.feature_weights.keys()]
            for s in samples
        ])
        
        # 对抗网络预测权重
        with torch.no_grad():
            weights = self.adversarial_network(
                torch.tensor(feature_vectors, dtype=torch.float32)
            ).sigmoid().numpy()
            
        # 更新历史分布
        self._update_distribution(feature_vectors)
        return weights.squeeze()

    def _build_adversarial_network(self, input_dim: int, hidden_dim: int) -> torch.nn.Module:
        """构建对抗平衡网络"""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
