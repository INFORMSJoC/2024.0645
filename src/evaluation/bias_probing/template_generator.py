# src/evaluation/bias_probing/template_generator.py
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict
import json
import hashlib

class AgeBiasTemplateEngine:
    """动态年龄偏见模板生成（支持上下文扰动）"""
    def __init__(self, 
                 #template_dir: str = "templates/age_bias",
                 template_dir: str = "../templates/",
                 perturbation_rules: Dict[str, List[str]] = None):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.perturbation_rules = perturbation_rules or self._default_perturbations()
        self.config = self._load_scenario_config()
    
    def _load_scenario_config(self) -> Dict:
        """加载场景配置文件"""
        with open(f"../templates/config.yaml") as f:
            return yaml.safe_load(f)
    
    def _default_perturbations(self) -> Dict[str, List[str]]:
        """默认上下文扰动规则（符合WHO年龄歧视分类标准）"""
        return {
            'age_terms': ["young", "elderly", "middle-aged"],
            'implicit_bias_triggers': ["tech-savvy", "experienced", "slow learner"]
        }
    
    def generate(self, 
                scenario: str,
                age_groups: List[str],
                perturbation_level: int = 2) -> List[Dict]:
        """生成多维度偏见探测模板"""
        template = self.env.get_template(f"{scenario}.jinja")
        probes = []
        
        for age in age_groups:
            for perturb in self._select_perturbations(perturbation_level):
                context = template.render(
                    age_group=age,
                    perturbation=perturb,
                    scenario_specific=self.config[scenario]
                )
                probes.append({
                    'id': self._generate_hash(context),
                    'context': context,
                    'expected_unbiased': True,
                    'metadata': {
                        'age_group': age,
                        'perturbation_type': perturb['type'],
                        'severity_level': perturb['level']
                    }
                })
        return probes
    
    def _select_perturbations(self, level: int) -> List[Dict]:
        """基于扰动级别选择干扰因素（论文表3的实现）"""
        perturbations = []
        for term in self.perturbation_rules['age_terms']:
            perturbations.append({
                'type': 'explicit',
                'level': level,
                'term': term
            })
        for trigger in self.perturbation_rules['implicit_bias_triggers']:
            perturbations.append({
                'type': 'implicit',
                'level': level,
                'term': trigger
            })
        return perturbations
    
    def _generate_hash(self, text: str) -> str:
        """生成唯一的模板ID"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
