# results/experiment_controller.py
import os
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import yaml
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig
from ray.util.multiprocessing import Pool
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid

class ExperimentController:
    """��������֧�ֶ�̬���ù��������"""
    
    def __init__(self, config_root: str = "configs", 
                 max_parallel: int = 4,
                 db_uri: str = "mongodb://experiments:27017/"):
        self.config_root = Path(config_root)
        self.max_parallel = max_parallel
        self.experiment_registry = {}
        
        # ��ʼ����ʥʵ����
        self.ex = Experiment('fairllm_experiments')
        self.ex.observers.append(MongoObserver(url=db_uri, db_name='fairllm'))
        
        # ����ʵ��Ŀ¼�ṹ
        self._init_experiment_folders()
        
    def _init_experiment_folders(self):
        """��ʼ��ʵ��Ŀ¼�ṹ"""
        (self.config_root / "config_overrides").mkdir(exist_ok=True)
        (self.config_root / "results_analysis").mkdir(exist_ok=True)
        
    def register_experiment(self, name: str, params: Dict, 
                           priority: int = 0) -> str:
        """ע����ʵ�鵽������"""
        exp_id = f"{name}_{uuid.uuid4().hex[:6]}"
        self.experiment_registry[exp_id] = {
            'params': params,
            'status': 'pending',
            'priority': priority
        }
        return exp_id
    
    @ex.automain
    def run_experiment(self, _config: DictConfig):
        """ִ�е���ʵ��ĺ����߼�"""
        # ��̬�ϲ�����
        config = self._merge_configs(_config)
        
        # ʵ����Դ����
        self._acquire_resources(config)
        
        try:
            # ѵ������
            trainer = self._init_trainer(config)
            metrics = trainer.train()
            
            # ��������
            evaluator = Evaluator(config)
            results = evaluator.evaluate()
            
            # ������
            self._save_results(exp_id=config.exp_id, 
                             metrics={**metrics, **results})
            
            return results
        except Exception as e:
            self._handle_experiment_failure(config.exp_id, e)
        finally:
            self._release_resources(config)
    
    def run_all(self, strategy: str = "priority"):
        """ִ������ע��ʵ��"""
        sorted_exps = self._sort_experiments(strategy)
        
        with Pool(self.max_parallel) as pool:
            futures = []
            for exp_id in sorted_exps:
                config = self._generate_exp_config(exp_id)
                future = pool.apply_async(self._run_single, (config,))
                futures.append(future)
                
            for future in futures:
                try:
                    future.get()
                except Exception as e:
                    print(f"Experiment failed: {str(e)}")
    
    def _sort_experiments(self, strategy: str) -> List[str]:
        """���ݲ�������ʵ��"""
        if strategy == "priority":
            return sorted(self.experiment_registry.keys(),
                         key=lambda x: -self.experiment_registry[x]['priority'])
        elif strategy == "resource_asc":
            return sorted(self.experiment_registry.keys(),
                         key=lambda x: self._calc_resource_cost(x))
        else:
            return list(self.experiment_registry.keys())
    
    def _generate_exp_config(self, exp_id: str) -> DictConfig:
        """����ʵ��ר������"""
        base_config = OmegaConf.load(self.config_root / "base_config.yaml")
        override = OmegaConf.create(self.experiment_registry[exp_id]['params'])
        return OmegaConf.merge(base_config, override, {"exp_id": exp_id})
    
    def _merge_configs(self, config: DictConfig) -> DictConfig:
        """��̬�ϲ�����ʱ����"""
        with initialize_config_dir(str(self.config_root)):
            overrides = [f"{k}={v}" for k, v in config.items()]
            return compose(config_name="main", overrides=overrides)
    
    def _acquire_resources(self, config: DictConfig):
        """����ʵ����Դ��GPU���ڴ�ȣ�"""
        # ʵ����ԴԤ���߼�
        pass
    
    def _save_results(self, exp_id: str, metrics: Dict):
        """����ʵ���������ݿ���ļ�ϵͳ"""
        # ���浽MongoDB
        self.ex.current_run.log_scalars(metrics)
        
        # ���浽Parquet�ļ�
        df = pd.DataFrame([metrics])
        df.to_parquet(self.config_root / f"./results_analysis/{exp_id}.parquet")
        
        # ����ע���״̬
        self.experiment_registry[exp_id]['status'] = 'completed'
        self.experiment_registry[exp_id]['results'] = metrics

class ConfigOverrideGenerator:
    """��̬������������֧�������������������"""
    
    def __init__(self, base_config: str):
        self.base = OmegaConf.load(base_config)
        self.override_dir = Path(base_config).parent / "config_overrides"
        self.override_dir.mkdir(exist_ok=True)
        
    def generate_grid(self, param_grid: Dict) -> List[str]:
        """����������������"""
        grid = ParameterGrid(param_grid)
        config_paths = []
        
        for i, params in enumerate(grid):
            override = OmegaConf.create(params)
            path = self.override_dir / f"grid_{i}.yaml"
            OmegaConf.save(override, path)
            config_paths.append(str(path))
            
        return config_paths
    
    def generate_random(self, param_dist: Dict, n_samples: int) -> List[str]:
        """���������������"""
        config_paths = []
        
        for i in range(n_samples):
            sample = {k: np.random.choice(v) if isinstance(v, list) else v 
                     for k, v in param_dist.items()}
            override = OmegaConf.create(sample)
            path = self.override_dir / f"random_{i}.yaml"
            OmegaConf.save(override, path)
            config_paths.append(str(path))
            
        return config_paths
