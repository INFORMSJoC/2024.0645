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
    """控制器，支持动态配置管理与调度"""
    
    def __init__(self, config_root: str = "configs", 
                 max_parallel: int = 4,
                 db_uri: str = "mongodb://experiments:27017/"):
        self.config_root = Path(config_root)
        self.max_parallel = max_parallel
        self.experiment_registry = {}
        
        # 初始化神圣实验框架
        self.ex = Experiment('fairllm_experiments')
        self.ex.observers.append(MongoObserver(url=db_uri, db_name='fairllm'))
        
        # 创建实验目录结构
        self._init_experiment_folders()
        
    def _init_experiment_folders(self):
        """初始化实验目录结构"""
        (self.config_root / "config_overrides").mkdir(exist_ok=True)
        (self.config_root / "results_analysis").mkdir(exist_ok=True)
        
    def register_experiment(self, name: str, params: Dict, 
                           priority: int = 0) -> str:
        """注册新实验到控制器"""
        exp_id = f"{name}_{uuid.uuid4().hex[:6]}"
        self.experiment_registry[exp_id] = {
            'params': params,
            'status': 'pending',
            'priority': priority
        }
        return exp_id
    
    @ex.automain
    def run_experiment(self, _config: DictConfig):
        """执行单个实验的核心逻辑"""
        # 动态合并配置
        config = self._merge_configs(_config)
        
        # 实验资源分配
        self._acquire_resources(config)
        
        try:
            # 训练流程
            trainer = self._init_trainer(config)
            metrics = trainer.train()
            
            # 评估流程
            evaluator = Evaluator(config)
            results = evaluator.evaluate()
            
            # 保存结果
            self._save_results(exp_id=config.exp_id, 
                             metrics={**metrics, **results})
            
            return results
        except Exception as e:
            self._handle_experiment_failure(config.exp_id, e)
        finally:
            self._release_resources(config)
    
    def run_all(self, strategy: str = "priority"):
        """执行所有注册实验"""
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
        """根据策略排序实验"""
        if strategy == "priority":
            return sorted(self.experiment_registry.keys(),
                         key=lambda x: -self.experiment_registry[x]['priority'])
        elif strategy == "resource_asc":
            return sorted(self.experiment_registry.keys(),
                         key=lambda x: self._calc_resource_cost(x))
        else:
            return list(self.experiment_registry.keys())
    
    def _generate_exp_config(self, exp_id: str) -> DictConfig:
        """生成实验专属配置"""
        base_config = OmegaConf.load(self.config_root / "base_config.yaml")
        override = OmegaConf.create(self.experiment_registry[exp_id]['params'])
        return OmegaConf.merge(base_config, override, {"exp_id": exp_id})
    
    def _merge_configs(self, config: DictConfig) -> DictConfig:
        """动态合并运行时配置"""
        with initialize_config_dir(str(self.config_root)):
            overrides = [f"{k}={v}" for k, v in config.items()]
            return compose(config_name="main", overrides=overrides)
    
    def _acquire_resources(self, config: DictConfig):
        """申请实验资源（GPU、内存等）"""
        # 实现资源预留逻辑
        pass
    
    def _save_results(self, exp_id: str, metrics: Dict):
        """保存实验结果到数据库和文件系统"""
        # 保存到MongoDB
        self.ex.current_run.log_scalars(metrics)
        
        # 保存到Parquet文件
        df = pd.DataFrame([metrics])
        df.to_parquet(self.config_root / f"./results_analysis/{exp_id}.parquet")
        
        # 更新注册表状态
        self.experiment_registry[exp_id]['status'] = 'completed'
        self.experiment_registry[exp_id]['results'] = metrics

class ConfigOverrideGenerator:
    """动态配置生成器，支持网格搜索和随机采样"""
    
    def __init__(self, base_config: str):
        self.base = OmegaConf.load(base_config)
        self.override_dir = Path(base_config).parent / "config_overrides"
        self.override_dir.mkdir(exist_ok=True)
        
    def generate_grid(self, param_grid: Dict) -> List[str]:
        """生成网格搜索配置"""
        grid = ParameterGrid(param_grid)
        config_paths = []
        
        for i, params in enumerate(grid):
            override = OmegaConf.create(params)
            path = self.override_dir / f"grid_{i}.yaml"
            OmegaConf.save(override, path)
            config_paths.append(str(path))
            
        return config_paths
    
    def generate_random(self, param_dist: Dict, n_samples: int) -> List[str]:
        """生成随机采样配置"""
        config_paths = []
        
        for i in range(n_samples):
            sample = {k: np.random.choice(v) if isinstance(v, list) else v 
                     for k, v in param_dist.items()}
            override = OmegaConf.create(sample)
            path = self.override_dir / f"random_{i}.yaml"
            OmegaConf.save(override, path)
            config_paths.append(str(path))
            
        return config_paths
