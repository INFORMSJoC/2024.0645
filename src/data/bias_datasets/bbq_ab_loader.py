# src/data/bias_datasets/bbq_ab_loader.py
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from itertools import combinations
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from datasets import load_dataset
from models.fairllm_core.bias_detectors import MultidimensionalBiasDetector

class BBQABDataset(Dataset):
    """BBQƫ�����ݼ���������֧�ֶ�̬ƫ������������ƽ���ƫ���Ŵ�"""
    
    def __init__(self, config: Dict, split: str = 'train'):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        self.bias_amplification = config.get('bias_amplification', 1.0)
        self.bias_detector = MultidimensionalBiasDetector(config['bias_config'])
        
        # ���ز�Ԥ����ԭʼ����
        if config['data_source'] == 'huggingface':
            raw_data = load_dataset('bbq_ab', split=split)
        else:
            with open(config['data_path']) as f:
                raw_data = json.load(f)[split]
        
        self.samples = self._process_raw_data(raw_data)
        
        # ����ƽ�⴦��
        if config['balance_strategy'] == 'stratified':
            self._stratified_balance()
        elif config['balance_strategy'] == 'oversample':
            self._oversample_minority()
            
        # �����Կ�����
        if config.get('adversarial_examples'):
            self.samples = self._inject_adversarial_examples(self.samples)
            
        # �������ǩ����
        self.label_mappers = self._create_label_hierarchy()

    def _process_raw_data(self, raw_data) -> List[Dict]:
        """ִ�ж�׶�����Ԥ����"""
        processed = []
        for item in raw_data:
            # ƫ������
            bias_report = self.bias_detector.detect(item['context'])
            
            processed.append({
                'text': self._clean_text(item['context']),
                'label': item['label'],
                'bias_metrics': bias_report['fairness_metrics'],
                'demographics': self._extract_demographics(item['context']),
                'original_id': item.get('id', None)
            })
        return processed

    def _clean_text(self, text: str) -> str:
        """�༶�ı���ϴ�ܵ�"""
        text = re.sub(r'\[.*?\]', '', text)  # �Ƴ���ע
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower() if self.config['lowercase'] else text

    def _stratified_balance(self):
        """����ƫ��ָ��ķֲ����ƽ��"""
        stratify_labels = [
            'high' if m['bias_score'] > 0.7 else 'medium' if m['bias_score'] > 0.4 else 'low'
            for m in self.samples
        ]
        groups = [s['demographics']['gender'] + s['demographics']['age_group'] for s in self.samples]
        
        splitter = StratifiedGroupKFold(n_splits=5)
        indices = next(splitter.split(
            X=np.zeros(len(self.samples)),
            y=stratify_labels,
            groups=groups
        ))[0]  # ȡ��һ����Ƭ��Ϊƽ�����ݼ�
        
        self.samples = [self.samples[i] for i in indices]

    def _oversample_minority(self):
        """����������Ⱥ������"""
        # ʵ�ֹ������߼�
        pass

    def _extract_demographics(self, text: str) -> Dict:
        """��ȡ�˿�ͳ��ѧ��Ϣ"""
        # ʵ���˿�ͳ��ѧ��Ϣ��ȡ�߼�
        return {'age_group': 'unknown', 'gender': 'unknown'}

    def _amplify_bias(self, text: str, bias_type: str, factor: float) -> str:
        """��������ռ������Ŷ��Ŵ�ƫ��"""
        # ʹ�ô�Ƕ���������ƫ��
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt')
            embeddings = self.tokenizer.get_input_embeddings()(inputs.input_ids)
            
            # ��ȡƫ����������
            bias_vector = self._get_bias_direction(bias_type)
            
            # Ӧ�������Ŷ�
            perturbed = embeddings + factor * bias_vector
            reconstructed = self._reconstruct_text(perturbed)
            
        return reconstructed

    def _get_bias_direction(self, bias_type: str) -> torch.Tensor:
        """��ȡƫ����������"""
        # ʵ��ƫ������������ȡ�߼�
        return torch.randn(768)  # ʾ��

    def _reconstruct_text(self, embeddings: torch.Tensor) -> str:
        """��Ƕ�������ؽ��ı�"""
        # ʵ���ı��ؽ��߼�
        return "reconstructed text"  # ʾ��

    def _create_label_hierarchy(self) -> Dict:
        """���������ȱ�ǩ��ϵ"""
        return {
            'coarse': ['age', 'gender', 'race'],
            'fine': {
                'age': ['young', 'old'],
                'gender': ['male', 'female', 'non-binary'],
                'race': ['white', 'black', 'asian']
            }
        }

    def _inject_adversarial_examples(self, data: List[Dict]) -> List[Dict]:
        """ע��Կ�����"""
        # ʵ�ֶԿ����������߼�
        return data

    def __getitem__(self, idx: int) -> Dict:
        example = self.samples[idx]
        inputs = self.tokenizer(
            example['text'],
            padding='max_length',
            max_length=self.config['max_length'],
            truncation=True,
            return_tensors='pt'
        )
        
        # �������ǩ����
        labels = {
            'main': torch.tensor(example['label']),
            'bias_type': self._encode_bias_type(example.get('bias_type', 'unknown')),
            'bias_score': torch.tensor(example.get('bias_score', 0.0))
        }
        
        return {**inputs, **labels}

    def __len__(self):
        return len(self.samples)

    def _encode_bias_type(self, bias_type: str) -> torch.Tensor:
        """����ƫ������"""
        # ʵ��ƫ�����ͱ����߼�
        return torch.tensor(0)  # ʾ��

class StratifiedBatchSampler:
    """�ֲ���������������ÿ��batch��ƫ���ֲ�ƽ��"""
    
    def __init__(self, dataset, batch_size: int, num_bins: int = 5):
        self.batch_size = batch_size
        scores = [ex.get('bias_score', 0.0) for ex in dataset.samples]
        
        # ����ƫ����������
        self.bins = pd.qcut(scores, num_bins, labels=False)
        self.indices_per_bin = [
            torch.where(torch.tensor(self.bins) == i)[0].tolist()
            for i in range(num_bins)
        ]
        
    def __iter__(self):
        # ��̬ƽ���������
        batches = []
        for bin_indices in self.indices_per_bin:
            np.random.shuffle(bin_indices)
            batches += [bin_indices[i:i+self.batch_size] 
                       for i in range(0, len(bin_indices), self.batch_size)]
        
        np.random.shuffle(batches)
        return iter(batches)

class DynamicDataLoader(DataLoader):
    """��̬���ݼ�����������ʵʱ������ǿ�뻺���Ż�"""
    
    def __init__(self, dataset, batch_size: int, num_workers: int, 
                 augmenter: callable = None, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True,
            **kwargs
        )
        self.augmenter = augmenter
        self._init_shared_cache()

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """��̬����������ǿ"""
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        if self.augmenter:
            texts = [self.augmenter(t) for t in texts]
            
        tokenized = self.dataset.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.dataset.config['max_length'],
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': torch.tensor(labels),
            'bias_metrics': torch.stack([torch.tensor(item.get('bias_metrics', 0.0)) for item in batch])
        }

    def _init_shared_cache(self):
        """��ʼ������̹����ڴ滺��"""
        manager = torch.multiprocessing.Manager()
        self.shared_cache = manager.dict()
        self.cache_lock = manager.Lock()
