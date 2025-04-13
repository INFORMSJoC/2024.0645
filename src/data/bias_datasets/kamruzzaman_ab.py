# src/data/bias_datasets/kamruzzaman_ab.py
import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_from_disk
from transformers import AutoTokenizer, BatchEncoding
from zlib import adler32
from models.fairllm_core.bias_detectors import MultidimensionalBiasDetector
from src.data.synthetic_data.demographic_balancer import DemographicBalancer

class KamruzzamanABDataset(Dataset):
    """Kamruzzaman多维度偏见数据集加载器，支持动态偏见缓解策略"""
    
    def __init__(self, config: dict, split: str = 'train'):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        self.bias_detector = MultidimensionalBiasDetector(config['bias_config'])
        self.balancer = DemographicBalancer(config['balance_config'])
        
        # 加载原始数据并构建内存映射索引
        self.data_path = os.path.join(config['data_root'], f"{split}_memmap")
        self._init_memory_mapping()
        
        # 动态数据增强配置
        self.augmenter = self._init_augmenter(config['augment_config']) if config['augment'] else None
        
        # 缓存管理
        self.cache = {}
        self.cache_size = config.get('cache_size', 1000)
        self.enable_cache = config.get('enable_cache', True)

    def _init_memory_mapping(self):
        """初始化内存映射文件系统"""
        if not os.path.exists(self.data_path):
            raw_data = load_from_disk(os.path.join(self.config['data_root'], "raw"))
            self._convert_to_memmap(raw_data)
            
        self.meta = np.load(os.path.join(self.data_path, "meta.npz"))
        self.texts = np.memmap(
            os.path.join(self.data_path, "texts.memmap"),
            dtype='uint16',
            mode='r',
            shape=tuple(self.meta['text_shape'])
        )
        self.labels = np.memmap(
            os.path.join(self.data_path, "labels.memmap"),
            dtype=np.float32,
            mode='r',
            shape=tuple(self.meta['label_shape'])
        )

    def _convert_to_memmap(self, raw_data):
        """将原始数据集转换为内存映射格式"""
        os.makedirs(self.data_path, exist_ok=True)
        
        # 文本量化处理
        text_encoder = self.tokenizer.get_vocab()
        encoded_texts = []
        for item in raw_data:
            tokens = self.tokenizer.encode(item['text'], add_special_tokens=False)
            encoded_texts.append(np.array(tokens, dtype='uint16'))
        
        max_len = max(len(t) for t in encoded_texts)
        text_memmap = np.memmap(
            os.path.join(self.data_path, "texts.memmap"),
            dtype='uint16',
            mode='w+',
            shape=(len(encoded_texts), max_len)
        )
        for i, txt in enumerate(encoded_texts):
            text_memmap[i, :len(txt)] = txt
            text_memmap[i, len(txt):] = self.tokenizer.pad_token_id
            
        # 标签处理
        label_memmap = np.memmap(
            os.path.join(self.data_path, "labels.memmap"),
            dtype=np.float32,
            mode='w+',
            shape=(len(raw_data), len(raw_data.features['bias_labels'].feature.names))
        )
        label_memmap[:] = np.array([item['bias_labels'] for item in raw_data])
        
        # 保存元数据
        np.savez(
            os.path.join(self.data_path, "meta.npz"),
            text_shape=text_memmap.shape,
            label_shape=label_memmap.shape
        )

    def __getitem__(self, index: int) -> dict:
        # 缓存检查
        cache_key = adler32(str(index).encode()) % self.cache_size
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        # 内存映射读取
        text_ids = self.texts[index].astype(np.int32)
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        labels = self.labels[index].copy()
        
        # 动态增强
        if self.augmenter and np.random.rand() < self.config['augment_prob']:
            text = self.augmenter.augment(text)
            
        # 实时偏见分析
        bias_report = self.bias_detector.detect(text)
        
        item = {
            'text': text,
            'labels': labels,
            'bias_metrics': bias_report['fairness_metrics'],
            'demographics': self._extract_demographics(text),
            'original_id': index
        }
        
        # 更新缓存
        if self.enable_cache:
            self.cache[cache_key] = item
            
        return item

    def __len__(self) -> int:
        return self.texts.shape[0]

    def get_balanced_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """获取平衡数据加载器"""
        # 计算样本权重
        all_indices = range(len(self))
        features = np.array([self._demographic_to_vector(self[i]['demographics']) for i in all_indices])
        weights = self.balancer.reweight_samples(features)
        
        # 加权采样器
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(self),
            replacement=True
        )
        
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

    def _collate_fn(self, batch: list) -> BatchEncoding:
        """动态批处理与编码"""
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])
        
        # 动态截断与填充
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            max_length=self.config['max_length'],
            truncation=True,
            return_tensors='pt'
        )
        
        return BatchEncoding({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'bias_metrics': torch.tensor([item['bias_metrics'] for item in batch])
        })

class MultimodalKamruzzamanDataset(KamruzzamanABDataset):
    """多模态扩展版本，支持图像文本联合分析"""
    
    def __init__(self, config: dict, split: str = 'train'):
        super().__init__(config, split)
        self.image_processor = AutoImageProcessor.from_pretrained(config['image_model'])
        self.image_memmap = np.memmap(
            os.path.join(self.data_path, "images.memmap"),
            dtype=np.float16,
            mode='r',
            shape=tuple(self.meta['image_shape'])
        )
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        
        # 加载图像特征
        image_features = self.image_memmap[index].copy()
        item['image'] = torch.tensor(image_features)
        
        return item

    def _collate_fn(self, batch):
        base_batch = super()._collate_fn(batch)
        images = torch.stack([item['image'] for item in batch])
        return BatchEncoding({**base_batch, 'images': images})

class StreamingKamruzzamanLoader:
    """流式大数据加载器，支持实时过滤与动态平衡"""
    
    def __init__(self, config: dict):
        self.config = config
        self.shard_pattern = os.path.join(config['data_root'], "shard-*.parquet")
        self.current_shard = 0
        self.buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        
        # 初始化动态过滤器
        self.filters = [
            BiasThresholdFilter(config['bias_thresholds']),
            DemographicBalancer(config['balance_config']),
            QualityFilter(min_length=config['min_length'])
        ]
        
    def __iter__(self):
        while True:
            if not self.buffer:
                self._load_next_shard()
                
            sample = self.buffer.pop(0)
            if self._apply_filters(sample):
                yield sample

    def _load_next_shard(self):
        """加载下一个数据分片并进行预处理"""
        shard_path = self.shard_pattern.replace('*', str(self.current_shard))
        dataset = load_from_disk(shard_path)
        
        # 并行预处理
        with ThreadPoolExecutor(max_workers=8) as executor:
            self.buffer = list(executor.map(self._preprocess_sample, dataset))
            
        self.current_shard = (self.current_shard + 1) % self.config['total_shards']

    def _preprocess_sample(self, sample):
        """样本预处理管道"""
        # 文本清洗
        sample['text'] = re.sub(r'\s+', ' ', sample['text']).strip()
        
        # 图像预处理
        if 'image' in sample:
            sample['image'] = self.image_processor(sample['image'])
            
        return sample

    def _apply_filters(self, sample) -> bool:
        """应用动态过滤链"""
        return all(f(sample) for f in self.filters)
