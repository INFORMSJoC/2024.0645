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
    """Kamruzzaman��ά��ƫ�����ݼ���������֧�ֶ�̬ƫ���������"""
    
    def __init__(self, config: dict, split: str = 'train'):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        self.bias_detector = MultidimensionalBiasDetector(config['bias_config'])
        self.balancer = DemographicBalancer(config['balance_config'])
        
        # ����ԭʼ���ݲ������ڴ�ӳ������
        self.data_path = os.path.join(config['data_root'], f"{split}_memmap")
        self._init_memory_mapping()
        
        # ��̬������ǿ����
        self.augmenter = self._init_augmenter(config['augment_config']) if config['augment'] else None
        
        # �������
        self.cache = {}
        self.cache_size = config.get('cache_size', 1000)
        self.enable_cache = config.get('enable_cache', True)

    def _init_memory_mapping(self):
        """��ʼ���ڴ�ӳ���ļ�ϵͳ"""
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
        """��ԭʼ���ݼ�ת��Ϊ�ڴ�ӳ���ʽ"""
        os.makedirs(self.data_path, exist_ok=True)
        
        # �ı���������
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
            
        # ��ǩ����
        label_memmap = np.memmap(
            os.path.join(self.data_path, "labels.memmap"),
            dtype=np.float32,
            mode='w+',
            shape=(len(raw_data), len(raw_data.features['bias_labels'].feature.names))
        )
        label_memmap[:] = np.array([item['bias_labels'] for item in raw_data])
        
        # ����Ԫ����
        np.savez(
            os.path.join(self.data_path, "meta.npz"),
            text_shape=text_memmap.shape,
            label_shape=label_memmap.shape
        )

    def __getitem__(self, index: int) -> dict:
        # ������
        cache_key = adler32(str(index).encode()) % self.cache_size
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        # �ڴ�ӳ���ȡ
        text_ids = self.texts[index].astype(np.int32)
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        labels = self.labels[index].copy()
        
        # ��̬��ǿ
        if self.augmenter and np.random.rand() < self.config['augment_prob']:
            text = self.augmenter.augment(text)
            
        # ʵʱƫ������
        bias_report = self.bias_detector.detect(text)
        
        item = {
            'text': text,
            'labels': labels,
            'bias_metrics': bias_report['fairness_metrics'],
            'demographics': self._extract_demographics(text),
            'original_id': index
        }
        
        # ���»���
        if self.enable_cache:
            self.cache[cache_key] = item
            
        return item

    def __len__(self) -> int:
        return self.texts.shape[0]

    def get_balanced_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """��ȡƽ�����ݼ�����"""
        # ��������Ȩ��
        all_indices = range(len(self))
        features = np.array([self._demographic_to_vector(self[i]['demographics']) for i in all_indices])
        weights = self.balancer.reweight_samples(features)
        
        # ��Ȩ������
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
        """��̬�����������"""
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])
        
        # ��̬�ض������
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
    """��ģ̬��չ�汾��֧��ͼ���ı����Ϸ���"""
    
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
        
        # ����ͼ������
        image_features = self.image_memmap[index].copy()
        item['image'] = torch.tensor(image_features)
        
        return item

    def _collate_fn(self, batch):
        base_batch = super()._collate_fn(batch)
        images = torch.stack([item['image'] for item in batch])
        return BatchEncoding({**base_batch, 'images': images})

class StreamingKamruzzamanLoader:
    """��ʽ�����ݼ�������֧��ʵʱ�����붯̬ƽ��"""
    
    def __init__(self, config: dict):
        self.config = config
        self.shard_pattern = os.path.join(config['data_root'], "shard-*.parquet")
        self.current_shard = 0
        self.buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        
        # ��ʼ����̬������
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
        """������һ�����ݷ�Ƭ������Ԥ����"""
        shard_path = self.shard_pattern.replace('*', str(self.current_shard))
        dataset = load_from_disk(shard_path)
        
        # ����Ԥ����
        with ThreadPoolExecutor(max_workers=8) as executor:
            self.buffer = list(executor.map(self._preprocess_sample, dataset))
            
        self.current_shard = (self.current_shard + 1) % self.config['total_shards']

    def _preprocess_sample(self, sample):
        """����Ԥ����ܵ�"""
        # �ı���ϴ
        sample['text'] = re.sub(r'\s+', ' ', sample['text']).strip()
        
        # ͼ��Ԥ����
        if 'image' in sample:
            sample['image'] = self.image_processor(sample['image'])
            
        return sample

    def _apply_filters(self, sample) -> bool:
        """Ӧ�ö�̬������"""
        return all(f(sample) for f in self.filters)
