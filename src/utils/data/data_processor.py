# src/utils/data/data_processor.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import logging
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Any

class DataProcessor:
    """���ݴ�������"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('tokenizer_name', 'bert-base-uncased'))
        self.bias_types = config.get('bias_types', ['age', 'gender', 'race'])
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 32)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ԥ��������"""
        # ��ϴ�ı�
        data['text'] = data['text'].apply(self._clean_text)
        
        # ��ȡ�˿�ͳ��ѧ��Ϣ
        data = self._extract_demographics(data)
        
        # ƽ�����ݼ�
        data = self._balance_dataset(data)
        
        return data

    def _clean_text(self, text: str) -> str:
        """��ϴ�ı�"""
        text = re.sub(r'\[.*?\]', '', text)  # �Ƴ���ע
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower() if self.config.get('lowercase', True) else text

    def _extract_demographics(self, data: pd.DataFrame) -> pd.DataFrame:
        """��ȡ�˿�ͳ��ѧ��Ϣ"""
        for bias_type in self.bias_types:
            data[f'{bias_type}_group'] = data['text'].apply(self._detect_demographic_group, args=(bias_type,))
        return data

    def _detect_demographic_group(self, text: str, bias_type: str) -> str:
        """����˿�ͳ��ѧȺ��"""
        # ģ�����߼�
        groups = {
            'age': ['young', 'middle', 'senior'],
            'gender': ['male', 'female', 'non-binary'],
            'race': ['white', 'black', 'asian']
        }
        return np.random.choice(groups.get(bias_type, ['unknown']))

    def _balance_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """ƽ�����ݼ�"""
        try:
            skf = StratifiedGroupKFold(n_splits=5)
            split_indices = next(skf.split(data, data['label'], data['age_group']))
            return data.iloc[split_indices[0]]
        except Exception as e:
            self.logger.error(f"Dataset balancing failed: {str(e)}")
            return data

    def create_dataloader(self, data: pd.DataFrame, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """�������ݼ�����"""
        dataset = torch.utils.data.Dataset()
        dataset.__getitem__ = lambda idx: {
            'input_ids': torch.tensor(self.tokenizer(data['text'].iloc[idx], truncation=True, max_length=self.max_length)['input_ids']),
            'labels': torch.tensor(data['label'].iloc[idx])
        }
        dataset.__len__ = lambda: len(data)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count() // 2
        )

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """��ǿ����"""
        augmented_data = []
        for _, row in data.iterrows():
            # ���ɶ�����ǿ�汾
            for _ in range(self.config.get('augmentation_factor', 3)):
                augmented_text = self._augment_text(row['text'])
                augmented_data.append({
                    'text': augmented_text,
                    'label': row['label']
                })
        return pd.concat([data, pd.DataFrame(augmented_data)], ignore_index=True)

    def _augment_text(self, text: str) -> str:
        """��ǿ�ı�"""
        # ģ����ǿ�߼�
        augmentations = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
            self._random_deletion
        ]
        augmentation = np.random.choice(augmentations)
        return augmentation(text)

    def _synonym_replacement(self, text: str) -> str:
        """ͬ����滻"""
        # ģ��ʵ��
        return text

    def _random_insertion(self, text: str) -> str:
        """�������"""
        # ģ��ʵ��
        return text

    def _random_swap(self, text: str) -> str:
        """�������"""
        # ģ��ʵ��
        return text

    def _random_deletion(self, text: str) -> str:
        """���ɾ��"""
        # ģ��ʵ��
        return text

    def mitigate_bias(self, data: pd.DataFrame) -> pd.DataFrame:
        """���������е�ƫ��"""
        for bias_type in self.bias_types:
            data = self._mitigate_bias_by_type(data, bias_type)
        return data

    def _mitigate_bias_by_type(self, data: pd.DataFrame, bias_type: str) -> pd.DataFrame:
        """��ƫ�����ͻ���ƫ��"""
        try:
            # ���ƫ��
            bias_scores = data['text'].apply(self._calculate_bias_score, args=(bias_type,))
            data[f'{bias_type}_bias_score'] = bias_scores
            
            # ���˸�ƫ������
            threshold = self.config.get('bias_threshold', 0.7)
            data = data[data[f'{bias_type}_bias_score'] < threshold]
            
            return data
        except Exception as e:
            self.logger.error(f"Bias mitigation for {bias_type} failed: {str(e)}")
            return data

    def _calculate_bias_score(self, text: str, bias_type: str) -> float:
        """����ƫ������"""
        # ģ�����
        return np.random.uniform(0.0, 1.0)
