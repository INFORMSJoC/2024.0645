"""
数据处理工具类，用于加载和处理数据集。
"""

import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from copy import deepcopy
from itertools import chain

IGNORE_INDEX = -100  # 忽略索引，用于标记不计算损失的位置

def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    """
    根据数据集名称获取原始数据集。
    """
    # 这里可以根据需要加载不同的数据集
    # 为了简化，这里仅返回一个测试数据集
    return ExampleDataset()

def get_shuffle_idx(seed, size):
    """
    生成随机索引，用于数据集的随机打乱。
    """
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

class ExampleDataset(Dataset):
    """
    测试数据集类，用于测试数据。
    """
    def __init__(self):
        self.data = [torch.randn(10) for _ in range(100)]  # 测试数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class PromptDataset(Dataset):
    """
    提示数据集类，用于处理提示和响应数据。
    """
    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset, pad_token_id, train_phase):
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        if self.train_phase == 1:
            return len(self.chosen_dataset)
        else:
            return len(self.prompt_dataset)

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"],
            }
        else:
            return (
                self.prompt_dataset[idx]["input_ids"],
                self.prompt_dataset[idx]["attention_mask"],
                self.pad_token_id,
            )

def create_dataset_split(current_dataset, train_phase, tokenizer, max_seq_len):
    """
    创建数据集分割。
    """
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []

    if train_phase == 1:
        for data in current_dataset:
            # 这里可以根据需要处理数据
            chosen_dataset.append({"input_ids": torch.randn(10), "attention_mask": torch.randn(10), "labels": torch.randn(10)})
    else:
        for data in current_dataset:
            # 这里可以根据需要处理数据
            prompt_dataset.append({"input_ids": torch.randn(10), "attention_mask": torch.randn(10)})

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset, tokenizer.pad_token_id, train_phase)

def create_dataset(local_rank, dataset_name, data_split, output_path, train_phase, seed, tokenizer, max_seq_len):
    """
    创建数据集。
    """
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset
    train_dataset = create_dataset_split(train_dataset, train_phase, tokenizer, max_seq_len)
    eval_dataset = create_dataset_split(train_dataset, train_phase, tokenizer, max_seq_len)
    return train_dataset, eval_dataset

class DataCollator:
    """
    数据收集器，用于将多个样本合并为一个批次。
    """
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = pad_sequence([f[0] for f in data], padding_value=0, batch_first=True)
        batch["attention_mask"] = pad_sequence([f[1] for f in data], padding_value=0, batch_first=True)
        return batch

def get_unsupervised_data(args, tokenizer):
    """
    获取无监督数据。
    """
    unsupervised_raw_datasets = load_dataset(args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    return tokenized_datasets["train"]

class MiniDataset:
    """
    小数据集类，用于将大数据集分割为小批次。
    """
    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if isinstance(large_batch, (list, tuple)):
                large_size = len(large_batch[0])
            elif isinstance(large_batch, dict):
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if isinstance(large_batch, (list, tuple)):
                    small_dataset.append([x[i:i + self.small_batch_size] for x in large_batch])
                elif isinstance(large_batch, dict):
                    small_dataset.append({k: v[i:i + self.small_batch_size] for k, v in large_batch.items()})
                else:
                    small_dataset.append(large_batch[i:i + self.small_batch_size])
        self.free()
        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError("The dataset is full but we did not stop it. There is a bug in the code.")

    def free(self):
        self.dataset = []