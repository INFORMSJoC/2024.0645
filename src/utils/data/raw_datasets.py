from datasets import load_dataset
from torch.utils.data import Subset
import re


class CustomDataset:
    """
    自定义数据集类，用于加载和处理数据集。
    """
    def __init__(self, dataset_name, output_path, seed, local_rank):
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_datasets = self._load_dataset()

    def _load_dataset(self):
        """
        加载数据集。
        """
        return load_dataset(self.dataset_name)

    def get_train_data(self):
        """
        获取训练数据。
        """
        return self.raw_datasets["train"]

    def get_eval_data(self):
        """
        获取评估数据。
        """
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        """
        获取提示文本。
        """
        return sample.get("prompt", "") or sample.get("question", "")

    def get_chosen(self, sample):
        """
        获取选择的响应。
        """
        return sample.get("chosen", "") or sample.get("answer", "")

    def get_rejected(self, sample):
        """
        获取拒绝的响应。
        """
        return sample.get("rejected", "") or sample.get("rejected_answer", "")

    def get_prompt_and_chosen(self, sample):
        """
        获取提示和选择的响应。
        """
        return f"{self.get_prompt(sample)} {self.get_chosen(sample)}"

    def get_prompt_and_rejected(self, sample):
        """
        获取提示和拒绝的响应。
        """
        return f"{self.get_prompt(sample)} {self.get_rejected(sample)}"


class LocalJsonDataset(CustomDataset):
    """
    本地JSON文件数据集类。
    """
    def __init__(self, dataset_name, output_path, seed, local_rank, train_path, eval_path):
        super().__init__(dataset_name, output_path, seed, local_rank)
        self.train_path = train_path
        self.eval_path = eval_path
        self.raw_datasets = load_dataset("json", data_files={"train": train_path, "test": eval_path})

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]


class SplitDataset(CustomDataset):
    """
    分割数据集类，用于将数据集分割为训练集和评估集。
    """
    def __init__(self, dataset_name, output_path, seed, local_rank):
        super().__init__(dataset_name, output_path, seed, local_rank)

    def get_train_data(self):
        from data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name,
            self.seed,
            "train_eval",
            "9,1",
            0,
            len(dataset)
        )
        return Subset(dataset, index)

    def get_eval_data(self):
        from data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(
            self.local_rank,
            self.output_path,
            self.dataset_name,
            self.seed,
            "train_eval",
            "9,1",
            1,
            len(dataset)
        )
        return Subset(dataset, index)


class MultiLanguageDataset(CustomDataset):
    """
    语言数据集类，用于处理数据。
    """
    def __init__(self, dataset_name, output_path, seed, local_rank, lang_key):
        super().__init__(dataset_name, output_path, seed, local_rank)
        self.lang_key = lang_key

    def get_prompt(self, sample):
        return sample.get(f"queries_{self.lang_key}", "")

    def get_chosen(self, sample):
        answers = sample.get(f"answers_{self.lang_key}", [])
        return answers[0].get("text", "") if answers else ""


if __name__ == "__main__":
    # 创建一个本地JSON数据集
    local_dataset = LocalJsonDataset(
        dataset_name="local/jsonfile",
        output_path="./output",
        seed=42,
        local_rank=0,
        train_path="./data/train.json",
        eval_path="./data/eval.json"
    )
    train_data = local_dataset.get_train_data()
    eval_data = local_dataset.get_eval_data()

    multi_lang_dataset = MultiLanguageDataset(
        dataset_name="mkqa",
        output_path="./output",
        seed=42,
        local_rank=0,
        lang_key="zh_cn"
    )
    prompt = multi_lang_dataset.get_prompt(train_data[0])
    chosen = multi_lang_dataset.get_chosen(train_data[0])
    print(f"Prompt: {prompt}")
    print(f"Chosen: {chosen}")
    