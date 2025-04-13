# src/data/augmentation/contextual_augmenter.py
import random
from typing import List, Dict
import nlpaug.augmenter.word as naw
from nlpaug import flow
from models.fairllm_core.attention_modifiers import SparseAttention
from models.third_party.mistral_quant import MistralQuantizer

class FairnessAwareAugmenter:
    """公平性感知数据增强管道，集成多种增强策略"""
    
    def __init__(self, config: Dict):
        # 初始化基础增强器
        self.aug_flow = flow.Sequential([
            naw.ContextualWordEmbsAug(
                model_path=config['aug_model'],
                action="substitute",
                device=config.get('device', 'cuda')
            ),
            naw.RandomWordAug(
                action="swap",
                aug_max=3
            ),
            naw.SynonymAug(aug_src='wordnet')
        ])
        
        # 偏见缓解模型
        self.bias_mitigator = MistralQuantizer(
            config['mitigator_model'],
            quant_config=config['quant_config']
        )
        
        # 增强策略配置
        self.strategy_weights = config['strategy_weights']
        self.max_retries = 3

    def augment(self, text: str, context: Dict) -> str:
        """执行多阶段增强并验证公平性"""
        for _ in range(self.max_retries):
            # 选择增强策略
            strategy = random.choices(
                list(self.strategy_weights.keys()),
                weights=list(self.strategy_weights.values())
            )[0]
            
            augmented = self._apply_strategy(strategy, text, context)
            
            # 验证增强结果
            if self._validate_augmentation(augmented, context):
                return augmented
                
        return text  # 回退到原始文本

    def _apply_strategy(self, strategy: str, text: str, context: Dict) -> str:
        """应用指定增强策略"""
        if strategy == 'paraphrase':
            return self.aug_flow.augment(text)[0]
        elif strategy == 'debias_rewrite':
            return self._debias_rewrite(text)
        elif strategy == 'neutralize':
            return self._neutralize_gender(text)
        else:
            return text

    def _debias_rewrite(self, text: str) -> str:
        """使用量化模型进行去偏见重写"""
        inputs = self.bias_mitigator.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.bias_mitigator.device)
        
        outputs = self.bias_mitigator.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        return self.bias_mitigator.tokenizer.decode(outputs[0], skip_special_tokens=True)

class AugmentationValidator:
    """增强验证器，确保增强后数据保持语义一致性与公平性"""
    
    def __init__(self, config: Dict):
        self.similarity_threshold = config['similarity_threshold']
        self.bias_detector = BiasDetectorPipeline(config['detector_config'])
        self.sentence_encoder = SentenceTransformer(config['sentence_model'])
        
    def validate(self, original: str, augmented: str) -> bool:
        """验证增强结果的合格性"""
        # 语义相似度检查
        orig_embed = self.sentence_encoder.encode(original)
        aug_embed = self.sentence_encoder.encode(augmented)
        similarity = np.dot(orig_embed, aug_embed) / (
            np.linalg.norm(orig_embed) * np.linalg.norm(aug_embed)
        )
        if similarity < self.similarity_threshold:
            return False
            
        # 偏见增量检查
        orig_bias = self.bias_detector.detect(original)['ml_scores']
        aug_bias = self.bias_detector.detect(augmented)['ml_scores']
        bias_diff = sum(abs(a - b) for a, b in zip(orig_bias.values(), aug_bias.values()))
        
        return bias_diff < 0.2

