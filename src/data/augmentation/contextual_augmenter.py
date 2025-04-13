# src/data/augmentation/contextual_augmenter.py
import random
from typing import List, Dict
import nlpaug.augmenter.word as naw
from nlpaug import flow
from models.fairllm_core.attention_modifiers import SparseAttention
from models.third_party.mistral_quant import MistralQuantizer

class FairnessAwareAugmenter:
    """��ƽ�Ը�֪������ǿ�ܵ������ɶ�����ǿ����"""
    
    def __init__(self, config: Dict):
        # ��ʼ��������ǿ��
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
        
        # ƫ������ģ��
        self.bias_mitigator = MistralQuantizer(
            config['mitigator_model'],
            quant_config=config['quant_config']
        )
        
        # ��ǿ��������
        self.strategy_weights = config['strategy_weights']
        self.max_retries = 3

    def augment(self, text: str, context: Dict) -> str:
        """ִ�ж�׶���ǿ����֤��ƽ��"""
        for _ in range(self.max_retries):
            # ѡ����ǿ����
            strategy = random.choices(
                list(self.strategy_weights.keys()),
                weights=list(self.strategy_weights.values())
            )[0]
            
            augmented = self._apply_strategy(strategy, text, context)
            
            # ��֤��ǿ���
            if self._validate_augmentation(augmented, context):
                return augmented
                
        return text  # ���˵�ԭʼ�ı�

    def _apply_strategy(self, strategy: str, text: str, context: Dict) -> str:
        """Ӧ��ָ����ǿ����"""
        if strategy == 'paraphrase':
            return self.aug_flow.augment(text)[0]
        elif strategy == 'debias_rewrite':
            return self._debias_rewrite(text)
        elif strategy == 'neutralize':
            return self._neutralize_gender(text)
        else:
            return text

    def _debias_rewrite(self, text: str) -> str:
        """ʹ������ģ�ͽ���ȥƫ����д"""
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
    """��ǿ��֤����ȷ����ǿ�����ݱ�������һ�����빫ƽ��"""
    
    def __init__(self, config: Dict):
        self.similarity_threshold = config['similarity_threshold']
        self.bias_detector = BiasDetectorPipeline(config['detector_config'])
        self.sentence_encoder = SentenceTransformer(config['sentence_model'])
        
    def validate(self, original: str, augmented: str) -> bool:
        """��֤��ǿ����ĺϸ���"""
        # �������ƶȼ��
        orig_embed = self.sentence_encoder.encode(original)
        aug_embed = self.sentence_encoder.encode(augmented)
        similarity = np.dot(orig_embed, aug_embed) / (
            np.linalg.norm(orig_embed) * np.linalg.norm(aug_embed)
        )
        if similarity < self.similarity_threshold:
            return False
            
        # ƫ���������
        orig_bias = self.bias_detector.detect(original)['ml_scores']
        aug_bias = self.bias_detector.detect(augmented)['ml_scores']
        bias_diff = sum(abs(a - b) for a, b in zip(orig_bias.values(), aug_bias.values()))
        
        return bias_diff < 0.2

