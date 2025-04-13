# src/agents/debate_agent.py
import asyncio
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import CrossEncoder

class DebateCoordinator:
    """��ģ�ͱ���Э������ʵ�ֶ�̬��ʶ����㷨"""
    """����Э���������ڹ����ģ�ͱ���"""
    
    def __init__(self, model_manager, max_rounds=5):
        self.mm = model_manager
        self.consensus_model = CrossEncoder('cross-encoder/nli-deberta-v3-large')
        self.debate_rules = {
            'initiation_threshold': 0.3,
            'consensus_threshold': 0.85,
            'diversity_penalty': 0.2
        }
        self.model_weights = {
            'gpt-3.5-turbo': 0.9,
            'gemini1.0-pro-001': 0.85,
            'llama2-7B-instruct': 0.95,
            'mistral-7B-instruct-v0.2': 0.88,
            'qwen2-7B-instruct': 0.87
        }
        self.max_rounds = max_rounds

    async def conduct_debate(self, context: str, participant_models: List[str]) -> Dict:
        """���б���"""
        arguments = await self._gather_initial_arguments(context, participant_models)
        debate_log = [arguments]
        
        for round in range(self.max_rounds):
            if self._check_consensus(arguments):
                break
            
            rebuttals = await self._generate_rebuttals(context, arguments)
            arguments = self._aggregate_arguments(arguments, rebuttals)
            debate_log.append(arguments)
        
        final_response = self._form_consensus_response(arguments)
        return {
            'final_answer': final_response,
            'debate_log': debate_log,
            'consensus_metrics': self._calculate_consensus_metrics(debate_log)
        }

    async def _gather_initial_arguments(self, context: str, models: List[str]) -> Dict:
        """�ռ���ʼ�۵�"""
        tasks = [self._get_argument(context, model_id) for model_id in models]
        results = await asyncio.gather(*tasks)
        
        return {
            model_id: {
                'argument': result['argument'],
                'confidence': result['confidence'],
                'embeddings': result['embeddings']
            }
            for model_id, result in zip(models, results)
        }

    async def _get_argument(self, context: str, model_id: str) -> Dict:
        """��ȡ����ģ�͵��۵�"""
        response = await self.mm.models[model_id].chat.completions.create(
            messages=[{"role": "user", "content": context}]
        )
        argument = response.choices[0].message.content
        confidence = self._calculate_confidence(argument)
        embeddings = self._get_semantic_embeddings(argument)
        return {'argument': argument, 'confidence': confidence, 'embeddings': embeddings}

    def _check_consensus(self, arguments: Dict) -> bool:
        """����Ƿ��ɹ�ʶ"""
        similarity_matrix = self._build_similarity_matrix(arguments)
        weighted_scores = np.array([
            self.model_weights[model] * similarity_matrix[i] 
            for i, model in enumerate(arguments.keys())
        ])
        return np.mean(weighted_scores) > self.debate_rules['consensus_threshold']

    def _build_similarity_matrix(self, arguments: Dict) -> np.ndarray:
        """�������ƶȾ���"""
        embeddings = [args['embeddings'] for args in arguments.values()]
        return self.consensus_model.predict(embeddings)

    async def _generate_rebuttals(self, context: str, arguments: Dict) -> Dict:
        """���ɷ���"""
        tasks = []
        for model_id, args in arguments.items():
            task = self._generate_rebuttal(context, model_id, arguments)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def _generate_rebuttal(self, context: str, model_id: str, arguments: Dict) -> Dict:
        """���ɵ���ģ�͵ķ���"""
        other_arguments = {k: v for k, v in arguments.items() if k != model_id}
        rebuttal_prompt = f"��������ģ�͵��۵㣺{other_arguments}"
        response = await self.mm.models[model_id].chat.completions.create(
            messages=[{"role": "user", "content": rebuttal_prompt}]
        )
        rebuttal = response.choices[0].message.content
        return {model_id: rebuttal}

    def _aggregate_arguments(self, arguments: Dict, rebuttals: List[Dict]) -> Dict:
        """�ۺ��۵�ͷ���"""
        aggregated = {}
        for model_id, args in arguments.items():
            rebuttal = next((r.get(model_id) for r in rebuttals if model_id in r), "")
            aggregated[model_id] = {
                'argument': args['argument'],
                'rebuttal': rebuttal,
                'confidence': args['confidence'],
                'embeddings': args['embeddings']
            }
        return aggregated

    def _form_consensus_response(self, arguments: Dict) -> str:
        """�γɹ�ʶ�ش�"""
        weighted_responses = [
            (args['argument'], 
             self.model_weights[model] * args['confidence']) 
            for model, args in arguments.items()
        ]
        return max(weighted_responses, key=lambda x: x[1])[0]

    def _calculate_consensus_metrics(self, debate_log: List[Dict]) -> Dict:
        """���㹲ʶָ��"""
        metrics = {
            'consensus_rounds': len(debate_log),
            'average_similarity': np.mean([
                self._build_similarity_matrix(log) for log in debate_log
            ])
        }
        return metrics

    def _calculate_confidence(self, argument: str) -> float:
        """�����۵�����Ŷ�"""
        return len(argument.split()) / 100.0  # ��ʼʾ��ֵ�����ݲ�ͬ������Ƹ�����

    def _get_semantic_embeddings(self, text: str) -> np.ndarray:
        """��ȡ����Ƕ��"""
        return np.random.rand(768)  # ��ʼʾ��ֵ�����ݲ�ͬ������Ƹ�����
