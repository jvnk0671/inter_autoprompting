import logging
from typing import Dict, Any
from .llm_engine import RobustLLMEngine
from .mutators import PromptMutator

logger = logging.getLogger(__name__)

class FullPromptTuner:
    
    def __init__(self, target_model: str, system_model: str):
        self.sys_engine = RobustLLMEngine(system_model)
        self.mutator = PromptMutator(self.sys_engine)

    def run(self, start_prompt: str, ch_lim: int, method: str = 'hype', epochs: int = 1) -> Dict[str, Any]:
        logger.info(f"Запуск FullPromptTuner (Fast Mode). Метод: {method}, Лимит: {ch_lim}")

        current_prompt = start_prompt
        
        for epoch in range(epochs):
            if method == 'hype':
                current_prompt = self.mutator.hype(current_prompt, ch_lim)
            elif method == 'distill':
                current_prompt = self.mutator.distill(current_prompt, ch_lim)

        return {
            'optimized_prompt': current_prompt.strip(),
            'init_metric': None,
            'final_metric': None
        }