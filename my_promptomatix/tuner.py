import logging
from typing import Dict, Any
from .llm_engine import RobustLLMEngine
from .synthetics import DataGenerator, Evaluator
from .mutators import PromptMutator
import config

logger = logging.getLogger(__name__)

class FullPromptTuner:
    """Главный оркестратор, заменяющий оригинальную библиотеку."""
    
    def __init__(self, target_model: str, system_model: str):
        self.sys_engine = RobustLLMEngine(system_model)
        self.target_engine = RobustLLMEngine(target_model)
        
        self.data_gen = DataGenerator(self.sys_engine)
        self.evaluator = Evaluator(target_engine=self.target_engine, judge_engine=self.sys_engine)
        self.mutator = PromptMutator(self.sys_engine)

    def run(self, start_prompt: str, ch_lim: int, method: str = 'hype', epochs: int = 1) -> Dict[str, Any]:
        # 1. Анализ задачи и генерация данных
        # 1. Анализ задачи и генерация данных
        logger.info("Извлечение сути задачи...")
        task_desc = self.sys_engine.generate(config.EXTRACT_OBJECTIVE_PROMPT, start_prompt)
        test_data = self.data_gen.generate_samples(task_desc, num_samples=config.NUM_TEST_SAMPLES)

        current_prompt = start_prompt
        init_metric = self.evaluator.score_prompt(current_prompt, test_data)
        logger.info(f"Начальная оценка промпта: {init_metric}")

        # 2. Итеративный процесс улучшения (Epochs)
        for epoch in range(epochs):
            logger.info(f"Эпоха {epoch+1}/{epochs}...")
            
            if method == 'hype':
                current_prompt = self.mutator.hype(current_prompt)
            elif method == 'distill':
                current_prompt = self.mutator.distill(current_prompt, ch_lim)
            
            # Для метода hype нужно финальное сжатие (distill), если он превысил лимит
            if len(current_prompt) > ch_lim:
                logger.info("Промпт превысил лимит, применяем Distill...")
                current_prompt = self.mutator.distill(current_prompt, ch_lim)

        # 3. Финальная оценка
        final_metric = self.evaluator.score_prompt(current_prompt, test_data)
        logger.info(f"Финальная оценка промпта: {final_metric}")

        return {
            'optimized_prompt': current_prompt.strip(),
            'init_metric': init_metric,
            'final_metric': final_metric
        }