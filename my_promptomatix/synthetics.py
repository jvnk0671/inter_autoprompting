import logging
from typing import List, Dict, Any
from .llm_engine import RobustLLMEngine
import config

logger = logging.getLogger(__name__)

class DataGenerator:
    """Генерация синтетических данных для тестирования промпта."""
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def generate_samples(self, task_description: str, num_samples: int = 3) -> List[Dict[str, str]]:
        logger.info(f"Генерация {num_samples} тестовых примеров...")
        sys_prompt = config.DATA_GEN_PROMPT.format(num_samples=num_samples)
        samples = self.engine.generate_json(sys_prompt, f"Task: {task_description}")
        return samples if samples else []

class Evaluator:
    """LLM-as-a-judge: оценивает, насколько хорошо работает текущий промпт."""
    def __init__(self, target_engine: RobustLLMEngine, judge_engine: RobustLLMEngine):
        self.target = target_engine
        self.judge = judge_engine

    def score_prompt(self, prompt: str, test_data: List[Dict[str, str]]) -> float:
        if not test_data:
            return 0.0
        
        total_score = 0.0
        for data in test_data:
            # 1. Заставляем целевую модель ответить на вопрос, используя тестируемый промпт
            actual_output = self.target.generate(prompt, data['input'])
            
            # 2. Судья оценивает ответ от 0 до 10
            sys_judge = config.EVALUATOR_SYS_PROMPT
            user_judge = config.EVALUATOR_USER_TEMPLATE.format(
                expected=data['expected_output'], 
                actual=actual_output
            )
            
            score_str = self.judge.generate(sys_judge, user_judge, temperature=0.0)
            try:
                score = int(filter(str.isdigit, score_str))
                total_score += (score / 10.0)
            except:
                total_score += 0.5 # Средний балл при ошибке парсинга

        return total_score / len(test_data)