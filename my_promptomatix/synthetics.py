import logging
from typing import List, Dict, Any
from .llm_engine import RobustLLMEngine

logger = logging.getLogger(__name__)

class DataGenerator:
    """Генерация синтетических данных для тестирования промпта."""
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def generate_samples(self, task_description: str, num_samples: int = 3) -> List[Dict[str, str]]:
        logger.info(f"Генерация {num_samples} тестовых примеров...")
        sys_prompt = (
            f"You are a data generator. Create {num_samples} diverse testing examples for the following task. "
            f"Output must be a JSON array of objects with keys 'input' and 'expected_output'."
        )
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
            sys_judge = "Evaluate the Actual Output against the Expected Output. Return ONLY a single integer score from 0 to 10."
            user_judge = f"Expected: {data['expected_output']}\nActual: {actual_output}"
            
            score_str = self.judge.generate(sys_judge, user_judge, temperature=0.0)
            try:
                score = int(filter(str.isdigit, score_str))
                total_score += (score / 10.0)
            except:
                total_score += 0.5 # Средний балл при ошибке парсинга

        return total_score / len(test_data)