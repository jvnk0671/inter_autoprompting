import logging
import re
from typing import List, Dict, Any
from .llm_engine import RobustLLMEngine

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def generate_samples(self, task_description: str, num_samples: int = 3) -> List[Dict[str, str]]:
        sys_prompt = (
            f"You are a data generator. Create {num_samples} diverse testing examples for the following task. "
            f"Output must be a JSON array of objects with keys 'input' and 'expected_output'."
        )
        samples = self.engine.generate_json(sys_prompt, f"Task: {task_description}")
        return samples if samples else []

class Evaluator:
    def __init__(self, target_engine: RobustLLMEngine, judge_engine: RobustLLMEngine):
        self.target = target_engine
        self.judge = judge_engine

    def score_prompt(self, prompt: str, test_data: List[Dict[str, str]]) -> float:
        if not test_data:
            return 0.0
        
        total_score = 0.0
        for data in test_data:
            safe_input = str(data.get('input', 'Test input'))
            safe_expected = str(data.get('expected_output', ''))
            actual_output = self.target.generate(prompt, safe_input)
            sys_judge = "Evaluate the Actual Output against the Expected Output. Return ONLY a single integer score from 0 to 10."
            user_judge = f"Expected: {safe_expected}\nActual: {actual_output}"
            score_str = self.judge.generate(sys_judge, user_judge, temperature=0.0)
            try:
                import re
                match = re.search(r'\b([0-9]|10)\b', score_str)
                score = int(match.group(1)) if match else 5
                total_score += min(max(score, 0), 10) / 10.0
            except:
                total_score += 0.5

        return total_score / len(test_data)