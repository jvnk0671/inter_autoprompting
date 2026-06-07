# evo_prompt.py
import random
import asyncio
from typing import List, Tuple, Optional, Callable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")


class SyntheticDataGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str, num_samples: int = 5) -> Tuple[List[str], List[str]]:

        request = f"""
            Generate {num_samples} input-output pairs for this task:
            Task description: {prompt}
            Return JSON format:
            {{"examples": [{{"input": "...", "output": "..."}}]}}
        """

        result = self.model.invoke(request)

        import json
        try:
            data = json.loads(result.content if hasattr(result, 'content') else result)
            examples = data["examples"]
        except:
            examples = [
                {"input": f"Test {i}", "output": f"Expected {i}"}
                for i in range(num_samples)
            ]

        dataset = [ex["input"] for ex in examples]
        targets = [ex["output"] for ex in examples]

        return dataset, targets


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, prompt: str, dataset: List[str], targets: List[str]) -> float:
        answers = []
        for sample in dataset:
            full_prompt = f"{prompt}\n\nInput: {sample}"
            result = self.model.invoke(full_prompt)
            answers.append(result.content if hasattr(result, 'content') else result)

        scores = []
        for ans, tgt in zip(answers, targets):
            score_prompt = f"Rate similarity 0-1:\nAnswer: {ans}\nTarget: {tgt}\nScore:"
            score_result = self.model.invoke(score_prompt)
            try:
                scores.append(float(score_result.content if hasattr(score_result, 'content') else score_result))
            except:
                scores.append(0.5)

        return sum(scores) / len(scores)


class EvoPromptEngine:
    def __init__(self, target_model: str, system_model: str,
                 pop_size: int = 3, generations: int = 1, eval_samples: int = 1): # ЗАМЕНИТЬ НА 5-3-3 (это ради тестов)
        self.target_model = target_model
        self.system_model = system_model
        self.pop_size = pop_size
        self.generations = generations
        self.eval_samples = eval_samples

        self.llm = ChatOpenAI(
            openai_api_key=KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo",
        )

    async def _mutate(self, prompt: str) -> str:
        mutations = [
            f"Rephrase while keeping meaning: {prompt}",
            f"Simplify and clarify: {prompt}",
            f"Add helpful context: {prompt}",
            f"Restructure for clarity: {prompt}"
        ]
        result = self.llm.invoke(random.choice(mutations))
        return result.content if hasattr(result, 'content') else result

    async def _optimize(
            self,
            prompt: str,
            ch_limit: int,
            dataset: Optional[List[str]] = None,
            targets: Optional[List[str]] = None,
            metric_fn: Optional[Callable] = None
    ) -> dict:

        if dataset is None:
            generator = SyntheticDataGenerator(self.llm)
            dataset, targets = generator.generate(prompt, num_samples=self.eval_samples)

        if metric_fn is None:
            evaluator = Evaluator(self.llm)

            def default_metric(p: str) -> float:
                return evaluator.evaluate(p, dataset[:self.eval_samples], targets[:self.eval_samples])

            metric_fn = default_metric

        init_score = metric_fn(prompt)
        population = [prompt]
        for _ in range(self.pop_size - 1):
            population.append(await self._mutate(prompt))

        for _ in range(self.generations):
            scores = [(metric_fn(p), p) for p in population]
            scores.sort(reverse=True)
            population = [p for _, p in scores[:2]]

            while len(population) < self.pop_size:
                parent = random.choice(population)
                population.append(await self._mutate(parent))

        best = population[0]
        if len(best) > ch_limit:
            best = best[:ch_limit].rsplit(" ", 1)[0]

        return {
            "optimized_prompt": best,
            "init_metric": init_score,
            "final_metric": metric_fn(best)
        }

    def run(self, prompt: str, ch_limit: int, **kwargs) -> dict:
        return asyncio.run(self._optimize(
            prompt, ch_limit,
            dataset=kwargs.get('dataset'),
            targets=kwargs.get('targets'),
            metric_fn=kwargs.get('metric_fn')
        ))
