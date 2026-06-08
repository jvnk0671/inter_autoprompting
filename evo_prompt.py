import logging
import random
from my_promptomatix.llm_engine import RobustLLMEngine

logger = logging.getLogger(__name__)

class EvoPromptEngine:
    
    def __init__(self, target_model: str, system_model: str,
                 pop_size: int = 5, generations: int = 3, eval_samples: int = 3):
        self.target_model = target_model
        self.system_model = system_model
        self.pop_size = pop_size
        self.generations = generations
        self.eval_samples = eval_samples

        self.sys_engine = RobustLLMEngine(system_model)
        self.target_engine = RobustLLMEngine(target_model)

    def _mutate(self, prompt: str) -> str:
        mutations = [
            f"Rephrase the following prompt while keeping its exact meaning: {prompt}",
            f"Simplify and clarify the instructions in this prompt: {prompt}",
            f"Add helpful context and structure to this prompt: {prompt}",
            f"Restructure this prompt using Markdown headers and bullet points for better clarity: {prompt}"
        ]
        
        sys_prompt = (
            "You are an AI Prompt Engineer. Apply the requested transformation to the prompt. "
            "CRITICAL: Return ONLY the new prompt text. No markdown blocks, no intro, no outro."
        )
        
        result = self.sys_engine.generate(sys_prompt, random.choice(mutations))
        return result.strip()

    def run(self, prompt: str, ch_limit: int) -> dict:
        from my_promptomatix.synthetics import DataGenerator, Evaluator

        logger.info(f" Запуск EvoPrompt (Population: {self.pop_size}, Generations: {self.generations})")
        
        task_desc = self.sys_engine.generate("Extract the core objective from the prompt in 1 sentence.", prompt)
        generator = DataGenerator(self.sys_engine)
        test_data = generator.generate_samples(task_desc, num_samples=self.eval_samples)

        evaluator = Evaluator(self.target_engine, self.sys_engine)

        def metric_fn(p: str) -> float:
            return evaluator.score_prompt(p, test_data)

        init_score = metric_fn(prompt)
        population = [prompt]
        
        for _ in range(self.pop_size - 1):
            population.append(self._mutate(prompt))

        for gen in range(self.generations):
            logger.info(f"🔄 Эпоха {gen + 1}/{self.generations}")
            
            scores = [(metric_fn(p), p) for p in population]
            scores.sort(reverse=True, key=lambda x: x[0]) 
            
            population = [p for _, p in scores[:2]]

            while len(population) < self.pop_size:
                parent = random.choice(population[:2])
                population.append(self._mutate(parent))

        final_scores = [(metric_fn(p), p) for p in population]
        final_scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_prompt = final_scores[0]

        if len(best_prompt) > ch_limit:
            best_prompt = best_prompt[:ch_limit].rsplit(" ", 1)[0]

        logger.info(f" EvoPrompt Оценка: {init_score:.2f} -> {best_score:.2f}")
        
        return {
            "optimized_prompt": best_prompt,
            "init_metric": init_score,
            "final_metric": best_score
        }