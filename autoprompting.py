import logging
from typing import Any, Optional
from transformers import AutoTokenizer

from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod
from cool_prompt import coolprompt_optimize
# from promptomatix import promptomatix_optimize
# from my_promptomatix.tuner import FullPromptTuner
from promptomatix_wrapper import promptomatix_optimize


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    optimized_prompt: str
    init_metric: Optional[Any] = None
    final_metric: Optional[Any] = None
    init_tokens: Optional[int] = None
    final_tokens: Optional[int] = None


class PromptOptimizer(ABC):
    @abstractmethod
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        pass




class CoolPromptOptimizer(PromptOptimizer):
    def __init__(self, target_model: str, system_model: str):
        self.target_model = target_model
        self.system_model = system_model

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        optimized = coolprompt_optimize(
            prompt=prompt,
            model=self.target_model,
            system_model=self.system_model,
            ch_lim=ch_lim
        )

        return OptimizationResult(
            optimized_prompt=optimized['optimized_prompt'],
            init_metric=optimized['init_metric'],
            final_metric=optimized['final_metric']
        )




class ExampleOptimiser(PromptOptimizer):
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        return OptimizationResult(optimized_prompt=prompt[:ch_lim])




# TODO: когда окончательно определимся с моделями, можно попробовать скачать файлы токенизаторов
@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    """Получить токенизатор по модели"""
    logger.info(f"Getting tokenizer...")
    return AutoTokenizer.from_pretrained(model)

def token_counter(prompt: str, model: str) -> int:
    """Используя токенизатор, посчитать токены по промпту"""
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(prompt))

def radical_cut(prompt: str, ch_limit: int, uncertainty: int) -> str:
    """Прямая обрезка промпта со слегка щадящей погрешностью и приоритетом обрезания символов"""
    max_limit = uncertainty + ch_limit
    if len(prompt) <= max_limit:
        return prompt

    min_limit = max(0, ch_limit - uncertainty)
    cut = prompt[:max_limit]

    markers_prior = [
        ['\n'],
        ['.', '!', '?'],
        [',', ';'],
        [' ']
    ]

    for i in markers_prior:
        further_idx = max(cut.rfind(t) for t in i)
        if further_idx >= min_limit:
            return cut[:further_idx + 1].rstrip(' ')

    space = cut.rfind(' ')
    if space != -1:
        return cut[:space]

    return cut



class Pipeline:
    def __init__(self, optimizer: PromptOptimizer, model: str):
        self.optimizer = optimizer
        self.model = model

    def run(self, prompt: str, ch_limit: int, uncertainty: int) -> OptimizationResult:
        logger.info(f"Prompt to optimize: {prompt}")
        res = self.optimizer.optimize(prompt, ch_limit)
        res.optimized_prompt = radical_cut(res.optimized_prompt, ch_limit, uncertainty)
        res.init_tokens = token_counter(prompt, self.model)
        res.final_tokens = token_counter(res.optimized_prompt, self.model)

        logger.info(f"Optimized successfully! {res.init_tokens} -> {res.final_tokens}")

        if res.init_metric is not None:
            logger.info(f"Additional metrics: {res.init_metric} -> {res.final_metric}")

        return res

# class CustomPromptomatixOptimizer(PromptOptimizer):
#     def __init__(self, target_model: str, system_model: str):
#         self.tuner = FullPromptTuner(
#             target_model=target_model, 
#             system_model=system_model
#         )

#     def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
#         # Мы используем method='hype' для расширения, а тюнер сам сделает distill под лимит
#         result = self.tuner.run(
#             start_prompt=prompt, 
#             ch_lim=ch_lim, 
#             method='hype', 
#             epochs=1 # Можно увеличить до 2-3 для более сильной мутации, но это дольше
#         )
        
#         return OptimizationResult(
#             optimized_prompt=result['optimized_prompt'],
#             init_metric=result['init_metric'],
#             final_metric=result['final_metric']
#         )


if __name__ == '__main__':
    prompt_test = 'I have a question for you that is directly related to mathematics. Please answer, what is 2 plus 2? Thank you'
    ch_lim_test = 40
    unsertainty_test = 35
    SYSTEM_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'
    TARGET_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'

    coolprompt_opt = CoolPromptOptimizer(
        target_model=TARGET_MODEL,
        system_model=SYSTEM_MODEL
    )

    promptomatix_opt = PromptomatixOptimizer(
           target_model=TARGET_MODEL,
           system_model=SYSTEM_MODEL
    )

    pipeline = Pipeline(
        # optimizer=coolprompt_opt,
        optimizer=promptomatix_opt,
        model=SYSTEM_MODEL.replace(':free', '')
    )

    try:
        res = pipeline.run(prompt=prompt_test, ch_limit=ch_lim_test, uncertainty=unsertainty_test)
        logger.info(f"Finally: {res.optimized_prompt}")
    except Exception as e:
        logger.error(f"ERROR: \n\n{e}\n")


# class PromptomatixOptimizer(PromptOptimizer):
#     def __init__(self, target_model: str, system_model: str):
#         self.target_model = target_model
#         self.system_model = system_model

#     def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
#         # Вызываем функцию из нашего нового файла
#         optimized = promptomatix_optimize(
#             prompt=prompt,
#             model=self.target_model,
#             system_model=self.system_model,
#             ch_lim=ch_lim
#         )

#         return OptimizationResult(
#             optimized_prompt=optimized['optimized_prompt'],
#             init_metric=optimized['init_metric'],
#             final_metric=optimized['final_metric']
#         )
    
