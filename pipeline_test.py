# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()
# KEY = os.getenv("OPENROUTER_API_KEY")

# test_content = 'give me the definition of LLM'
# url = 'https://openrouter.ai/api/v1/chat/completions'
# header = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
# json_data = {'model': 'openai/gpt-oss-120b:free', 'messages': [{'role': 'user', 'content': test_content}]}

# ans = requests.post(url=url, headers=header, json=json_data)
# ans.raise_for_status()
# print('OK')

# aj = ans.json()
# # for i in aj:
# #     print(f"{i} - \t{aj[i]}")
# print(f'\nRESPONSE:\n{aj["choices"][0]["message"]["content"]}')


import logging
from typing import Any, Optional
from transformers import AutoTokenizer

from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod

from cool_prompt import coolprompt_optimize
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

# РАСКОММЕНТИРОВАННЫЙ КЛАСС ПРОМПТОМАТИКСА
class PromptomatixOptimizer(PromptOptimizer):
    def __init__(self, target_model: str, system_model: str):
        self.target_model = target_model
        self.system_model = system_model

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        optimized = promptomatix_optimize(
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

@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    logger.info(f"Getting tokenizer...")
    return AutoTokenizer.from_pretrained(model)

def token_counter(prompt: str, model: str) -> int:
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(prompt))

def radical_cut(prompt: str, ch_limit: int, uncertainty: int) -> str:
    max_limit = uncertainty + ch_limit
    if len(prompt) <= max_limit:
        return prompt

    min_limit = max(0, ch_limit - uncertainty)
    cut = prompt[:max_limit]

    markers_prior = [['\n'], ['.', '!', '?'], [',', ';'], [' ']]

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


if __name__ == '__main__':
    prompt_test = 'I have a question for you that is directly related to mathematics. Please answer, what is 2 plus 2? Thank you'
    ch_lim_test = 40
    unsertainty_test = 35
    
    # Модели из твоего source: 7, чтобы не было ошибки лимитов
    SYSTEM_MODEL = 'inclusionai/ling-2.6-1t:free'
    TARGET_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'

    # coolprompt_opt = CoolPromptOptimizer(target_model=TARGET_MODEL, system_model=SYSTEM_MODEL)
    promptomatix_opt = PromptomatixOptimizer(target_model=TARGET_MODEL, system_model=SYSTEM_MODEL)

    pipeline = Pipeline(
        optimizer=promptomatix_opt,
        model='gpt2' # Заглушка для токенизатора, чтобы не качать огромную модель
    )

    try:
        res = pipeline.run(prompt=prompt_test, ch_limit=ch_lim_test, uncertainty=unsertainty_test)
        logger.info(f"Finally: {res.optimized_prompt}")
    except Exception as e:
        logger.error(f"ERROR: \n\n{e}\n")