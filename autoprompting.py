import logging
from typing import Any, Optional
from transformers import AutoTokenizer
from functools import lru_cache
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Импорты наших модулей и конфига
from cool_prompt import coolprompt_optimize
import promptomatix_wrapper
import config

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

class PromptomatixOptimizer(PromptOptimizer):
    def __init__(self, target_model: str, system_model: str, use_custom: bool = True):
        self.target_model = target_model
        self.system_model = system_model
        self.use_custom = use_custom

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        promptomatix_wrapper.USE_CUSTOM_TUNER = self.use_custom
        result = promptomatix_wrapper.promptomatix_optimize(
            prompt=prompt, 
            model=self.target_model, 
            system_model=self.system_model,
            ch_lim=ch_lim
        )
        return OptimizationResult(
            optimized_prompt=result['optimized_prompt'],
            init_metric=result.get('init_metric', 0.0),
            final_metric=result.get('final_metric', 0.0)
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
    
    # 🧠 АВТОМАТИЧЕСКИЙ ВЫБОР ИЗ CONFIG.PY
    if config.ACTIVE_OPTIMIZER == 'coolprompt':
        logger.info("Выбран оптимизатор: CoolPrompt")
        active_optimizer = CoolPromptOptimizer(
            target_model=config.TARGET_MODEL,
            system_model=config.SYSTEM_MODEL
        )
    else:
        logger.info(f"Выбран оптимизатор: Promptomatix (Custom={config.USE_CUSTOM_TUNER})")
        active_optimizer = PromptomatixOptimizer(
           target_model=config.TARGET_MODEL,
           system_model=config.SYSTEM_MODEL,
           use_custom=config.USE_CUSTOM_TUNER
        )

    pipeline = Pipeline(
        optimizer=active_optimizer,
        model='gpt2' 
    )

    try:
        res = pipeline.run(
            prompt=prompt_test, 
            ch_limit=config.DEFAULT_CHAR_LIMIT, 
            uncertainty=config.DEFAULT_UNCERTAINTY
        )
        logger.info(f"Finally: {res.optimized_prompt}")
    except Exception as e:
        logger.error(f"ERROR: \n\n{e}\n")