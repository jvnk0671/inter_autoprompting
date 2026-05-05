import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

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


# TODO: когда окончательно определимся с моделями, можно попробовать скачать файлы токенизаторов
@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    """Получить токенизатор по модели"""
    try:
        from transformers import AutoTokenizer

        logger.info("Getting tokenizer...")
        return AutoTokenizer.from_pretrained(model)
    except Exception as exc:
        logger.warning("Tokenizer is unavailable (%s). Using rough token estimate.", exc)
        return None


def token_counter(prompt: str, model: str) -> int:
    """Используя токенизатор, посчитать токены по промпту"""
    tokenizer = get_tokenizer(model)
    if tokenizer is None:
        return max(1, len(prompt.split()))
    return len(tokenizer.encode(prompt))


def radical_cut(prompt: str, ch_limit: int, uncertainty: int) -> str:
    """Прямая обрезка промпта со слегка щадящей погрешностью и приоритетом обрезания символов"""
    if ch_limit <= 0:
        return ""

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
    if space != -1 and space >= min_limit:
        return cut[:space].rstrip(' ')

    return cut.rstrip(' ')


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
