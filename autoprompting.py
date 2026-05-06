import logging
from transformers import AutoTokenizer
from typing import Any, Optional
from abc import ABC, abstractmethod

from pipeline import OptimizationResult
import promptomatix_wrapper
from cool_prompt import coolprompt_optimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

std_sys_model = "meta-llama/llama-3.3-70b-instruct:free"
std_sys_model2 = "inclusionai/ling-2.6-1t:free"
reasoning_trg_model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


@dataclass
class OptimizationResult:
    optimized_prompt: str
    init_tokens: Optional[int] = None
    final_tokens: Optional[int] = None


class PromptOptimizer(ABC):
    @abstractmethod
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        pass


class CoolPromptOptimizer(PromptOptimizer):
    def __init__(self, target_model_: str = std_sys_model2):
        self.target_model = target_model_

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        optimized = coolprompt_optimize(
            prompt=prompt, model=self.target_model, ch_lim=ch_lim
        )

        return OptimizationResult(optimized_prompt=optimized)


class ExampleOptimiser(PromptOptimizer):
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        return OptimizationResult(optimized_prompt=prompt[:ch_lim])


class PromptomatixOptimizer(PromptOptimizer):
    def __init__(self, model: str = std_sys_model2, use_custom=False):
        self.target_model = model
        self.system_model = model
        self.use_custom = use_custom

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        promptomatix_wrapper.USE_CUSTOM_TUNER = self.use_custom
        result = promptomatix_wrapper.promptomatix_optimize(
            prompt=prompt,
            model=self.target_model,
            system_model=self.system_model,
            ch_lim=ch_lim,
        )
        return OptimizationResult()


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

    markers_prior = [["\n"], [".", "!", "?"], [",", ";"], [" "]]

    for i in markers_prior:
        further_idx = max(cut.rfind(t) for t in i)
        if further_idx >= min_limit:
            return cut[: further_idx + 1].rstrip(" ")

    space = cut.rfind(" ")
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

        return res


if __name__ == "__main__":
    prompt_test = (
        "You are a helpful mathematical assistant. Answer the question: investigate the convergence of the "
        "integral from 1 to +inf (sin(x))^2/x"
    )
    ch_lim_test = 40
    unsertainty_test = 35
    TARGET_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

    coolprompt_opt = CoolPromptOptimizer(
        target_model=TARGET_MODEL,
    )

    pipeline = Pipeline(
        optimizer=coolprompt_opt, model=TARGET_MODEL.replace(":free", "")
    )

    try:
        res = pipeline.run(
            prompt=prompt_test, ch_limit=ch_lim_test, uncertainty=unsertainty_test
        )
        logger.info(f"Finally: {res.optimized_prompt}")
    except Exception as e:
        logger.error(f"ERROR: \n\n{e}\n")
