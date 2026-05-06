import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

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


class ExampleOptimiser(PromptOptimizer):
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        return OptimizationResult(optimized_prompt=_fallback_cut(prompt, ch_lim))


class CoolPromptOptimizer(PromptOptimizer):
    def __init__(
        self, target_model: str = std_sys_model2, system_model: Optional[str] = None
    ):
        self.target_model = target_model
        self.system_model = system_model or target_model

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        optimized = coolprompt_optimize(
            prompt=prompt,
            model=self.target_model,
            ch_lim=ch_lim,
        )
        return OptimizationResult(optimized_prompt=optimized)


class PromptomatixOptimizer(PromptOptimizer):
    def __init__(
        self,
        target_model: str = reasoning_trg_model,
        system_model: str = std_sys_model2,
        use_custom: bool = True,
    ):
        self.target_model = target_model
        self.system_model = system_model
        self.use_custom = use_custom

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        promptomatix_wrapper.USE_CUSTOM_TUNER = self.use_custom
        result = promptomatix_wrapper.promptomatix_optimize(
            prompt=prompt,
            model=self.target_model,
            system_model=self.system_model,
            ch_lim=ch_lim,
        )
        return OptimizationResult(
            optimized_prompt=str(
                result.get("optimized_prompt", _fallback_cut(prompt, ch_lim))
            )
        )


@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer

        logger.info("Getting tokenizer...")
        return AutoTokenizer.from_pretrained(model)
    except Exception as exc:
        logger.warning(
            "Tokenizer is unavailable (%s). Using rough token estimate.", exc
        )
        return None


def token_counter(prompt: str, model: str) -> int:
    tokenizer = get_tokenizer(model)
    if tokenizer is None:
        return max(1, len(prompt.split()))
    return len(tokenizer.encode(prompt))


def _fallback_cut(prompt: str, ch_limit: int) -> str:
    text = " ".join(prompt.split())
    if ch_limit <= 0:
        return ""
    if len(text) <= ch_limit:
        return text
    return text[:ch_limit].rsplit(" ", 1)[0] or text[:ch_limit]


def radical_cut(prompt: str, ch_limit: int, uncertainty: int) -> str:
    if ch_limit <= 0:
        return ""

    max_limit = ch_limit + max(0, uncertainty)
    if len(prompt) <= max_limit:
        return prompt.rstrip(" ")

    min_limit = max(0, ch_limit - max(0, uncertainty))
    cut = prompt[:max_limit]
    markers_prior = [["\n"], [".", "!", "?"], [",", ";"], [" "]]

    for markers in markers_prior:
        further_idx = max(cut.rfind(marker) for marker in markers)
        if further_idx >= min_limit:
            return cut[: further_idx + 1].rstrip(" ")

    space = cut.rfind(" ")
    if space >= min_limit:
        return cut[:space].rstrip(" ")

    return cut.rstrip(" ")


class Pipeline:
    def __init__(self, optimizer: PromptOptimizer, model: str):
        self.optimizer = optimizer
        self.model = model

    def run(self, prompt: str, ch_limit: int, uncertainty: int) -> OptimizationResult:
        logger.info("Prompt to optimize: %s", prompt)
        res = self.optimizer.optimize(prompt, ch_limit)
        res.optimized_prompt = radical_cut(res.optimized_prompt, ch_limit, uncertainty)
        res.init_tokens = token_counter(prompt, self.model)
        res.final_tokens = token_counter(res.optimized_prompt, self.model)

        logger.info(
            "Optimized successfully! %s -> %s", res.init_tokens, res.final_tokens
        )
        return res


if __name__ == "__main__":
    prompt_test = (
        "You are a helpful mathematical assistant. Answer the question: investigate "
        "the convergence of the integral from 1 to +inf (sin(x))^2/x"
    )
    pipeline = Pipeline(
        optimizer=ExampleOptimiser(),
        model=std_sys_model2.replace(":free", ""),
    )
    result = pipeline.run(prompt=prompt_test, ch_limit=40, uncertainty=35)
    logger.info("Finally: %s", result.optimized_prompt)
