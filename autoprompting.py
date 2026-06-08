import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import promptomatix_wrapper
from cool_prompt import coolprompt_optimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

std_sys_model = "meta-llama/llama-3.3-70b-instruct"
std_sys_model2 = "tencent/hy3-preview"
reasoning_trg_model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


@dataclass
class OptimizationResult:
    optimized_prompt: str
    init_tokens: Optional[int] = None
    final_tokens: Optional[int] = None
    init_score: Optional[float] = None 
    final_score: Optional[float] = None 


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


class EvoPromptOptimizer(PromptOptimizer):
    def __init__(
        self,
        target_model: str = reasoning_trg_model,
        system_model: str = std_sys_model,
        pop_size: int = 5,
        generations: int = 3,
        eval_samples: int = 3,
    ):
        self.target_model = target_model
        self.system_model = system_model
        self.pop_size = pop_size
        self.generations = generations
        self.eval_samples = eval_samples

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        try:
            from evo_prompt import EvoPromptEngine
            engine = EvoPromptEngine(
                target_model=self.target_model,
                system_model=self.system_model,
                pop_size=self.pop_size,
                generations=self.generations,
                eval_samples=self.eval_samples,
            )
            result = engine.run(prompt=prompt, ch_limit=ch_lim)
            
            return OptimizationResult(
                optimized_prompt=result.get("optimized_prompt", _fallback_cut(prompt, ch_lim)),
                init_score=result.get("init_metric"),
                final_score=result.get("final_metric")
            )
        except Exception as exc:
            logger.error(f"Ошибка в EvoPromptOptimizer: {exc}")
            return OptimizationResult(optimized_prompt=_fallback_cut(prompt, ch_lim))

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
        result = promptomatix_wrapper.promptomatix_optimize(
            prompt=prompt,
            model=self.target_model,
            system_model=self.system_model,
            ch_lim=ch_lim,
            use_custom_tuner=self.use_custom
        )
        return OptimizationResult(
            optimized_prompt=str(result.get("optimized_prompt", _fallback_cut(prompt, ch_lim))),
            init_score=result.get("init_metric"),  
            final_score=result.get("final_metric") 
        )


@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception:
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
    def __init__(self, optimizer: PromptOptimizer, sys_model: str, target_model: str):
        self.optimizer = optimizer
        self.sys_model = sys_model
        self.target_model = target_model

    def run(self, prompt: str, ch_limit: int, uncertainty: int, evaluate: bool = False, translate: bool = False) -> OptimizationResult:
        
        working_prompt = prompt
        original_lang = "English"
        s_eng = None
        
        if translate:
            from my_promptomatix.llm_engine import RobustLLMEngine
            s_eng = RobustLLMEngine(self.sys_model)
            lang_prompt = (
                "Identify the HUMAN language of the user's text. "
                "Return ONLY the human language name in English (e.g., Russian, Spanish, English). "
                "CRITICAL: Do NOT write 'Python', 'C++', or 'SQL'. If the text is in Russian but mentions Python, the language is Russian. "
                "Do not write any other words."
            )
            original_lang = s_eng.generate(lang_prompt, prompt, temperature=0.1).strip().capitalize()
            
            if "English" not in original_lang:
                trans_to_eng = (
                    "You are a strict translator. Translate the text to English. "
                    "CRITICAL: Output ONLY the plain translated text. DO NOT use JSON, DO NOT use markdown code blocks, DO NOT add explanations."
                )
                working_prompt = s_eng.generate(trans_to_eng, prompt, temperature=0.1)
        
        res = self.optimizer.optimize(working_prompt, ch_limit)
    
        if translate and "English" not in original_lang and s_eng:
            logger.info(f"Переводим улучшенный промпт обратно на {original_lang}...")
            trans_back = (
                f"You are a strict translator. Translate the text to {original_lang}. "
                "CRITICAL: Output ONLY the plain translated text. DO NOT use JSON, DO NOT use markdown code blocks, DO NOT add explanations."
            )
            res.optimized_prompt = s_eng.generate(trans_back, res.optimized_prompt, temperature=0.1)
        res.optimized_prompt = radical_cut(res.optimized_prompt, ch_limit, uncertainty)
        res.init_tokens = token_counter(prompt, self.sys_model)
        res.final_tokens = token_counter(res.optimized_prompt, self.sys_model)
        if evaluate:
            if s_eng is None:
                from my_promptomatix.llm_engine import RobustLLMEngine
                s_eng = RobustLLMEngine(self.sys_model)
            from my_promptomatix.synthetics import DataGenerator, Evaluator
            
            t_eng = RobustLLMEngine(self.target_model)
            
            task_desc = s_eng.generate("Extract the core objective from the prompt in 2 sentences.", prompt)
            test_data = DataGenerator(s_eng).generate_samples(task_desc, num_samples=2)
            
            evaluator = Evaluator(t_eng, s_eng)
            res.init_score = evaluator.score_prompt(prompt, test_data)
            res.final_score = evaluator.score_prompt(res.optimized_prompt, test_data)

        logger.info(
            "Optimized! Tokens: %s -> %s | Score: %s -> %s", 
            res.init_tokens, res.final_tokens, res.init_score, res.final_score
        )
        return res