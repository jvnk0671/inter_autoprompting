import faulthandler
faulthandler.enable()
import os
from typing import Any

from dotenv import load_dotenv
from pipeline import OptimizationResult, PromptOptimizer

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

# системную модель лучше делать быстродумающей, т.к. существует лимит, который вызывает ошибку
# std_sys_model = 'meta-llama/llama-3.3-70b-instruct:free'
std_sys_model = 'inclusionai/ling-2.6-1t:free'
# с reasoning работает очень долго. наверное, можно сделать как опцию в будущем
reasoning_trg_model = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'


def _fallback(prompt: str, ch_lim: int) -> str:
    text = " ".join(prompt.split())
    if len(text) <= ch_lim:
        return text
    return text[:ch_lim].rsplit(" ", 1)[0] or text[:ch_lim]


# изначально планировалось использовать distill, но он делает большое количество запросов, что мешает завершить
# обработку даже для одного промпта (используем только бесплатные LLM для openrouter, и не хватает лимите)
def coolprompt_optimize(prompt: str, model: str = reasoning_trg_model, ch_lim: int = 100,
                        method: str = 'hype', system_model: str = std_sys_model) -> dict[str, str | Any]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return {
            'optimized_prompt': _fallback(prompt, ch_lim),
            'init_metric': None,
            'final_metric': 'fallback: OPENROUTER_API_KEY is not set'
        }

    try:
        from langchain_openai import ChatOpenAI
        from coolprompt.assistant import PromptTuner
    except Exception as exc:
        return {
            'optimized_prompt': _fallback(prompt, ch_lim),
            'init_metric': None,
            'final_metric': f'fallback: CoolPrompt dependencies are unavailable: {exc}'
        }

    system_llm = ChatOpenAI(openai_api_key=key, openai_api_base='https://openrouter.ai/api/v1', model_name=system_model)
    trg_llm = ChatOpenAI(openai_api_key=key, openai_api_base='https://openrouter.ai/api/v1', model_name=model)

    tuner = PromptTuner(target_model=trg_llm, system_model=system_llm)
    optimized = tuner.run(
        start_prompt=prompt,
        method=method,
        feedback=False,
        verbose=2,
        train_as_test=True,
        problem_description=f'Strict limitation: the final prompt must not exceed {ch_lim} characters. '
                            f'Loss of meaning is unacceptable.',

        # по умолчанию, их 10. можно попробовать реализовать параметр "эффективность", который будет влиять на аргумент
        generate_num_samples=3
    )
    return {
        'optimized_prompt': str(optimized),
        'init_metric': getattr(tuner, 'init_metric', None),
        'final_metric': getattr(tuner, 'final_metric', None)
    }


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
            init_metric=optimized.get('init_metric'),
            final_metric=optimized.get('final_metric')
        )


if __name__ == '__main__':
    testpr = 'I have a question for you that is directly related to mathematics. Please answer, what is 2 plus 2? Thank you'
    test_chars_lim = 40
    res = coolprompt_optimize(testpr, model=std_sys_model, ch_lim=test_chars_lim)
    for i in res:
        print(f'{i}\t{res[i]}')
