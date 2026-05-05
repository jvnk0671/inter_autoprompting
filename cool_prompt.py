import faulthandler
faulthandler.enable()
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from coolprompt.assistant import PromptTuner

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

# системную модель лучше делать быстродумающей, т.к. существует лимит, который вызывает ошибку
# std_sys_model = 'meta-llama/llama-3.3-70b-instruct:free'
std_sys_model = 'inclusionai/ling-2.6-1t:free'
# с reasoning работает очень долго. наверное, можно сделать как опцию в будущем
reasoning_trg_model = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'


# изначально планировалось использовать distill, но он делает большое количество запросов, что мешает завершить
# обработку даже для одного промпта (используем только бесплатные LLM для openrouter, и не хватает лимите)
def coolprompt_optimize(prompt: str, model:str, ch_lim: int, method: str = 'hype',
                        system_model: str = std_sys_model) -> dict[str, str | Any]:

    system_llm = ChatOpenAI(openai_api_key=KEY, openai_api_base='https://openrouter.ai/api/v1',model_name=system_model)
    trg_llm = ChatOpenAI(openai_api_key=KEY, openai_api_base='https://openrouter.ai/api/v1', model_name=model)

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
    return {'optimized_prompt': optimized, 'init_metric': tuner.init_metric, 'final_metric': tuner.final_metric}


if __name__ == '__main__':
    testpr = 'I have a question for you that is directly related to mathematics. Please answer, what is 2 plus 2? Thank you'
    test_chars_lim = 40
    res = coolprompt_optimize(testpr, model=std_sys_model, ch_lim=test_chars_lim)
    for i in res:
        print(f'{i}\t{res[i]}')

