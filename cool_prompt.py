import faulthandler
faulthandler.enable()
from typing import Any

from langchain_openai import ChatOpenAI
from coolprompt.assistant import PromptTuner

# Берем настройки из нашего центрального файла
import config

def coolprompt_optimize(prompt: str, model: str = config.TARGET_MODEL, ch_lim: int = config.DEFAULT_CHAR_LIMIT, method: str = 'hype',
                        system_model: str = config.SYSTEM_MODEL) -> dict[str, str | Any]:

    system_llm = ChatOpenAI(
        openai_api_key=config.OPENROUTER_API_KEY, 
        openai_api_base='https://openrouter.ai/api/v1',
        model_name=system_model
    )
    trg_llm = ChatOpenAI(
        openai_api_key=config.OPENROUTER_API_KEY, 
        openai_api_base='https://openrouter.ai/api/v1', 
        model_name=model
    )

    tuner = PromptTuner(target_model=trg_llm, system_model=system_llm)
    optimized = tuner.run(
        start_prompt=prompt, 
        method=method,
        feedback=False,
        verbose=2,
        train_as_test=True,
        problem_description=f'Strict limitation: the final prompt must not exceed {ch_lim} characters. Loss of meaning is unacceptable.',
        generate_num_samples=3
    )
    return {'optimized_prompt': optimized, 'init_metric': tuner.init_metric, 'final_metric': tuner.final_metric}

if __name__ == '__main__':
    testpr = 'I have a question for you that is directly related to mathematics. Please answer, what is 2 plus 2? Thank you'
    res = coolprompt_optimize(testpr, model=config.TARGET_MODEL, ch_lim=config.DEFAULT_CHAR_LIMIT)
    for i in res:
        print(f'{i}\t{res[i]}')