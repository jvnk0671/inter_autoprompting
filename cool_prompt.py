import faulthandler

from coolprompt.optimizer.hype import hype_optimizer

faulthandler.enable()
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI



load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

std_sys_model = 'meta-llama/llama-3.3-70b-instruct:free'
std_sys_model2 = 'inclusionai/ling-2.6-1t:free'
reasoning_trg_model = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'


def coolprompt_optimize(prompt: str, model:str, ch_lim: int) -> str:

    system_llm = ChatOpenAI(openai_api_key=KEY, openai_api_base='https://openrouter.ai/api/v1',model_name=model)
    problem_description = (f'Strict limitation: the final prompt must not exceed {ch_lim} characters. '
                           f'Loss of meaning is unacceptable.')

    optimized_hype = hype_optimizer(
        model=system_llm,
        prompt=prompt,
        problem_description=problem_description
    )

    return optimized_hype


if __name__ == '__main__':
    testpr = ('You are a helpful mathematical assistant. Answer the question: investigate the convergence of the '
              'integral from 1 to +inf (sin(x))^2/x')
    test_chars_lim = 40
    res = coolprompt_optimize(testpr, model=std_sys_model2, ch_lim=test_chars_lim)
    for i in res:
        print(f'{i}\t{res[i]}')

