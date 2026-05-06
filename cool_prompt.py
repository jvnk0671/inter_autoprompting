import faulthandler
import os

from dotenv import load_dotenv

faulthandler.enable()
load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

try:
    from coolprompt.optimizer.hype import hype_optimizer
except Exception:
    hype_optimizer = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


def _fallback(prompt: str, ch_lim: int) -> str:
    text = " ".join(prompt.split())
    if ch_lim <= 0:
        return ""
    if len(text) <= ch_lim:
        return text
    return text[:ch_lim].rsplit(" ", 1)[0] or text[:ch_lim]


def coolprompt_optimize(prompt: str, model: str, ch_lim: int) -> str:
    if not KEY or hype_optimizer is None or ChatOpenAI is None:
        return _fallback(prompt, ch_lim)

    system_llm = ChatOpenAI(
        openai_api_key=KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model,
    )
    problem_description = (
        f"Strict limitation: the final prompt must not exceed {ch_lim} characters. "
        "Loss of meaning is unacceptable."
    )

    optimized = hype_optimizer(
        model=system_llm,
        prompt=prompt,
        problem_description=problem_description,
    )
    return str(optimized)


if __name__ == "__main__":
    test_prompt = (
        "You are a helpful mathematical assistant. Answer the question: investigate "
        "the convergence of the integral from 1 to +inf (sin(x))^2/x"
    )
    print(
        coolprompt_optimize(
            test_prompt, model="inclusionai/ling-2.6-1t:free", ch_lim=40
        )
    )
