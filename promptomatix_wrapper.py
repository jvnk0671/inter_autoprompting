import os
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import json

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")
USE_CUSTOM_TUNER = True


def _configure_openrouter() -> None:
    if not KEY:
        return

    os.environ["OPENAI_API_KEY"] = KEY
    os.environ["OPENROUTER_API_KEY"] = KEY
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    try:
        import litellm

        _orig_litellm_comp = litellm.completion

        def _patched_litellm_comp(*args, **kwargs):
            kwargs["max_tokens"] = 1500
            if "api_base" in kwargs:
                kwargs["api_base"] = "https://openrouter.ai/api/v1"
            m = str(kwargs.get("model", ""))
            if "gpt" in m:
                kwargs["model"] = "openrouter/inclusionai/ling-2.6-1t:free"
            return _orig_litellm_comp(*args, **kwargs)

        litellm.completion = _patched_litellm_comp
    except Exception:
        pass

    try:
        import openai

        if hasattr(openai, "OpenAI"):
            _original_init = openai.OpenAI.__init__

            def _patched_init(self, *args, **kwargs):
                kwargs["base_url"] = "https://openrouter.ai/api/v1"
                kwargs["api_key"] = KEY
                _original_init(self, *args, **kwargs)

            openai.OpenAI.__init__ = _patched_init

        if hasattr(openai.resources.chat.completions.Completions, "create"):
            _orig_create = openai.resources.chat.completions.Completions.create

            def _patched_create(self, *args, **kwargs):
                kwargs["model"] = "meta-llama/llama-3.3-70b-instruct"
                response = _orig_create(self, *args, **kwargs)
                if hasattr(response, "choices"):
                    for choice in response.choices:
                        if hasattr(choice, "message") and hasattr(
                            choice.message, "content"
                        ):
                            content = choice.message.content
                            if content is None:
                                choice.message.content = ""
                            elif isinstance(content, dict) or isinstance(content, list):
                                choice.message.content = json.dumps(
                                    content, ensure_ascii=False
                                )
                            else:
                                choice.message.content = str(content)
                return response

            openai.resources.chat.completions.Completions.create = _patched_create
    except Exception:
        pass


ROOT_PATH = Path(__file__).resolve().parent
LIB_PATH = ROOT_PATH / "promptomatix" / "src"
if str(LIB_PATH) not in sys.path:
    sys.path.append(str(LIB_PATH))
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

_configure_openrouter()


def _fallback(prompt: str, ch_lim: int) -> str:
    text = " ".join(prompt.split())
    if len(text) <= ch_lim:
        return text
    return text[:ch_lim].rsplit(" ", 1)[0] or text[:ch_lim]


def _safe_model_name(system_model: str) -> str:
    if system_model.startswith("openrouter/"):
        return system_model
    return f"openrouter/{system_model}"


# =================================================================
# БЛОК 2: ГЛАВНАЯ ФУНКЦИЯ-ОБЕРТКА
# =================================================================
def promptomatix_optimize(
    prompt: str, model: str, ch_lim: int, system_model: str
) -> dict[str, str | Any]:
    if not KEY:
        return {
            "optimized_prompt": _fallback(prompt, ch_lim),
            "init_metric": None,
            "final_metric": "fallback: OPENROUTER_API_KEY is not set",
        }

    if USE_CUSTOM_TUNER:
        try:
            print("\n🚀 Используем бронебойный кастомный движок (my_promptomatix)...")
            from my_promptomatix.tuner import FullPromptTuner

            # Запуск твоего оркестратора[cite: 13]
            tuner = FullPromptTuner(target_model=model, system_model=system_model)
            result = tuner.run(
                start_prompt=prompt, ch_lim=ch_lim, method="hype", epochs=1
            )
            return {
                "optimized_prompt": result.get(
                    "optimized_prompt", _fallback(prompt, ch_lim)
                ),
                "init_metric": result.get("init_metric"),
                "final_metric": result.get("final_metric"),
            }
        except Exception as custom_exc:
            custom_error = custom_exc
    else:
        custom_error = None

    try:
        print("\n🐌 Используем официальную библиотеку Salesforce...")
        from promptomatix.main import process_input

        task_instruction = f"Strict limitation: the final prompt must not exceed {ch_lim} characters. Loss of meaning is unacceptable. Prompt to optimize: {prompt}"
        safe_model_name = _safe_model_name(system_model)

        # config = {
        #     "raw_input": task_instruction,
        #     "model_name": safe_model_name,
        #     "model_api_key": KEY,
        #     "model_provider": "openai",
        #     "backend": "simple_meta_prompt",
        #     "synthetic_data_size": 1,
        #     "task_type": "generation",
        #     "max_tokens": 300,
        #     "api_base": "https://openrouter.ai/api/v1"
        # }

        config = {
            "raw_input": task_instruction,
            "model_name": safe_model_name,
            "model_api_key": KEY,
            "model_provider": "openai",
            "backend": "simple_meta_prompt",
            "task_type": "generation",
            "synthetic_data_size": 1,
            "train_ratio": 0.99,
            "temperature": 0.1,
            "max_tokens": 300,
            "api_base": "https://openrouter.ai/api/v1",
        }

        result = process_input(**config)
        optimized = result.get("result") if isinstance(result, dict) else None
        if not optimized:
            optimized = f"Optimization failed silently. Library returned: {result}"

        return {"optimized_prompt": optimized, "init_metric": 0.0, "final_metric": 0.0}
    except Exception as official_exc:
        return {
            "optimized_prompt": _fallback(prompt, ch_lim),
            "init_metric": None,
            "final_metric": f"fallback: custom promptomatix failed: {custom_error}; official promptomatix failed: {official_exc}",
        }
