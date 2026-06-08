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
            if "api_base" in kwargs:
                kwargs["api_base"] = "https://openrouter.ai/api/v1"
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
                response = _orig_create(self, *args, **kwargs)
                if hasattr(response, "choices"):
                    for choice in response.choices:
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            content = choice.message.content
                            if content is None:
                                choice.message.content = ""
                            elif isinstance(content, dict) or isinstance(content, list):
                                choice.message.content = json.dumps(content, ensure_ascii=False)
                            else:
                                choice.message.content = str(content)
                return response

            openai.resources.chat.completions.Completions.create = _patched_create
    except Exception:
        pass


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


def promptomatix_optimize(
    prompt: str, model: str, ch_lim: int, system_model: str, use_custom_tuner: bool = True
) -> dict[str, str | Any]:
    if not KEY:
        return {
            "optimized_prompt": _fallback(prompt, ch_lim),
            "init_metric": None,
            "final_metric": "fallback: OPENROUTER_API_KEY is not set",
        }

    if use_custom_tuner:
        try:
            from my_promptomatix.tuner import FullPromptTuner

            tuner = FullPromptTuner(target_model=model, system_model=system_model)
            result = tuner.run(
                start_prompt=prompt, ch_lim=ch_lim, method="hype", epochs=1
            )
            return {
                "optimized_prompt": result.get("optimized_prompt", _fallback(prompt, ch_lim)),
                "init_metric": result.get("init_metric"),
                "final_metric": result.get("final_metric"),
            }
        except Exception as custom_exc:
            print(f"Ошибка в кастомном тюнере: {custom_exc}")
            
    try:
        from promptomatix.main import process_input

        task_instruction = f"Strict limitation: the final prompt must not exceed {ch_lim} characters. Loss of meaning is unacceptable. Prompt to optimize: {prompt}"
        safe_model_name = _safe_model_name(system_model)
        
        dynamic_max_tokens = max(300, int(ch_lim))  
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
            "max_tokens": dynamic_max_tokens,
            "api_base": "https://openrouter.ai/api/v1",
            "teacher_model_name": "gpt-4o",
            "prompt_model_name": "gpt-4o",
            "eval_model_name": "gpt-4o",
            "meta_prompt_model": "gpt-4o"
        }

        result = process_input(**config)
        optimized = result.get("result") if isinstance(result, dict) else None
        
        if optimized:
            optimized = optimized.replace("<optimized_prompt>", "").replace("</optimized_prompt>", "").strip()
        else:
            optimized = f"Optimization failed silently. Library returned: {result}"

        return {"optimized_prompt": optimized, "init_metric": 0.0, "final_metric": 0.0}

    except Exception as official_exc:
        print(f"\nОШИБКА БИБЛИОТЕКИ PROMPTOMATIX:\n{official_exc}", flush=True)
        return {
            "optimized_prompt": _fallback(prompt, ch_lim),
            "init_metric": None,
            "final_metric": f"fallback: {official_exc}",
        }