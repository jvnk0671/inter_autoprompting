# import os
# import sys
# from typing import Any
# from dotenv import load_dotenv

# load_dotenv()
# KEY = os.getenv("OPENROUTER_API_KEY")

# # =================================================================
# # ULTIMATE INTERCEPTOR v7: 100% Рабочая резервная модель
# # =================================================================
# os.environ["OPENAI_API_KEY"] = KEY
# os.environ["OPENROUTER_API_KEY"] = KEY
# os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# # 1. Перехват LiteLLM (Только для внутренних нужд библиотеки)
# try:
#     import litellm
#     _orig_litellm_comp = litellm.completion
#     def _patched_litellm_comp(*args, **kwargs):
#         kwargs["max_tokens"] = 1500
#         if "api_base" in kwargs:
#             kwargs["api_base"] = "https://openrouter.ai/api/v1"
#         m = str(kwargs.get("model", ""))
#         # Меняем платные GPT на модель, которая 100% работает (проверено логами)
#         if "gpt" in m:
#             kwargs["model"] = "openrouter/inclusionai/ling-2.6-1t:free"
#         return _orig_litellm_comp(*args, **kwargs)
#     litellm.completion = _patched_litellm_comp
# except Exception:
#     pass

# # 2. Перехват чистого OpenAI SDK
# try:
#     import openai
#     if hasattr(openai, "OpenAI"):
#         _original_init = openai.OpenAI.__init__
#         def _patched_init(self, *args, **kwargs):
#             kwargs['base_url'] = "https://openrouter.ai/api/v1"
#             kwargs['api_key'] = KEY
#             _original_init(self, *args, **kwargs)
#         openai.OpenAI.__init__ = _patched_init
# except Exception:
#     pass
# # =================================================================


import os
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

# =================================================================
# ПЕРЕКЛЮЧАТЕЛЬ ДВИЖКА ОПТИМИЗАЦИИ
# True  = твой легкий и быстрый my_promptomatix (защита от 429)
# False = официальная библиотека promptomatix (возможны падения по лимитам)
# =================================================================
USE_CUSTOM_TUNER = True

# =================================================================
# БЛОК 1: НАСТРОЙКА И ПЕРЕХВАТЫ ДЛЯ ОФИЦИАЛЬНОЙ БИБЛИОТЕКИ
# =================================================================
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
                kwargs['base_url'] = "https://openrouter.ai/api/v1"
                kwargs['api_key'] = KEY
                _original_init(self, *args, **kwargs)

            openai.OpenAI.__init__ = _patched_init

        if hasattr(openai.resources.chat.completions.Completions, "create"):
            _orig_create = openai.resources.chat.completions.Completions.create

            def _patched_create(self, *args, **kwargs):
                kwargs["max_tokens"] = 1500
                m = kwargs.get("model", "")
                if "gpt" in m:
                    kwargs["model"] = "inclusionai/ling-2.6-1t:free"
                return _orig_create(self, *args, **kwargs)

            openai.resources.chat.completions.Completions.create = _patched_create
    except Exception:
        pass


# Добавляем пути к обеим библиотекам
ROOT_PATH = Path(__file__).resolve().parent
LIB_PATH = ROOT_PATH / 'promptomatix' / 'src'
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
def promptomatix_optimize(prompt: str, model: str, ch_lim: int,
                          system_model: str) -> dict[str, str | Any]:
    if not KEY:
        return {
            'optimized_prompt': _fallback(prompt, ch_lim),
            'init_metric': None,
            'final_metric': 'fallback: OPENROUTER_API_KEY is not set'
        }

    if USE_CUSTOM_TUNER:
        try:
            print("\n🚀 Используем бронебойный кастомный движок (my_promptomatix)...")
            from my_promptomatix.tuner import FullPromptTuner

            # Запуск твоего оркестратора[cite: 13]
            tuner = FullPromptTuner(target_model=model, system_model=system_model)
            result = tuner.run(
                start_prompt=prompt,
                ch_lim=ch_lim,
                method='hype',
                epochs=1
            )
            return {
                'optimized_prompt': result.get('optimized_prompt', _fallback(prompt, ch_lim)),
                'init_metric': result.get('init_metric'),
                'final_metric': result.get('final_metric')
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

            # 1. Самый дешевый бэкенд. Отключает сложные графы DSPy[cite: 7].
            "backend": "simple_meta_prompt",

            # 2. ЖЕСТКО задаем тип задачи.
            # Иначе она тратит целый запрос (около 500-1000 токенов) просто чтобы понять,
            # что "2+2" — это задача типа "generation"[cite: 7].
            "task_type": "generation",

            # 3. Минимум синтетических данных.
            # 1 пример — это физический минимум, чтобы библиотека не упала с ошибкой деления на ноль.
            "synthetic_data_size": 1,

            # 4. Соотношение данных.
            # Ставим 1.0 (или 0.99), чтобы все данные (наш 1 пример) ушли в train,
            # и она не тратила токены на дополнительный этап валидации (Valid: 0).
            "train_ratio": 0.99,

            # 5. Низкая температура.
            # Делает ответы модели сухими, короткими и без "воды"[cite: 7, 9].
            "temperature": 0.1,

            # 6. Ограничение ответа.
            # Не даем ей разглагольствовать в ответах.
            "max_tokens": 300,

            "api_base": "https://openrouter.ai/api/v1"
        }

        result = process_input(**config)
        optimized = result.get('result') if isinstance(result, dict) else None
        if not optimized:
            optimized = f"Optimization failed silently. Library returned: {result}"

        return {
            'optimized_prompt': optimized,
            'init_metric': 0.0,
            'final_metric': 0.0
        }
    except Exception as official_exc:
        return {
            'optimized_prompt': _fallback(prompt, ch_lim),
            'init_metric': None,
            'final_metric': f'fallback: custom promptomatix failed: {custom_error}; official promptomatix failed: {official_exc}'
        }
