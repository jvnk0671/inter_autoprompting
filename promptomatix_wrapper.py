import os
import sys
import json
from typing import Any

import config

# =================================================================
# БЛОК 1: ЖЕСТКАЯ ХИРУРГИЯ БИБЛИОТЕК (MONKEY PATCHING)
# =================================================================
os.environ["OPENAI_API_KEY"] = config.OPENROUTER_API_KEY or ""
os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY or ""
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 💉 ПАТЧИМ КЛИЕНТ OPENAI
try:
    import openai
    if hasattr(openai, "OpenAI"):
        _original_init = openai.OpenAI.__init__
        def _patched_init(self, *args, **kwargs):
            kwargs['base_url'] = "https://openrouter.ai/api/v1"
            kwargs['api_key'] = config.OPENROUTER_API_KEY
            _original_init(self, *args, **kwargs)
        openai.OpenAI.__init__ = _patched_init
        
    if hasattr(openai.resources.chat.completions.Completions, "create"):
        _orig_create = openai.resources.chat.completions.Completions.create
        def _patched_create(self, *args, **kwargs):
            # Жестко пробрасываем модель из конфига
            kwargs["model"] = config.SYSTEM_MODEL
            
            # Делаем реальный запрос к API
            response = _orig_create(self, *args, **kwargs)
            
            # 🛡 ИСПРАВЛЯЕМ БАГИ БИБЛИОТЕКИ ПРЯМО НА ЛЕТУ
            # Насильно превращаем любой ответ в строку, чтобы DSPy не подавился
            if hasattr(response, 'choices'):
                for choice in response.choices:
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                        if content is None:
                            # Спасает от 'NoneType' object has no attribute 'split'
                            choice.message.content = "" 
                        elif isinstance(content, dict) or isinstance(content, list):
                            # Спасает от 'dict' object has no attribute 'strip'
                            choice.message.content = json.dumps(content, ensure_ascii=False)
                        else:
                            choice.message.content = str(content)
            return response
        openai.resources.chat.completions.Completions.create = _patched_create
except Exception as e:
    print(f"Ошибка применения патча OpenAI: {e}")

# 💉 ПАТЧИМ КЛИЕНТ LITELLM (на всякий случай, если библиотека пойдет этим путем)
try:
    import litellm
    _orig_litellm_comp = litellm.completion
    def _patched_litellm_comp(*args, **kwargs):
        if "api_base" in kwargs:
            kwargs["api_base"] = "https://openrouter.ai/api/v1"
        
        sys_mod = config.SYSTEM_MODEL
        if not sys_mod.startswith("openrouter/"):
            sys_mod = f"openrouter/{sys_mod}"
        kwargs["model"] = sys_mod
        
        response = _orig_litellm_comp(*args, **kwargs)
        
        # Тот же самый бронежилет для ответов
        if hasattr(response, 'choices'):
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    if content is None:
                        choice.message.content = ""
                    elif isinstance(content, dict) or isinstance(content, list):
                        choice.message.content = json.dumps(content, ensure_ascii=False)
                    else:
                        choice.message.content = str(content)
        return response
    litellm.completion = _patched_litellm_comp
except Exception:
    pass



# =================================================================
# БЛОК 2: ГЛАВНАЯ ФУНКЦИЯ-ОБЕРТКА
# =================================================================
def promptomatix_optimize(prompt: str, model: str, ch_lim: int, system_model: str) -> dict[str, str | Any]:
    
    if config.USE_CUSTOM_TUNER:
        print("\n🚀 Используем бронебойный кастомный движок (my_promptomatix)...")
        from my_promptomatix.tuner import FullPromptTuner
        
        tuner = FullPromptTuner(target_model=model, system_model=system_model)
        result = tuner.run(
            start_prompt=prompt, 
            ch_lim=ch_lim, 
            method='hype', 
            epochs=1
        )
        return {
            'optimized_prompt': result['optimized_prompt'],
            'init_metric': result.get('init_metric', 0.0),
            'final_metric': result.get('final_metric', 0.0)
        }
        
    else:
        print("\n🐌 Используем официальную библиотеку Salesforce...")
        from promptomatix.main import process_input
        
        task_instruction = f"Strict limitation: the final prompt must not exceed {ch_lim} characters. Loss of meaning is unacceptable. Prompt to optimize: {prompt}"
        
        safe_model_name = system_model
        if not safe_model_name.startswith("openrouter/"):
            safe_model_name = f"openrouter/{safe_model_name}"
        
        setup_config = {
            "raw_input": task_instruction,
            "model_name": safe_model_name, 
            "model_api_key": config.OPENROUTER_API_KEY,
            "model_provider": "openai", 
            "backend": "simple_meta_prompt",
            "task_type": "generation", 
            "synthetic_data_size": 1, 
            "train_ratio": 0.99,
            "temperature": 0.1,
            "max_tokens": 800, 
            "api_base": "https://openrouter.ai/api/v1"
        }
        
        try:
            result = process_input(**setup_config)
            if result and 'result' in result:
                optimized = result['result']
            else:
                optimized = f"Optimization failed silently. Library returned: {result}"
        except Exception as e:
            optimized = f"Error during optimization: {e}"
        
        return {
            'optimized_prompt': optimized,
            'init_metric': 0.0,
            'final_metric': 0.0
        }