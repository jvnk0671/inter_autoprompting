import os
import time
import json
import re
import logging
from openai import OpenAI
from typing import Optional, Dict, Any, List
import config

logger = logging.getLogger(__name__)

class RobustLLMEngine:
    """Бронебойный клиент для OpenRouter с умным парсингом и retry-логикой."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/jvnk0671",
                "X-Title": "MyPromptomatix"
            }
        )

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Отправка запроса с защитой от 429 (Rate Limit) и 402 (Лимит баланса)."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str:
                    wait = (attempt + 1) * 10
                    logger.warning(f"[{self.model_name}] Лимит запросов. Ждем {wait} сек...")
                    time.sleep(wait)
                elif "402" in err_str or "404" in err_str:
                    logger.error(f"[{self.model_name}] Фатальная ошибка модели: {e}. Смените модель.")
                    raise e
                else:
                    logger.error(f"[{self.model_name}] Неизвестная ошибка: {e}")
                    time.sleep(5)
        return ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
       def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
        system_prompt += config.JSON_ENFORCER
        raw_text = self.generate(system_prompt, user_prompt, temperature=0.1)
        
        # Извлекаем JSON даже если модель обернула его в markdown ```json ...
        json_match = re.search(r'\[.*\]', raw_text.replace('\n', ' '), re.IGNORECASE | re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return None