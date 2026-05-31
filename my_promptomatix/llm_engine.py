import os
import time
import json
import re
import logging
from openai import OpenAI
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class RobustLLMEngine:
    
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
        max_retries = 3
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
                    wait = 2 ** attempt
                    logger.warning(f"[{self.model_name}] лимит запросов жди {wait} сек...")
                    time.sleep(wait)
                elif "402" in err_str or "404" in err_str:
                    logger.error(f"[{self.model_name}] ошибка модели: {e}")
                    raise e
                else:
                    logger.error(f"[{self.model_name}] ошибка: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)
        return ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
        system_prompt += "\n\nCRITICAL: Return ONLY valid JSON array. No markdown blocks, no intro, no outro."
        raw_text = self.generate(system_prompt, user_prompt, temperature=0.1)
        
        json_match_array = re.search(r'\[.*\]', raw_text.replace('\n', ' '), re.IGNORECASE | re.DOTALL)
        if json_match_array:
            try:
                return json.loads(json_match_array.group(0))
            except json.JSONDecodeError:
                pass
            
        json_match_obj = re.search(r'\{.*\}', raw_text.replace('\n', ' '), re.IGNORECASE | re.DOTALL)
        if json_match_obj:
            try:
                obj = json.loads(json_match_obj.group(0))
                return [obj]
            except json.JSONDecodeError:
                pass
                
        return None