import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")

test_content = 'give me the definition of LLM'
url = 'https://openrouter.ai/api/v1/chat/completions'
header = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
json_data = {'model': 'openai/gpt-oss-120b:free', 'messages': [{'role': 'user', 'content': test_content}]}

ans = requests.post(url=url, headers=header, json=json_data)
ans.raise_for_status()
print('OK')

aj = ans.json()
# for i in aj:
#     print(f"{i} - \t{aj[i]}")
print(f'\nRESPONSE:\n{aj["choices"][0]["message"]["content"]}')


