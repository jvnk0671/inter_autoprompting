import os
from dotenv import load_dotenv

# Подгружаем скрытые ключи из .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==========================================
# 🎛️ ГЛАВНЫЕ НАСТРОЙКИ ПРОЕКТА
# ==========================================

# Выбор моделей
SYSTEM_MODEL = 'meta-llama/llama-3.3-70b-instruct'
TARGET_MODEL = 'meta-llama/llama-3.3-70b-instruct'

# Какой алгоритм оптимизации использовать?
# Варианты: 'coolprompt' или 'promptomatix'
ACTIVE_OPTIMIZER = 'promptomatix'

# Если выбран 'promptomatix', какой движок использовать внутри?
# True  -> Наш легкий кастомный гибридный тюнер (защита от лимитов)
# False -> Официальный тяжелый движок Salesforce (DSPy)
USE_CUSTOM_TUNER = False

# Настройки обрезки по умолчанию
DEFAULT_CHAR_LIMIT = 40
DEFAULT_UNCERTAINTY = 35