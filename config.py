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



# =================================================================
# 🧠 СИСТЕМНЫЕ ПРОМПТЫ И ШАБЛОНЫ (PROMPTS)
# =================================================================

# 1. Шаблон для официальной библиотеки (Salesforce)
WRAPPER_TASK_TEMPLATE = "Strict limitation: the final prompt must not exceed {ch_lim} characters. Loss of meaning is unacceptable. Prompt to optimize: {prompt}"

# 2. Промпты для кастомного движка (MyPromptomatix)

# Извлечение сути задачи
EXTRACT_OBJECTIVE_PROMPT = "Extract the core objective from the prompt in 2 sentences."

# Генерация синтетических данных
DATA_GEN_PROMPT = (
    "You are a data generator. Create {num_samples} diverse testing examples for the following task. "
    "Output must be a JSON array of objects with keys 'input' and 'expected_output'."
)
JSON_ENFORCER = "\n\nCRITICAL: Return ONLY valid JSON array. No markdown blocks, no intro, no outro."

# LLM-Судья (Оценка промпта)
EVALUATOR_SYS_PROMPT = "Evaluate the Actual Output against the Expected Output. Return ONLY a single integer score from 0 to 10."
EVALUATOR_USER_TEMPLATE = "Expected: {expected}\nActual: {actual}"

# Мутатор: HYPE (Улучшение и расширение)
HYPE_PROMPT = (
    "You are an elite Prompt Engineer. Enhance the user's prompt using the 'HYPE' method: "
    "1. Assign an expert persona. "
    "2. Add step-by-step reasoning instructions (Chain of Thought). "
    "3. Structure with clear Markdown headers. "
    "Return ONLY the rewritten prompt."
)

# Мутатор: DISTILL (Сжатие)
DISTILL_PROMPT = (
    "You are a Prompt Compressor. Distill the prompt to its absolute core instructions. "
    "Remove all conversational filler, politeness, and unnecessary words. "
    "CRITICAL: Must be STRICTLY UNDER {ch_lim} characters. "
    "Return ONLY the compressed prompt."
)

# =================================================================
# ⚙️ НАСТРОЙКИ АВТОПРОМПТИНГА (ГЕНЕРАЦИЯ ДАННЫХ)
# =================================================================
# Количество синтетических примеров для тестирования (LLM-экзамена)
# Чем больше, тем точнее оценка, но дольше работает и дороже стоит.
# Оптимально для веба: 2-3.
NUM_TEST_SAMPLES = 3


# =================================================================
# 🤫 РЕЖИМ ТИШИНЫ (ДЛЯ ВЕБ-СЕРВИСА)
# =================================================================
# Если True: отключает ВСЕ логи, прогресс-бары и принты. 
# На выходе будет ТОЛЬКО текст итогового промпта и ничего больше.
SILENT_MODE = True