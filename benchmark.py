import os
import sys
import warnings
import logging

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DSPY_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)

import time
from tabulate import tabulate
from main import OptimizeRequest, optimize
from my_promptomatix.llm_engine import RobustLLMEngine
from my_promptomatix.synthetics import DataGenerator, Evaluator

def run_benchmark():
    
    test_prompts = [
        "Объясни квантовую запутанность для 5-летнего ребенка, используя аналогии с машинками.",
        "Напиши скрипт на питоне который подключается к базе postgresql, берет всех юзеров со статусом active и сохраняет в csv."
    ]
    
    test_configs = [
        {"method": "my_promptomatix", "translate": False, "name": "MY_PROMPTOMATIX (Без перевода)"},
        {"method": "my_promptomatix", "translate": True,  "name": "MY_PROMPTOMATIX (+ АНГЛ. БУСТ 🇬)"},
        {"method": "coolprompt",      "translate": False, "name": "COOLPROMPT (Без перевода)"},
        {"method": "coolprompt",      "translate": True,  "name": "COOLPROMPT (+ АНГЛ. БУСТ 🇬)"},
        {"method": "promptomatix",    "translate": False, "name": "PROMPTOMATIX (Официальный)"},
        {"method": "promptomatix",    "translate": True,  "name": "PROMPTOMATIX (+ АНГЛ. БУСТ 🇬)"},
    ]
    
    model_name = "deepseek/deepseek-v4-flash"
    s_eng = RobustLLMEngine(model_name)
    t_eng = RobustLLMEngine(model_name)
    
    for idx, prompt in enumerate(test_prompts, 1):
        task_desc = s_eng.generate("Extract the core objective from the prompt in 1 sentence.", prompt)
        test_data = DataGenerator(s_eng).generate_samples(task_desc, num_samples=3)
        evaluator = Evaluator(t_eng, s_eng)
        
        baseline_score = evaluator.score_prompt(prompt, test_data)
        
        results_table = []
        for cfg in test_configs:
            req = OptimizeRequest(
                prompt=prompt, method=cfg['method'], 
                target_model=model_name, system_model=model_name,
                evaluate=False, translate=cfg['translate']
            )
            
            start_time = time.time()
            try:
                res = optimize(req)
                elapsed = time.time() - start_time
                final_score = evaluator.score_prompt(res.optimized_prompt, test_data)
                if final_score >= 0.75 or final_score >= (baseline_score - 0.15):
                    meaning_preserved = "Да"
                else:
                    meaning_preserved = "Потерян"
                
                score_str = f"{baseline_score:.2f} -> {final_score:.2f}"
                
                results_table.append([
                    cfg['name'], 
                    f"{res.init_tokens} -> {res.final_tokens}", 
                    f"{elapsed:.1f}s", 
                    score_str,
                    meaning_preserved
                ])
            except Exception as e:
                results_table.append([cfg['name'], "ERROR", "-", "-", "Ошибка"])

        print("\n" + tabulate(
            results_table, 
            headers=["Алгоритм", "Токены", "Время", "Оценка (Ориг -> Итог)", "Смысл сохранен?"], 
            tablefmt="pretty"
        ))

if __name__ == "__main__":
    run_benchmark()