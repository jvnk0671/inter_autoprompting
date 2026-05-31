import os
import sys
import warnings
import logging

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("hype.hype_optimizer").setLevel(logging.ERROR)
logging.getLogger("promptomatix").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import time
from tabulate import tabulate
from main import OptimizeRequest, optimize
from my_promptomatix.llm_engine import RobustLLMEngine
from my_promptomatix.synthetics import DataGenerator, Evaluator

logging.getLogger("autoprompting").setLevel(logging.ERROR)
logging.getLogger("my_promptomatix").setLevel(logging.ERROR)


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
    
    model_name = "meta-llama/llama-3.3-70b-instruct"
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
                prompt=prompt, 
                method=cfg['method'], 
                ch_limit=2000, 
                target_model=model_name, 
                system_model=model_name,
                evaluate=False,
                translate=cfg['translate']
            )
            
            start_time = time.time()
            try:
                res = optimize(req)
                elapsed = time.time() - start_time
                
                final_score = evaluator.score_prompt(res.optimized_prompt, test_data)
                meaning_preserved = " Да" if final_score >= (baseline_score - 0.1) else " Потерян"
                
                results_table.append([
                    cfg['name'], 
                    f"{res.init_tokens} -> {res.final_tokens}", 
                    f"{elapsed:.1f}s", 
                    f"{final_score:.2f}",
                    meaning_preserved
                ])
            except Exception as e:
                results_table.append([cfg['name'], "ERROR", "-", "-", f" Ошибка"])

        print("\n" + tabulate(
            results_table, 
            headers=["Алгоритм", "Токены", "Время", "Оценка (0-1)", "Смысл сохранен?"], 
            tablefmt="pretty"
        ))

if __name__ == "__main__":
    run_benchmark()