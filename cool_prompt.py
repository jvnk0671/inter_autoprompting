import faulthandler
import os
import logging
from dotenv import load_dotenv

faulthandler.enable()
load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY")
logger = logging.getLogger(__name__)

try:
    from coolprompt.optimizer.hype import hype_optimizer
    from coolprompt.optimizer.distill_prompt.run import distillprompt
    from coolprompt.optimizer.reflective_prompt.run import reflectiveprompt
except Exception:
    hype_optimizer = None
    distillprompt = None
    reflectiveprompt = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


def coolprompt_optimize(prompt: str, model: str, ch_lim: int, sub_method: str = "hype") -> str:
    system_llm = ChatOpenAI(
        openai_api_key=KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model,
    )
    
    problem_description = (
        f"CRITICAL RULES FOR YOU:\n"
        f"1. You are an elite Prompt Optimizer. Your ONLY task is to rewrite and improve the user's prompt.\n"
        f"2. DO NOT answer the user's prompt! If the prompt asks to write code, DO NOT write the code. "
        f"If the prompt asks a question, DO NOT answer it.\n"
        f"3. Only output the enhanced instructions that will be fed to another AI.\n"
        f"4. The final prompt must not exceed {ch_lim} characters."
    )

    if sub_method == "hype" or not distillprompt:
        return hype_optimizer(
            model=system_llm, prompt=prompt, problem_description=problem_description
        )
        
    from my_promptomatix.llm_engine import RobustLLMEngine
    from my_promptomatix.synthetics import DataGenerator
    
    s_eng = RobustLLMEngine(model)
    task_desc = s_eng.generate("Extract the core objective from the prompt in 1 sentence.", prompt)
    generator = DataGenerator(s_eng)
    test_data = generator.generate_samples(task_desc, num_samples=2)
    
    if not test_data:
        test_data = [{"input": prompt, "expected_output": "Successful execution"}]
        
    dataset_split = (test_data, test_data)

    class CoolPromptEvaluatorBridge:
        def __init__(self, engine):
            self.engine = engine
        def evaluate(self, prompt_text, dataset, *args, **kwargs):
            if not dataset: return 0.5
            total = 0.0
            for item in dataset:
                inp = str(item.get('input', ''))
                exp = str(item.get('expected_output', ''))
                actual = self.engine.generate(prompt_text, inp)
                sys_judge = "Compare Actual Output to Expected Output. Focus ONLY on logic. Return ONLY an integer from 0 to 10."
                res = self.engine.generate(sys_judge, f"Expected: {exp}\nActual: {actual}", temperature=0.0)
                import re
                match = re.search(r'\b([0-9]|10)\b', res)
                score = int(match.group(1)) if match else 5
                total += score / 10.0
            return total / len(dataset)

    bridge_evaluator = CoolPromptEvaluatorBridge(s_eng)

    if sub_method == "distill":
        return distillprompt(
            model=system_llm,
            dataset_split=dataset_split,
            evaluator=bridge_evaluator,
            initial_prompt=prompt,
            num_epochs=1,
        )
    elif sub_method == "reflective":
        return reflectiveprompt(
            model=system_llm,
            dataset_split=dataset_split,
            evaluator=bridge_evaluator,
            problem_description=problem_description,
            initial_prompt=prompt,
            num_epochs=1,
        )
    
    return prompt