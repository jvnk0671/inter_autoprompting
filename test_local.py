import sys
from main import OptimizeRequest, optimize

def run_tests():
    test_prompt = "You are an assistant. Help me write a python script to reverse a string."
    methods_to_test = ["my_promptomatix", "promptomatix", "coolprompt"]
    
    for method in methods_to_test:
        req = OptimizeRequest(
            prompt=test_prompt,
            method=method,
            ch_limit=3000,
            uncertainty=20,
            target_model="meta-llama/llama-3.3-70b-instruct",
            system_model="meta-llama/llama-3.3-70b-instruct"
        )
        
        try:
            res = optimize(req)
            print(f"📊 Токены (до/после): {res.init_tokens} -> {res.final_tokens}")
            print(f"✨ Результат:\n{res.optimized_prompt}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Ошибка в методе {method}: {e}")
            print("-" * 50)

if __name__ == "__main__":
    run_tests()