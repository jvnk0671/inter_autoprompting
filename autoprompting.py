from promptomatix_wrapper import promptomatix_optimize
from pipeline import OptimizationResult, PromptOptimizer, radical_cut

# from promptomatix import promptomatix_optimize
# from my_promptomatix.tuner import FullPromptTuner


class ExampleOptimiser(PromptOptimizer):
    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        return OptimizationResult(
            optimized_prompt=radical_cut(prompt, ch_lim, uncertainty=0),
            init_metric=0.0,
            final_metric=0.0,
        )


# РАСКОММЕНТИРОВАННЫЙ КЛАСС ПРОМПТОМАТИКСА
class PromptomatixOptimizer(PromptOptimizer):
    def __init__(self, target_model: str, system_model: str):
        self.target_model = target_model
        self.system_model = system_model

    def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
        optimized = promptomatix_optimize(
            prompt=prompt,
            model=self.target_model,
            system_model=self.system_model,
            ch_lim=ch_lim
        )
        return OptimizationResult(
            optimized_prompt=optimized['optimized_prompt'],
            init_metric=optimized.get('init_metric'),
            final_metric=optimized.get('final_metric')
        )


# class CustomPromptomatixOptimizer(PromptOptimizer):
#     def __init__(self, target_model: str, system_model: str):
#         self.tuner = FullPromptTuner(
#             target_model=target_model,
#             system_model=system_model
#         )

#     def optimize(self, prompt: str, ch_lim: int) -> OptimizationResult:
#         # Мы используем method='hype' для расширения, а тюнер сам сделает distill под лимит
#         result = self.tuner.run(
#             start_prompt=prompt,
#             ch_lim=ch_lim,
#             method='hype',
#             epochs=1 # Можно увеличить до 2-3 для более сильной мутации, но это дольше
#         )

#         return OptimizationResult(
#             optimized_prompt=result['optimized_prompt'],
#             init_metric=result['init_metric'],
#             final_metric=result['final_metric']
#         )
