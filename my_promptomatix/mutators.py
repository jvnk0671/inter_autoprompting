from .llm_engine import RobustLLMEngine
import config

class PromptMutator:
    """Реализация методов оптимизации из оригинального Promptomatix."""
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def hype(self, current_prompt: str, feedback: str = "") -> str:
        sys_prompt = config.HYPE_PROMPT
        if feedback:
            sys_prompt += f"\nIncorporate this feedback to improve: {feedback}"
        return self.engine.generate(sys_prompt, current_prompt)

    def distill(self, current_prompt: str, ch_lim: int) -> str:
        sys_prompt = config.DISTILL_PROMPT.format(ch_lim=ch_lim)
        return self.engine.generate(sys_prompt, current_prompt)