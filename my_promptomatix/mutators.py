from .llm_engine import RobustLLMEngine

class PromptMutator:
    """Реализация методов оптимизации из оригинального Promptomatix."""
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def hype(self, current_prompt: str, feedback: str = "") -> str:
        """Метод HYPE: добавляет персону, пошаговость и лучшие практики."""
        sys_prompt = (
            "You are an elite Prompt Engineer. Enhance the user's prompt using the 'HYPE' method: "
            "1. Assign an expert persona. "
            "2. Add step-by-step reasoning instructions (Chain of Thought). "
            "3. Structure with clear Markdown headers. "
            "Return ONLY the rewritten prompt."
        )
        if feedback:
            sys_prompt += f"\nIncorporate this feedback to improve: {feedback}"
            
        return self.engine.generate(sys_prompt, current_prompt)

    def distill(self, current_prompt: str, ch_lim: int) -> str:
        """Метод DISTILL: сжимает промпт, сохраняя только инструкции, чтобы влезть в лимит."""
        sys_prompt = (
            f"You are a Prompt Compressor. Distill the prompt to its absolute core instructions. "
            f"Remove all conversational filler, politeness, and unnecessary words. "
            f"CRITICAL: Must be STRICTLY UNDER {ch_lim} characters. "
            f"Return ONLY the compressed prompt."
        )
        return self.engine.generate(sys_prompt, current_prompt)