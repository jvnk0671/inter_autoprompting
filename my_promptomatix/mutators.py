from .llm_engine import RobustLLMEngine

class PromptMutator:
    def __init__(self, engine: RobustLLMEngine):
        self.engine = engine

    def hype(self, current_prompt: str, ch_lim: int, feedback: str = "") -> str:
        """Метод HYPE: добавляет персону, пошаговость и лучшие практики."""
        sys_prompt = (
            "You are an elite Prompt Engineer. Rewrite the user's input into a highly detailed prompt for an AI.\n\n"
            "STRICT RULES FOR YOU:\n"
            "1. You are writing a prompt, NOT solving the task. If the user asks for code, write a prompt asking for code, but DO NOT write the code yourself.\n"
            "2. Never include introductory phrases like 'Here is your prompt'. Output ONLY the raw prompt text.\n"
            f"3. Maximum length: {ch_lim} characters.\n\n"
            "HOW TO UPGRADE THE PROMPT (HYPE Framework):\n"
            "- Add a specific Expert Persona (e.g., 'Act as a Senior Python Developer').\n"
            "- Ask the AI to think step-by-step (Chain of Thought).\n"
            "- Structure the prompt with clear Markdown headers."
        )
        if feedback:
            sys_prompt += f"\nIncorporate this feedback to improve: {feedback}"
            
        return self.engine.generate(sys_prompt, current_prompt)

    def distill(self, current_prompt: str, ch_lim: int) -> str:
        sys_prompt = (
            f"You are a Prompt Compressor. Distill the prompt to its absolute core instructions. "
            f"Remove all conversational filler, politeness, and unnecessary words. "
            f"CRITICAL: Must be STRICTLY UNDER {ch_lim} characters. "
            f"Return ONLY the compressed prompt."
        )
        return self.engine.generate(sys_prompt, current_prompt)