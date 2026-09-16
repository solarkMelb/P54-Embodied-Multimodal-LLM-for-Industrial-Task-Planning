"""
llm_backend
-----------
Natural language instruction parser for the Multimodal LLM Industrial Task Planning project.
COS40005 Capstone -- Swinburne University / ARENA2036

Supports OpenAI, Gemini, DeepSeek, and Ollama (local, no API key) backends.
Select via LLM_BACKEND in your .env:
    LLM_BACKEND=openai        (default)
    LLM_BACKEND=gemini
    LLM_BACKEND=deepseek
    LLM_BACKEND=ollama

Public interface:
    from llm_backend import parse_instruction
    from llm_backend import ParsedInstruction, ActionType, ConfidenceLevel
"""

from .schema import ParsedInstruction, ActionType, ConfidenceLevel

# Lazy import so API key is not required at import time
def parse_instruction(instruction: str, max_retries: int = 2):
    from .custom_LLM_parser import parse_instruction as _parse
    return _parse(instruction, max_retries)

__all__ = ["parse_instruction", "ParsedInstruction", "ActionType", "ConfidenceLevel"]