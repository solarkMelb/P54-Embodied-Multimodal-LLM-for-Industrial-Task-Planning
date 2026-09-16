"""
custom_LLM_parser.py
--------------------
LLM instruction parser -- backend-agnostic.
Converts natural language robot instructions into structured JSON
conforming to the ParsedInstruction schema.

Backend is selected at runtime via the LLM_BACKEND environment variable:
    LLM_BACKEND=openai    (default) -- uses OpenAI GPT-4o
    LLM_BACKEND=gemini               -- uses Google Gemini
    LLM_BACKEND=deepseek             -- uses DeepSeek
    LLM_BACKEND=ollama               -- uses a local Ollama model, no API key

All API credentials and model config are owned entirely by the backend
modules (backends/openai_backend.py, backends/gemini_backend.py,
backends/deepseek_backend.py, backends/ollama_backend.py).
This file contains zero credential logic.

Full pipeline per call:
    1. Pre-check  : reject empty instructions immediately.
    2. Pre-check  : short-circuit vague instructions (no API call made).
    3. Normalise  : strip/collapse whitespace before sending to LLM.
    4. LLM call   : with configurable retry on JSON parse failure.
    5. Post-check : downgrade confidence for unknown objects/destinations.

Usage:
    from llm_backend.custom_LLM_parser import parse_instruction
    result = parse_instruction("pick up the red block")
"""

import os
import logging
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from .schema import ParsedInstruction
from .prompts import build_system_prompt
from .edge_cases import (
    is_empty_instruction,
    is_too_vague,
    normalise_instruction,
    validate_parsed_result,
    make_vague_result,
)
from .backends import get_llm

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -- Environment ---------------------------------------------------------------
load_dotenv()

# -- Setup ---------------------------------------------------------------------
# Build the output parser and system prompt once at module load.
# We use SystemMessage / HumanMessage directly instead of ChatPromptTemplate
# to avoid LangChain's f-string template engine misinterpreting the JSON
# braces inside the system prompt (format_instructions + few-shot examples).
output_parser = PydanticOutputParser(pydantic_object=ParsedInstruction)
system_prompt = build_system_prompt(output_parser.get_format_instructions())

# -- Lazy LLM initialisation --------------------------------------------------
# The LLM is NOT created at import time. This avoids crashing the entire
# module if an API key is missing or the backend is misconfigured.
# The LLM client is built on first use and then cached for subsequent calls.
_llm = None


def _get_llm():
    """Return the cached LLM client, building it on first call."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


# -- Helpers ------------------------------------------------------------------
def _clean_json(text: str) -> str:
    """
    Strip markdown code fences that some LLMs (e.g. OpenAI) wrap around JSON.

    Examples of what this handles:
        ```json\n{...}\n```   ->  {...}
        ```\n{...}\n```       ->  {...}
        {...}                  ->  {...}  (returned unchanged)
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[-1]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


# -- Public interface ----------------------------------------------------------
def parse_instruction(instruction: str, max_retries: int = 2) -> ParsedInstruction:
    # existing pre-checks ...
    if is_empty_instruction(instruction):
        raise ValueError("Instruction cannot be empty.")
    if is_too_vague(instruction):
        return make_vague_result(instruction)

    instruction = normalise_instruction(instruction)
    model = os.getenv("LLM_BACKEND", "openai")

    # ── Cache check ──────────────────────────────────────────────────────────
    from llm_backend.cache import get_cached, save_cache
    cached = get_cached(instruction, model)
    if cached:
        logger.info(f"[Cache] Returning cached result for: '{instruction}'")
        return ParsedInstruction(**cached)

    logger.info(f"Parsing: '{instruction}' via {model} backend")

    # existing LLM call ...
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Instruction: {instruction}"),
            ]
            response = _get_llm().invoke(messages)
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = " ".join(
                    str(item) if not isinstance(item, dict)
                    else item.get("text", "")
                    for item in raw_content
                )
            result = output_parser.parse(_clean_json(raw_content))
            result = validate_parsed_result(result)

            # ── Save to cache ─────────────────────────────────────────────
            save_cache(instruction, model, result.model_dump(mode="json"))

            logger.info(f"Parsed successfully on attempt {attempt}: {result}")
            return result

        except OutputParserException as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed (OutputParserException): {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed ({type(e).__name__}): {e}")

    raise ValueError(
        f"Failed to parse instruction after {max_retries} attempts. "
        f"Last error: {last_error}"
    )