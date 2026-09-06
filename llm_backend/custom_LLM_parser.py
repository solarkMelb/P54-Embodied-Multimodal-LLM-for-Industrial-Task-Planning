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
    LLM_BACKEND=huggingface          -- uses local HuggingFace model

All API credentials and model config are owned entirely by the backend
modules (backends/openai_backend.py, backends/gemini_backend.py,
backends/deepseek_backend.py, backends/huggingface_backend.py). 
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

from .schema import ParsedInstruction, MultiActionInstruction, ConfidenceLevel
from .prompts import build_system_prompt, build_multi_action_prompt
from .edge_cases import (
    is_empty_instruction,
    is_too_vague,
    normalise_instruction,
    validate_parsed_result,
    make_vague_result,
)
from .backends import get_llm
from .multi_action import split_instruction

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
# S5-3: prompt used when parsing one sub-instruction of a multi-action command.
multi_action_prompt = build_multi_action_prompt(output_parser.get_format_instructions())

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
def parse_instruction(
    instruction: str,
    max_retries: int = 2,
    system_prompt_override: str | None = None,
) -> ParsedInstruction:
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
                SystemMessage(content=system_prompt_override or system_prompt),
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

# -- S5-3: Multi-action public interface ---------------------------------------
_CONFIDENCE_RANK = {
    ConfidenceLevel.HIGH:   3,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.LOW:    1,
}


def parse_multi_instruction(
    instruction: str,
    max_retries: int = 2,
) -> MultiActionInstruction:
    """
    S5-3 — Parse an instruction that may contain SEVERAL sequential actions.

    Pipeline:
        1. Split the instruction into ordered single-action segments
           (llm_backend/multi_action.py — deterministic, no API call).
        2. Parse each segment with the existing parse_instruction(), so all
           existing behaviour is preserved per action: edge-case pre-checks,
           synonym mapping, retries, disk cache, and post-validation.
        3. Collect the results in EXECUTION ORDER.

    A single-action instruction still returns a MultiActionInstruction, with
    one action and is_multi_action=False, so callers need only one code path.

    Args:
        instruction : Raw natural language instruction.
        max_retries : Retries per sub-instruction on JSON parse failure.

    Returns:
        MultiActionInstruction — actions in execution order.

    Raises:
        ValueError: If the instruction is empty, or if any sub-instruction
                    fails to parse. The error names WHICH action failed so the
                    pipeline can report a clear reason instead of failing
                    silently or dropping the remaining actions.
    """
    if is_empty_instruction(instruction):
        raise ValueError("Instruction cannot be empty.")

    segments = split_instruction(instruction)
    if not segments:
        raise ValueError("Instruction cannot be empty.")

    multi = len(segments) > 1
    if multi:
        logger.info(
            f"[multi-action] Split into {len(segments)} actions: {segments}"
        )

    prompt_override = multi_action_prompt if multi else None

    actions: list[ParsedInstruction] = []
    collected_notes: list[str] = []

    for index, segment in enumerate(segments, start=1):
        try:
            parsed = parse_instruction(
                segment,
                max_retries=max_retries,
                system_prompt_override=prompt_override,
            )
        except Exception as exc:
            raise ValueError(
                f"Action {index}/{len(segments)} ('{segment}') could not be "
                f"parsed: {exc}"
            ) from exc

        actions.append(parsed)
        if parsed.notes:
            collected_notes.append(f"[action {index}] {parsed.notes}")

    # Overall confidence is the WEAKEST link — a plan is only as trustworthy
    # as its least certain step.
    overall = min(actions, key=lambda a: _CONFIDENCE_RANK[a.confidence]).confidence

    return MultiActionInstruction(
        raw_instruction=instruction,
        actions=actions,
        is_multi_action=multi,
        confidence=overall,
        segments=segments,
        notes=" ".join(collected_notes) or None,
    )
