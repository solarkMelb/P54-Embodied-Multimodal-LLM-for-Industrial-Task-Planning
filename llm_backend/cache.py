"""
cache.py
--------
Disk-based LLM response cache.
Caches ParsedInstruction outputs, keyed by instruction text + backend model +
system prompt (see _cache_key()) — an instruction parsed under a different
prompt, such as a multi-action sub-instruction, is a different cache entry.
On cache hit — returns instantly, zero API/LLM call.
On cache miss — calls LLM, saves result for next time.

Set in .env:
    LLM_CACHE_ENABLED=true      (default: false)
    LLM_CACHE_PATH=.llm_cache   (default: .llm_cache/)
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "false").lower() == "true"
CACHE_PATH    = Path(os.getenv("LLM_CACHE_PATH", ".llm_cache"))


def _cache_key(instruction: str, model: str, prompt: str = "") -> str:
    """
    Generate a stable cache key from instruction + model + system prompt.

    The prompt is part of the key because the same instruction parsed under a
    different system prompt is a different question: a multi-action segment sent
    with build_multi_action_prompt() and the same text sent with the standard
    prompt would otherwise collide, and whichever ran first would be returned
    for both.
    """
    prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:12] if prompt else "default"
    raw = f"{model}::{prompt_id}::{instruction.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(instruction: str, model: str, prompt: str = "") -> Optional[dict]:
    """Return cached result dict or None on miss."""
    if not CACHE_ENABLED:
        return None
    key  = _cache_key(instruction, model, prompt)
    path = CACHE_PATH / f"{key}.json"
    if path.exists():
        logger.info(f"[Cache] HIT — '{instruction[:50]}'")
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(instruction: str, model: str, result: dict, prompt: str = "") -> None:
    """Save result dict to cache."""
    if not CACHE_ENABLED:
        return
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    key  = _cache_key(instruction, model, prompt)
    path = CACHE_PATH / f"{key}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[Cache] SAVED — '{instruction[:50]}'")


def clear_cache() -> int:
    """Delete all cached responses. Returns count deleted."""
    if not CACHE_PATH.exists():
        return 0
    files = list(CACHE_PATH.glob("*.json"))
    for f in files:
        f.unlink()
    logger.info(f"[Cache] Cleared {len(files)} entries")
    return len(files)


def cache_stats() -> dict:
    """Return cache statistics."""
    if not CACHE_PATH.exists():
        return {"enabled": CACHE_ENABLED, "entries": 0, "size_kb": 0}
    files = list(CACHE_PATH.glob("*.json"))
    size  = sum(f.stat().st_size for f in files)
    return {
        "enabled":  CACHE_ENABLED,
        "entries":  len(files),
        "size_kb":  round(size / 1024, 1),
        "path":     str(CACHE_PATH),
    }