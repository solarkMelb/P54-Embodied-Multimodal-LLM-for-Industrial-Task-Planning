"""
multi_action.py
---------------
S5-3 — Multi-Action Command Support (Sprint 5).

Splits a single natural language instruction that contains SEVERAL sequential
actions into an ordered list of single-action sub-instructions, so each one can
be sent through the existing parse_instruction() -> TaskPlanner path.

Design notes
------------
The hard part is not finding connectors, it is deciding which connector marks a
genuinely new action and which one is just the second half of one pick-and-place:

    "pick up the red block and place it in the left tray"
        -> ONE action  (pick with a destination)

    "grab the blue block then drop it near the workstation"
        -> ONE action  ("it" refers back to the blue block)

    "move the green block to the left tray and then move the yellow block
     to the right tray"
        -> TWO actions (a second, different object is named)

The rule used here is therefore object-driven, not connector-driven:

    A segment starts a NEW action only if it names its own concrete workspace
    object.  A segment whose object slot holds a pronoun ("it", "them", "that")
    is a continuation and is merged back into the previous segment.

Verb ellipsis is handled too:

    "move the red block to the left tray, then the blue block to the right tray"

the second segment names a real object but has no verb, so the verb of the
previous segment is inherited.

Usage:
    from llm_backend.multi_action import split_instruction
    split_instruction("move the green block to the left tray "
                      "and then move the yellow block to the right tray")
    # -> ["move the green block to the left tray",
    #     "move the yellow block to the right tray"]
"""

import re
import logging
from typing import Optional

from .edge_cases import ACTION_SYNONYMS, ALLOWED_ACTIONS

logger = logging.getLogger(__name__)


# -- Workspace vocabulary -----------------------------------------------------
# Kept in sync with simulation_backend/scene_config.yaml and the 7 fine-tuned
# YOLO classes (red/blue/green/yellow block, left tray, right tray, workstation).
OBJECT_COLOURS = ("red", "blue", "green", "yellow")
OBJECT_NOUNS   = ("block", "cube", "tray", "workstation", "table", "box")

# Words that refer BACK to the object of the previous segment.
PRONOUNS = ("it", "them", "they", "that", "this", "those", "these")

# Connectors that may (but do not always) separate two sequential actions.
# Longest first so "and then" wins over "then".
# NOTE: bare "next" is deliberately excluded — "next to the blue block" is a
# spatial relation, not a sequence connector.
SEQUENCE_CONNECTORS = (
    "and after that",
    "and afterwards",
    "and finally",
    "and then",
    "after that",
    "afterwards",
    "followed by",
    "finally",
    "then",
)

# Every verb the parser understands, plus its synonyms.
_ACTION_VERBS = tuple(sorted(set(ALLOWED_ACTIONS) | set(ACTION_SYNONYMS.keys())))
_VERB_ALT     = "|".join(re.escape(v) for v in _ACTION_VERBS)
_VERB_RE      = re.compile(rf"\b({_VERB_ALT})\b", flags=re.IGNORECASE)

_SEMICOLON_RE = re.compile(r"\s*[;]\s*")
_CONNECTOR_RE = re.compile(
    r"\s*(?:,|\.)?\s*\b(?:"
    + "|".join(re.escape(c) for c in SEQUENCE_CONNECTORS)
    + r")\b\s*",
    flags=re.IGNORECASE,
)
_COMMA_AND_RE  = re.compile(r"\s*,\s*and\b\s*", flags=re.IGNORECASE)
_COMMA_VERB_RE = re.compile(rf"\s*,\s*(?=(?:{_VERB_ALT})\b)", flags=re.IGNORECASE)

_PRONOUN_TAIL_RE = re.compile(
    rf"\s+(?:up\s+|down\s+|off\s+)?(?:{'|'.join(PRONOUNS)})\b",
    flags=re.IGNORECASE,
)
_GRASPABLE_RE = re.compile(
    rf"\b(?:{'|'.join(OBJECT_COLOURS)})\s+(?:block|cube)\b", flags=re.IGNORECASE
)


# -- Helpers ------------------------------------------------------------------

def _tidy(text: str) -> str:
    """Strip stray punctuation/whitespace and leading conjunctions."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:and|then|,|;|\.)+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[,;\.\s]+$", "", text)
    return text.strip()


def has_concrete_object(segment: str) -> bool:
    """
    True if the segment names a real workspace object rather than a pronoun.

    "move the yellow block to the right tray" -> True
    "drop it near the workstation"            -> True  (workstation is concrete)
    "place it over there"                     -> False
    """
    lowered = segment.lower()
    return any(re.search(rf"\b{noun}s?\b", lowered) for noun in OBJECT_NOUNS)


def mentions_graspable(segment: str) -> bool:
    """True if the segment names a coloured block/cube — a graspable target."""
    return bool(_GRASPABLE_RE.search(segment))


def leading_verb(segment: str) -> Optional[str]:
    """Return the first action verb in the segment, or None."""
    match = _VERB_RE.search(segment)
    return match.group(1).lower() if match else None


def names_new_target(segment: str) -> bool:
    """
    True if the segment introduces a NEW action rather than continuing the
    previous one.

    A segment whose object slot is a pronoun ("drop it near the workstation")
    is a continuation, because the thing being acted on is the object the robot
    is already holding.
    """
    first_verb = _VERB_RE.search(segment)

    # The object of the segment's FIRST verb is a pronoun -> continuation.
    # Only the first verb matters: "pick up the blue block and place it in the
    # right tray" is a new action even though its second verb takes "it".
    if first_verb and _PRONOUN_TAIL_RE.match(segment[first_verb.end():]):
        return False

    # No verb at all: verb ellipsis. Still a new action if it names an object.
    if not first_verb:
        return has_concrete_object(segment)

    # Verb + a bare pronoun and no graspable target named -> continuation.
    has_pronoun = any(re.search(rf"\b{p}\b", segment, re.IGNORECASE) for p in PRONOUNS)
    if has_pronoun and not mentions_graspable(segment):
        return False

    return has_concrete_object(segment)


def _inherit_verb(segment: str, previous_verb: Optional[str]) -> str:
    """
    Prepend the previous segment's verb when this segment omits its own.

    "the blue block to the right tray" + verb="move"
        -> "move the blue block to the right tray"
    """
    if previous_verb and not _VERB_RE.search(segment):
        return f"{previous_verb} {segment}"
    return segment


# -- Public API ---------------------------------------------------------------

def split_instruction(instruction: str) -> list[str]:
    """
    Split a natural language instruction into ordered single-action segments.

    Always returns at least one segment for a non-empty instruction, so callers
    can treat single- and multi-action instructions uniformly.

    Args:
        instruction: Raw natural language instruction.

    Returns:
        Ordered list of sub-instruction strings — execution order preserved.
    """
    if not instruction or not instruction.strip():
        return []

    text = re.sub(r"\s+", " ", instruction.strip())
    tidy = _tidy(text)

    # Split on hard boundaries first, then soft connectors, then comma forms.
    parts: list[str] = [text]
    for pattern in (_SEMICOLON_RE, _CONNECTOR_RE, _COMMA_AND_RE, _COMMA_VERB_RE):
        expanded: list[str] = []
        for part in parts:
            expanded.extend(pattern.split(part))
        parts = expanded

    candidates = [c for c in (_tidy(p) for p in parts) if c]

    if len(candidates) <= 1:
        return [tidy] if tidy else []

    # Merge continuations back into the segment they belong to.
    segments: list[str] = [candidates[0]]
    for candidate in candidates[1:]:
        if names_new_target(candidate):
            segments.append(_inherit_verb(candidate, leading_verb(segments[-1])))
        else:
            # Continuation — rejoin with "then" so the sub-parser still sees the
            # complete pick-and-place phrasing.
            segments[-1] = f"{segments[-1]} then {candidate}"

    logger.debug("[multi-action] %r -> %r", instruction, segments)
    return segments


def is_multi_action(instruction: str) -> bool:
    """Convenience predicate — True if the instruction contains 2+ actions."""
    return len(split_instruction(instruction)) > 1
