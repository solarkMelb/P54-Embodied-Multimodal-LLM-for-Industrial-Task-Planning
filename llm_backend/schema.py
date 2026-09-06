"""
schema.py
---------
Pydantic schema for the structured output returned by the LLM instruction parser.
All other modules (scene representation, task planner) should import from here
to ensure a consistent interface contract.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    LOCATE = "locate"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ParsedInstruction(BaseModel):
    """
    Structured output from the LLM instruction parser.

    Example:
        Instruction: "pick up the red block and place it in the left tray"
        Output:
            action="pick", object_target="red block",
            destination="left tray", spatial_relation="in",
            confidence="high", raw_instruction="pick up the red block..."
    """

    action: ActionType = Field(
        description="The primary robot action to perform."
    )
    object_target: str = Field(
        description="The object the action applies to (e.g. 'red block', 'blue cube')."
    )
    destination: Optional[str] = Field(
        default=None,
        description="Where the object should go, if applicable (e.g. 'left tray', 'workstation A')."
    )
    spatial_relation: Optional[str] = Field(
        default=None,
        description="Spatial relationship described (e.g. 'left of', 'near', 'on top of')."
    )
    confidence: ConfidenceLevel = Field(
        description="How confident the parser is in this interpretation."
    )
    raw_instruction: str = Field(
        description="The original instruction string, preserved for logging and debugging."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any ambiguities, warnings, or clarifications the LLM flagged."
    )

class MultiActionInstruction(BaseModel):
    """
    S5-3 — a single natural language instruction that contains one or more
    sequential actions.

    This is a *wrapper* around ParsedInstruction, not a replacement.  Every
    existing consumer of ParsedInstruction (task planner, vision, evaluation,
    tests) keeps working unchanged; multi-action support is a layer on top.

    Example:
        Instruction: "move the green block to the left tray and then move the
                      yellow block to the right tray"
        Output:
            is_multi_action=True,
            action_count=2,
            actions=[
                ParsedInstruction(action=move, object_target="green block",
                                  destination="left tray", ...),
                ParsedInstruction(action=move, object_target="yellow block",
                                  destination="right tray", ...),
            ]
    """

    raw_instruction: str = Field(
        description="The original, unsplit instruction exactly as the user typed it."
    )
    actions: list[ParsedInstruction] = Field(
        description="Parsed actions in EXECUTION ORDER. Never empty."
    )
    is_multi_action: bool = Field(
        default=False,
        description="True when the instruction contained two or more actions."
    )
    confidence: ConfidenceLevel = Field(
        description="Lowest confidence across all parsed actions."
    )
    segments: list[str] = Field(
        default_factory=list,
        description="The sub-instruction strings the original was split into."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Warnings, ambiguities, or per-action notes collected during parsing."
    )

    @property
    def action_count(self) -> int:
        """Number of actions extracted from the instruction."""
        return len(self.actions)

    @property
    def primary(self) -> ParsedInstruction:
        """
        The first action — used as the single-action fallback so existing
        code paths that expect one ParsedInstruction keep working.
        """
        return self.actions[0]

    def summary(self) -> str:
        """One-line human readable summary, useful for logs and evidence."""
        parts = [
            f"{i + 1}. {a.action.value} '{a.object_target}'"
            + (f" -> '{a.destination}'" if a.destination else "")
            for i, a in enumerate(self.actions)
        ]
        return f"{self.action_count} action(s): " + " | ".join(parts)
