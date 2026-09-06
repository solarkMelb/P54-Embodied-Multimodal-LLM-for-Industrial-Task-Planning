"""
prompts.py
----------
Centralised prompt templates and few-shot examples for the LLM parser.
Keeping prompts here (not buried in parser.py) makes them easy to tune
without touching any logic code.
"""

# -- Few-shot examples ---------------------------------------------------------
# These are shown to the LLM every call to lock in the expected output format.
# Cover: simple, spatial, multi-object, ambiguous, and edge cases.

FEW_SHOT_EXAMPLES = [
    {
        "instruction": "pick up the red block",
        "output": {
            "action": "pick",
            "object_target": "red block",
            "destination": None,
            "spatial_relation": None,
            "confidence": "high",
            "raw_instruction": "pick up the red block",
            "notes": None
        }
    },
    {
        "instruction": "place the blue cube in the left tray",
        "output": {
            "action": "place",
            "object_target": "blue cube",
            "destination": "left tray",
            "spatial_relation": "in",
            "confidence": "high",
            "raw_instruction": "place the blue cube in the left tray",
            "notes": None
        }
    },
    {
        "instruction": "move the green block to the right of the workstation",
        "output": {
            "action": "move",
            "object_target": "green block",
            "destination": "workstation",
            "spatial_relation": "right of",
            "confidence": "high",
            "raw_instruction": "move the green block to the right of the workstation",
            "notes": None
        }
    },
    {
        "instruction": "where is the yellow block",
        "output": {
            "action": "locate",
            "object_target": "yellow block",
            "destination": None,
            "spatial_relation": None,
            "confidence": "high",
            "raw_instruction": "where is the yellow block",
            "notes": None
        }
    },
    {
        "instruction": "put that thing over there",
        "output": {
            "action": "move",
            "object_target": "unknown",
            "destination": "unknown",
            "spatial_relation": None,
            "confidence": "low",
            "raw_instruction": "put that thing over there",
            "notes": "Instruction is too vague. Object and destination are not specified."
        }
    },
    {
        "instruction": "grab the red block and drop it near the blue tray",
        "output": {
            "action": "pick",
            "object_target": "red block",
            "destination": "blue tray",
            "spatial_relation": "near",
            "confidence": "high",
            "raw_instruction": "grab the red block and drop it near the blue tray",
            "notes": "Mapped 'grab' to 'pick' and 'drop' to 'place' as closest valid actions."
        }
    },
]


def format_few_shot_examples() -> str:
    """Format few-shot examples as a string for injection into the system prompt."""
    import json
    lines = ["Here are examples of correct input/output pairs:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Instruction: \"{ex['instruction']}\"")
        lines.append(f"  Output: {json.dumps(ex['output'], indent=4)}\n")
    return "\n".join(lines)


# -- System prompt -------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are an instruction parser for an intelligent industrial robot operating \
in a simulated workspace.

Your job is to read a natural language task instruction and extract structured \
information from it as JSON.

ALLOWED ACTIONS (use only these):
- pick   : picking up / grabbing an object
- place  : putting / dropping / placing an object somewhere
- move   : moving an object to a location
- locate : finding / locating an object

WORKSPACE OBJECTS: coloured blocks (red, blue, green, yellow), trays (left tray, \
right tray), workstations.

RULES:
1. Always return valid JSON matching the schema exactly.
2. If an action word is not in the allowed list, map it to the closest valid action \
   and explain in notes.
3. If the instruction is ambiguous or vague, set confidence to "low" and explain in notes.
4. If no destination is mentioned, set destination to null.
5. If no spatial relation is mentioned, set spatial_relation to null.
6. Never invent objects or locations not mentioned in the instruction.
7. Always preserve the original instruction in raw_instruction exactly as given.
8. If the object or destination is completely unclear, set them to "unknown".

{few_shot_examples}

{format_instructions}
"""


def build_system_prompt(format_instructions: str) -> str:
    """Build the final system prompt with few-shot examples and format instructions."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        few_shot_examples=format_few_shot_examples(),
        format_instructions=format_instructions,
    )

# -- S5-3: Multi-action guidance ----------------------------------------------
# Appended to the system prompt so the LLM knows it is being handed ONE action
# at a time, already separated by llm_backend/multi_action.py.  Without this the
# model sometimes tries to summarise a whole compound sentence into one action.

MULTI_ACTION_NOTE = """
MULTI-ACTION HANDLING:
Compound instructions are split upstream, so the text you receive is a SINGLE
action. Parse exactly what you are given — do not merge, drop, or invent
additional actions, and do not try to describe a sequence in one object.

A pick and its matching place are ONE action, not two:
    "pick up the red block and place it in the left tray"
        -> action="pick", object_target="red block", destination="left tray"
"""


# -- S5-3: Few-shot examples for split sub-instructions ------------------------
MULTI_ACTION_EXAMPLES = [
    {
        "instruction": "move the green block to the left tray",
        "output": {
            "action": "move",
            "object_target": "green block",
            "destination": "left tray",
            "spatial_relation": "in",
            "confidence": "high",
            "raw_instruction": "move the green block to the left tray",
            "notes": None
        }
    },
    {
        "instruction": "move the yellow block to the right tray",
        "output": {
            "action": "move",
            "object_target": "yellow block",
            "destination": "right tray",
            "spatial_relation": "in",
            "confidence": "high",
            "raw_instruction": "move the yellow block to the right tray",
            "notes": None
        }
    },
]


def build_multi_action_prompt(format_instructions: str) -> str:
    """
    Build a system prompt for parsing ONE sub-instruction of a multi-action
    command. Same contract as build_system_prompt(), plus MULTI_ACTION_NOTE
    and the multi-action few-shot examples.
    """
    import json

    lines = ["Additional examples of split sub-instructions:\n"]
    for i, ex in enumerate(MULTI_ACTION_EXAMPLES, 1):
        lines.append(f"Multi-action example {i}:")
        lines.append(f"  Instruction: \"{ex['instruction']}\"")
        lines.append(f"  Output: {json.dumps(ex['output'], indent=4)}\n")

    return (
        SYSTEM_PROMPT_TEMPLATE.format(
            few_shot_examples=format_few_shot_examples(),
            format_instructions=format_instructions,
        )
        + MULTI_ACTION_NOTE
        + "\n"
        + "\n".join(lines)
    )
