"""
test_multi_action.py
--------------------
S5-3 — Multi-Action Command Support test suite (Sprint 5).

Covers the four acceptance criteria:
    1. Multiple actions can be parsed from one instruction
    2. Actions are planned in the correct order
    3. Actions execute in the correct order
    4. At least 5 multi-action instructions tested

Unit tests use the deterministic splitter, a fixture scene and MockRobot —
no API key, no network, no PyBullet needed:
    pytest tests/test_multi_action.py -v -m "not integration"

Tests marked @pytest.mark.integration call the real LLM backend and need a
valid key in .env:
    pytest tests/test_multi_action.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from llm_backend.multi_action import (
    split_instruction,
    is_multi_action,
    names_new_target,
    leading_verb,
)
from llm_backend.schema import (
    ParsedInstruction, MultiActionInstruction, ActionType, ConfidenceLevel,
)
from task_planner.planner import TaskPlanner
from simulation_backend.action_schema import CommandType
from simulation_backend.mock_robot import MockRobot
from simulation_backend.executor import Executor


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def scene():
    return {
        "objects": [
            {"label": "red block",    "position": (2.5, 1.0)},
            {"label": "blue block",   "position": (3.0, 2.0)},
            {"label": "green block",  "position": (1.5, 3.0)},
            {"label": "yellow block", "position": (4.0, 2.5)},
            {"label": "left tray",    "position": (6.0, 1.0)},
            {"label": "right tray",   "position": (8.0, 1.0)},
            {"label": "workstation",  "position": (5.0, 5.0)},
        ]
    }


@pytest.fixture
def planner():
    return TaskPlanner()


def _pi(action, obj, dest=None, spatial=None, raw=None):
    """Shorthand ParsedInstruction builder."""
    return ParsedInstruction(
        action=action,
        object_target=obj,
        destination=dest,
        spatial_relation=spatial,
        confidence=ConfidenceLevel.HIGH,
        raw_instruction=raw or f"{action.value} the {obj}",
    )


# ══ AC1 — Multiple actions parsed from one instruction ════════════════════════

# The 7 Sprint 5 evidence instructions. IDs match the sprint report table.
MULTI_ACTION_CASES = [
    ("MA01",
     "Move the green block to the left tray and then move the yellow block "
     "to the right tray.", 2),
    ("MA02",
     "Locate the red block, then move the blue block to the right tray.", 2),
    ("MA03",
     "Locate the yellow block then move the green block to the workstation.", 2),
    ("MA04",
     "Move the red block to the left tray, then the blue block to the right "
     "tray, then locate the green block.", 3),
    ("MA05",
     "Take the red block to the left tray then take the yellow block to the "
     "right tray and finally locate the blue block.", 3),
    ("MA06",
     "Pick up the red block and place it in the left tray, then pick up the "
     "blue block and place it in the right tray.", 2),
    ("MA07",
     "Move the red block to the left tray; move the yellow block to the "
     "workstation.", 2),
]

# Single-action instructions that must NOT be over-split.
SINGLE_ACTION_CASES = [
    ("SA01", "pick up the red block and place it in the left tray"),
    ("SA02", "grab the blue block then drop it near the workstation"),
    ("SA03", "move the green block next to the blue block"),
    ("SA04", "pick up the red block"),
    ("SA05", "where is the yellow block"),
    ("SA06", "pick up the red block then put it down"),
]


class TestInstructionSplitting:
    """AC1: multiple actions can be extracted from one instruction."""

    @pytest.mark.parametrize("case_id,instruction,expected", MULTI_ACTION_CASES)
    def test_multi_action_split_count(self, case_id, instruction, expected):
        segments = split_instruction(instruction)
        assert len(segments) == expected, (
            f"{case_id}: expected {expected} actions, got {len(segments)}: {segments}"
        )

    @pytest.mark.parametrize("case_id,instruction", SINGLE_ACTION_CASES)
    def test_single_action_not_over_split(self, case_id, instruction):
        segments = split_instruction(instruction)
        assert len(segments) == 1, (
            f"{case_id}: single action was wrongly split into {segments}"
        )

    def test_at_least_five_multi_action_instructions_covered(self):
        """AC4: at least 5 multi-action instructions tested."""
        assert len(MULTI_ACTION_CASES) >= 5

    def test_is_multi_action_predicate(self):
        assert is_multi_action("move the red block to the left tray then "
                               "move the blue block to the right tray") is True
        assert is_multi_action("pick up the red block") is False

    def test_empty_instruction_returns_no_segments(self):
        assert split_instruction("") == []
        assert split_instruction("   ") == []

    def test_pronoun_segment_is_a_continuation(self):
        assert names_new_target("drop it near the workstation") is False
        assert names_new_target("move the yellow block to the right tray") is True

    def test_verb_ellipsis_is_inherited(self):
        segments = split_instruction(
            "move the red block to the left tray, then the blue block to the right tray"
        )
        assert len(segments) == 2
        assert leading_verb(segments[1]) == "move"
        assert "blue block" in segments[1]


class TestSegmentOrder:
    """AC2 (parse side): split segments keep the order they were written in."""

    def test_order_is_preserved(self):
        segments = split_instruction(
            "move the green block to the left tray and then move the yellow "
            "block to the right tray"
        )
        assert "green" in segments[0]
        assert "yellow" in segments[1]

    def test_three_action_order_is_preserved(self):
        segments = split_instruction(
            "move the red block to the left tray, then the blue block to the "
            "right tray, then locate the green block"
        )
        assert [s.split()[1] if s.split()[0] != "locate" else "green"
                for s in segments][0] == "the"          # sanity: non-empty
        assert "red" in segments[0]
        assert "blue" in segments[1]
        assert "green" in segments[2]


# ══ AC2 — Actions planned in the correct order ════════════════════════════════

class TestMultiActionPlanning:

    def test_two_actions_planned_in_order(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "green block", "left tray",
                raw="move the green block to the left tray"),
            _pi(ActionType.MOVE, "yellow block", "right tray",
                raw="move the yellow block to the right tray"),
        ]
        plan = planner.plan_multi_step(actions, scene)

        targets = [c.target_object for c in plan.commands]
        assert targets.index("green block") < targets.index("yellow block")

    def test_step_numbers_are_sequential_and_start_at_one(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray"),
            _pi(ActionType.MOVE, "blue block", "right tray"),
            _pi(ActionType.LOCATE, "green block"),
        ]
        plan  = planner.plan_multi_step(actions, scene)
        steps = [c.step for c in plan.commands]
        assert steps == list(range(1, len(steps) + 1))

    def test_combined_plan_is_longer_than_either_sub_plan(self, planner, scene):
        a1 = _pi(ActionType.MOVE, "red block", "left tray")
        a2 = _pi(ActionType.MOVE, "blue block", "right tray")

        single   = planner.generate_plan(a1, scene)
        combined = planner.plan_multi_step([a1, a2], scene)
        assert combined.total_steps > single.total_steps

    def test_three_actions_produce_three_sub_plans(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray"),
            _pi(ActionType.MOVE, "blue block", "right tray"),
            _pi(ActionType.LOCATE, "green block"),
        ]
        plan   = planner.plan_multi_step(actions, scene)
        picks  = [c for c in plan.commands if c.command_type == CommandType.PICK]
        places = [c for c in plan.commands if c.command_type == CommandType.PLACE]
        assert len(picks) == 2 and len(places) == 2

    def test_original_scene_is_not_mutated(self, planner, scene):
        before = scene["objects"][0]["position"]
        planner.plan_multi_step(
            [_pi(ActionType.MOVE, "red block", "left tray")], scene
        )
        assert scene["objects"][0]["position"] == before

    def test_predicted_scene_state_between_actions(self, planner, scene):
        """
        Action 2 acts on the block action 1 already moved, so it must be planned
        against the block's NEW position, not its original one.
        """
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray",
                raw="move the red block to the left tray"),
            _pi(ActionType.MOVE, "red block", "right tray",
                raw="move the red block to the right tray"),
        ]
        plan = planner.plan_multi_step(actions, scene)

        # The "navigate to red block" move of the SECOND action should aim at
        # the left tray (where action 1 left it), not the original position.
        moves_to_block = [
            c for c in plan.commands
            if c.command_type == CommandType.MOVE
            and c.target_object == "red block"
            and c.target_position is not None
        ]
        assert len(moves_to_block) == 2
        assert moves_to_block[0].target_position.as_tuple()[:2] == (2.5, 1.0)
        assert moves_to_block[1].target_position.as_tuple()[:2] == (6.0, 1.0)

    def test_instruction_string_records_every_action(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray",
                raw="move the red block to the left tray"),
            _pi(ActionType.LOCATE, "green block", raw="locate the green block"),
        ]
        plan = planner.plan_multi_step(actions, scene)
        assert "move the red block to the left tray" in plan.instruction
        assert "locate the green block" in plan.instruction


# ══ Safe failure ══════════════════════════════════════════════════════════════

class TestMultiActionSafeFailure:

    def test_missing_object_names_the_failing_action(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray",
                raw="move the red block to the left tray"),
            _pi(ActionType.MOVE, "purple block", "right tray",
                raw="move the purple block to the right tray"),
        ]
        with pytest.raises(ValueError) as exc:
            planner.plan_multi_step(actions, scene)

        message = str(exc.value)
        assert "Action 2/2" in message
        assert "purple block" in message

    def test_single_gripper_conflict_is_caught_at_plan_time(self, planner, scene):
        """
        "pick up the red block, then place the blue block in the right tray"
        is impossible for a one-gripper robot: the red block is never put down.
        The planner must say so BEFORE anything executes.
        """
        actions = [
            _pi(ActionType.PICK, "red block",
                raw="pick up the red block"),
            _pi(ActionType.PLACE, "blue block", "right tray",
                raw="place the blue block in the right tray"),
        ]
        with pytest.raises(ValueError) as exc:
            planner.plan_multi_step(actions, scene)

        message = str(exc.value)
        assert "Action 2/2" in message
        assert "red block" in message
        assert "gripper" in message

    def test_gripper_is_free_again_after_a_place(self, planner, scene):
        """A pick WITH a destination releases the gripper, so action 2 is fine."""
        actions = [
            _pi(ActionType.PICK, "red block", "left tray",
                raw="pick up the red block and place it in the left tray"),
            _pi(ActionType.PICK, "blue block", "right tray",
                raw="pick up the blue block and place it in the right tray"),
        ]
        plan = planner.plan_multi_step(actions, scene)
        assert plan.total_steps == 10

    def test_missing_destination_fails_with_a_reason(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "loading dock",
                raw="move the red block to the loading dock"),
        ]
        with pytest.raises(ValueError) as exc:
            planner.plan_multi_step(actions, scene)
        assert "loading dock" in str(exc.value)


# ══ AC3 — Actions execute in the correct order ════════════════════════════════

class TestMultiActionExecution:

    def test_full_multi_action_plan_executes(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "green block", "left tray",
                raw="move the green block to the left tray"),
            _pi(ActionType.MOVE, "yellow block", "right tray",
                raw="move the yellow block to the right tray"),
        ]
        plan = planner.plan_multi_step(actions, scene)

        robot = MockRobot()
        robot.load_scene(scene)
        result = Executor(robot).execute(plan, verbose=False)

        assert result.success is True
        assert result.steps_completed == plan.total_steps

    def test_three_action_plan_executes_in_order(self, planner, scene):
        actions = [
            _pi(ActionType.MOVE, "red block", "left tray",
                raw="move the red block to the left tray"),
            _pi(ActionType.MOVE, "blue block", "right tray",
                raw="move the blue block to the right tray"),
            _pi(ActionType.LOCATE, "green block", raw="locate the green block"),
        ]
        plan = planner.plan_multi_step(actions, scene)

        robot = MockRobot()
        robot.load_scene(scene)
        result = Executor(robot).execute(plan, verbose=False)

        assert result.success is True

        picked = [
            c.target_object for c in plan.commands
            if c.command_type == CommandType.PICK
        ]
        assert picked == ["red block", "blue block"]


# ══ Schema ════════════════════════════════════════════════════════════════════

class TestMultiActionSchema:

    def test_wrapper_reports_count_and_primary(self):
        a1 = _pi(ActionType.MOVE, "green block", "left tray")
        a2 = _pi(ActionType.MOVE, "yellow block", "right tray")
        multi = MultiActionInstruction(
            raw_instruction="move both blocks",
            actions=[a1, a2],
            is_multi_action=True,
            confidence=ConfidenceLevel.HIGH,
            segments=["move the green block", "move the yellow block"],
        )
        assert multi.action_count == 2
        assert multi.primary is a1
        assert "green block" in multi.summary()
        assert "yellow block" in multi.summary()


# ══ Integration — real LLM backend (needs an API key) ═════════════════════════

@pytest.mark.integration
class TestMultiActionLLMIntegration:

    @pytest.mark.parametrize("case_id,instruction,expected", MULTI_ACTION_CASES)
    def test_parse_multi_instruction(self, case_id, instruction, expected):
        from llm_backend.custom_LLM_parser import parse_multi_instruction

        result = parse_multi_instruction(instruction)
        assert result.is_multi_action is True
        assert result.action_count == expected, f"{case_id}: {result.summary()}"
        assert len(result.actions) == expected

    def test_single_action_still_returns_one_action(self):
        from llm_backend.custom_LLM_parser import parse_multi_instruction

        result = parse_multi_instruction(
            "pick up the red block and place it in the left tray"
        )
        assert result.is_multi_action is False
        assert result.action_count == 1
        assert result.primary.object_target.lower().startswith("red")

    def test_end_to_end_multi_action_pipeline(self, scene, planner):
        from llm_backend.custom_LLM_parser import parse_multi_instruction

        parsed_set = parse_multi_instruction(
            "Move the green block to the left tray and then move the yellow "
            "block to the right tray."
        )
        plan = planner.plan_multi_step(parsed_set.actions, scene)

        robot = MockRobot()
        robot.load_scene(scene)
        result = Executor(robot).execute(plan, verbose=False)

        assert result.success is True
        targets = [c.target_object for c in plan.commands]
        assert targets.index("green block") < targets.index("yellow block")
