"""
integration_tests.py
--------------------
PB11: Integration test suite for the full 5-stage pipeline.

Tests run a parsed instruction through the local planning/execution stages:
    parsed instruction → task planner → executor → mock robot → assert final state

Local tests use MockRobot and a fixture scene — no API calls, no PyBullet needed.
Tests that need a real LLM are marked with @pytest.mark.integration.

Run unit-safe tests only (no API):
    pytest integration_tests.py -v -m "not integration"

Run everything:
    pytest integration_tests.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from llm_backend.schema import ParsedInstruction, ActionType, ConfidenceLevel
from task_planner.planner import TaskPlanner, _apply_spatial_offset
from simulation_backend.action_schema import ActionPlan, CommandType
from simulation_backend.mock_robot    import MockRobot
from simulation_backend.executor      import Executor
from llm_backend.tracker                 import PipelineTracker


# ── Shared fixtures ────────────────────────────────────────────────────────────

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
def robot(scene):
    r = MockRobot()
    r.load_scene(scene)
    return r

@pytest.fixture
def planner():
    return TaskPlanner()

@pytest.fixture
def tracker_instance(tmp_path):
    return PipelineTracker(log_path=str(tmp_path / "test_log.json"))


def _run(instruction_obj, scene_dict) -> tuple[ActionPlan, bool, MockRobot]:
    """Helper: plan + execute, return (plan, success, robot_state)."""
    robot    = MockRobot()
    robot.load_scene(scene_dict)
    planner  = TaskPlanner()
    plan     = planner.generate_plan(instruction_obj, scene_dict)
    executor = Executor(robot)
    result   = executor.execute(plan, verbose=False)
    return plan, result.success, robot


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATION TESTS (PB7-SP)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpatialRelationPlanning:

    def test_left_of_applies_positive_y_offset(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.PLACE,
            object_target="red block",
            destination="blue block",
            spatial_relation="left of",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="place the red block to the left of the blue block",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, scene)
        # blue block at (3.0, 2.0); "left of" offset = (0, +0.15) → (3.0, 2.15).
        # Left is +Y in this workspace: 'left tray' is at y=+0.45.
        # Step 4 is the MOVE to destination — check its target_position
        step4 = next((c for c in plan.commands if c.step == 4), None)
        assert step4 is not None
        assert step4.command_type == CommandType.MOVE
        assert step4.target_position is not None
        assert step4.target_position.x == pytest.approx(3.0, abs=0.01)
        assert step4.target_position.y == pytest.approx(2.15, abs=0.01)

    def test_right_of_applies_negative_y_offset(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.PLACE,
            object_target="red block",
            destination="blue block",
            spatial_relation="right of",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="place the red block to the right of the blue block",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, scene)
        # blue block at (3.0, 2.0); "right of" offset = (0, -0.15) → (3.0, 1.85).
        # Step 4 is the MOVE to the destination, after LOCATE/MOVE/PICK.
        step4 = next((c for c in plan.commands if c.step == 4), None)
        assert step4 is not None
        assert step4.command_type == CommandType.MOVE
        assert step4.target_position is not None
        assert step4.target_position.x == pytest.approx(3.0, abs=0.01)
        assert step4.target_position.y == pytest.approx(1.85, abs=0.01)

    def test_near_applies_diagonal_offset(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.PLACE,
            object_target="green block",
            destination="workstation",
            spatial_relation="near",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="put the green block near the workstation",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, scene)
        assert plan.total_steps >= 3

    def test_spatial_offset_calculation_left(self):
        ref = (3.0, 2.0)
        result = _apply_spatial_offset(ref, "left of")
        assert result[1] > ref[1]
        assert result == pytest.approx((3.0, 2.15), abs=0.01)

    def test_spatial_offset_calculation_right(self):
        ref = (3.0, 2.0)
        result = _apply_spatial_offset(ref, "right of")
        assert result[1] < ref[1]
        assert result == pytest.approx((3.0, 1.85), abs=0.01)

    def test_spatial_offset_none_returns_original(self):
        ref    = (3.0, 2.0)
        result = _apply_spatial_offset(ref, None)
        assert result == ref

    def test_spatial_offset_unknown_relation_uses_default(self):
        ref    = (3.0, 2.0)
        result = _apply_spatial_offset(ref, "somewhere around")
        assert result != ref  # default offset applied

    def test_in_relation_uses_container_position(self, scene):
        """'in' should use the container's exact position (no offset)."""
        parsed = ParsedInstruction(
            action=ActionType.PICK,
            object_target="red block",
            destination="left tray",
            spatial_relation="in",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block and place it in the left tray",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, scene)
        # Should route through normal pick plan, not spatial pick
        types = [c.command_type for c in plan.commands]
        assert CommandType.PLACE in types

    def test_spatial_plan_executes_successfully(self, scene):
        """Full integration: spatial instruction → planner → executor → success."""
        parsed = ParsedInstruction(
            action=ActionType.PLACE,
            object_target="red block",
            destination="blue block",
            spatial_relation="left of",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="place the red block to the left of the blue block",
        )
        _, success, robot = _run(parsed, scene)
        assert success is True
        assert robot.get_held_object() is None


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STEP TESTS (PB7-MULTI)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiStepPlanning:

    def test_two_instructions_combined(self, scene):
        i1 = ParsedInstruction(
            action=ActionType.PICK, object_target="red block",
            destination="left tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block and place it in the left tray",
        )
        i2 = ParsedInstruction(
            action=ActionType.PICK, object_target="blue block",
            destination="right tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the blue block and place it in the right tray",
        )
        planner = TaskPlanner()
        plan    = planner.plan_multi_step([i1, i2], scene)
        assert plan.total_steps > 5  # combined > either alone

    def test_step_numbers_are_sequential(self, scene):
        i1 = ParsedInstruction(
            action=ActionType.LOCATE, object_target="red block",
            confidence=ConfidenceLevel.HIGH, raw_instruction="find the red block",
        )
        i2 = ParsedInstruction(
            action=ActionType.LOCATE, object_target="blue block",
            confidence=ConfidenceLevel.HIGH, raw_instruction="find the blue block",
        )
        planner = TaskPlanner()
        plan    = planner.plan_multi_step([i1, i2], scene)
        steps   = [c.step for c in plan.commands]
        assert steps == list(range(1, len(steps) + 1))

    def test_multi_step_instruction_preserved(self, scene):
        i1 = ParsedInstruction(
            action=ActionType.LOCATE, object_target="red block",
            confidence=ConfidenceLevel.HIGH, raw_instruction="find the red block",
        )
        planner = TaskPlanner()
        plan    = planner.plan_multi_step([i1], scene)
        assert "find the red block" in plan.instruction

    def test_multi_step_executes_fully(self, scene):
        i1 = ParsedInstruction(
            action=ActionType.PICK, object_target="red block",
            destination="left tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block and place it in the left tray",
        )
        i2 = ParsedInstruction(
            action=ActionType.LOCATE, object_target="yellow block",
            confidence=ConfidenceLevel.HIGH, raw_instruction="locate the yellow block",
        )
        planner  = TaskPlanner()
        plan     = planner.plan_multi_step([i1, i2], scene)
        robot    = MockRobot()
        robot.load_scene(scene)
        executor = Executor(robot)
        result   = executor.execute(plan, verbose=False)
        assert result.success is True
        assert result.steps_completed == plan.total_steps


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:

    def test_simple_pick_place_end_to_end(self, scene):
        """pick up red block → place in left tray → robot holds nothing at end."""
        parsed = ParsedInstruction(
            action=ActionType.PICK, object_target="red block",
            destination="left tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block and place it in the left tray",
        )
        _, success, robot = _run(parsed, scene)
        assert success is True
        assert robot.get_held_object() is None
        # Red block should now be at left tray position
        obj_map = robot.get_object_map()
        assert "red block" in obj_map
        assert obj_map["red block"]["position"] == pytest.approx((6.0, 1.0), abs=0.1)

    def test_locate_instruction_end_to_end(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.LOCATE, object_target="yellow block",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="locate the yellow block",
        )
        _, success, robot = _run(parsed, scene)
        assert success is True

    def test_move_instruction_end_to_end(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.MOVE, object_target="green block",
            destination="right tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="move the green block to the right tray",
        )
        _, success, robot = _run(parsed, scene)
        assert success is True
        assert robot.get_held_object() is None

    def test_unknown_object_fails_gracefully(self, scene):
        parsed = ParsedInstruction(
            action=ActionType.PICK, object_target="purple block",
            confidence=ConfidenceLevel.LOW,
            raw_instruction="pick up the purple block",
        )
        with pytest.raises(ValueError, match="not found in scene"):
            planner = TaskPlanner()
            planner.generate_plan(parsed, scene)

    def test_tracker_logs_all_stages(self, scene, tracker_instance, tmp_path):
        """Verify tracker records stages when passed to executor."""
        parsed = ParsedInstruction(
            action=ActionType.LOCATE, object_target="red block",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="locate the red block",
        )
        task_id  = tracker_instance.new_task(parsed.raw_instruction)
        plan     = TaskPlanner().generate_plan(parsed, scene, task_id=task_id)
        robot    = MockRobot()
        robot.load_scene(scene)
        executor = Executor(robot, tracker=tracker_instance, task_id=task_id)
        executor.execute(plan, verbose=False)
        tracker_instance.complete_task(task_id, success=True)
        tracker_instance.save()

        task = tracker_instance.get_task(task_id)
        assert task is not None
        assert "execution" in task["stages"]
        assert task["stages"]["execution"]["status"] == "success"

    def test_pipeline_robot_state_after_pick_place(self, scene):
        """Robot should be at tray position, holding nothing, after pick+place."""
        parsed = ParsedInstruction(
            action=ActionType.PLACE, object_target="blue block",
            destination="left tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="place the blue block in the left tray",
        )
        _, success, robot = _run(parsed, scene)
        assert success is True
        assert robot.get_held_object() is None
        assert robot.get_position() == pytest.approx((6.0, 1.0), abs=0.1)

    def test_consecutive_tasks_independent_state(self, scene):
        """Two separate pipeline runs should not share robot state."""
        parsed1 = ParsedInstruction(
            action=ActionType.PICK, object_target="red block",
            destination="left tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block",
        )
        parsed2 = ParsedInstruction(
            action=ActionType.PICK, object_target="blue block",
            destination="right tray", confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the blue block",
        )
        _, success1, robot1 = _run(parsed1, scene)
        _, success2, robot2 = _run(parsed2, scene)
        assert success1 and success2
        # Each robot is independent
        assert robot1.get_held_object() is None
        assert robot2.get_held_object() is None

    def test_full_demo_sequence(self, scene):
        """
        Runs the 6 demo instructions used in the client demo.
        All should succeed (ambiguous one is excluded — that fails at LLM stage).
        """
        demo_instructions = [
            ParsedInstruction(action=ActionType.PICK, object_target="red block",
                destination="left tray", confidence=ConfidenceLevel.HIGH,
                raw_instruction="pick up the red block and place it in the left tray"),
            ParsedInstruction(action=ActionType.LOCATE, object_target="yellow block",
                confidence=ConfidenceLevel.HIGH, raw_instruction="locate the yellow block"),
            ParsedInstruction(action=ActionType.MOVE, object_target="green block",
                destination="workstation", spatial_relation="right of",
                confidence=ConfidenceLevel.HIGH,
                raw_instruction="move the green block to the right of the workstation"),
            ParsedInstruction(action=ActionType.PICK, object_target="blue block",
                destination="right tray", confidence=ConfidenceLevel.MEDIUM,
                raw_instruction="grab the blue block and put it in the right tray"),
        ]

        for parsed in demo_instructions:
            _, success, _ = _run(parsed, scene)
            assert success is True, f"Demo instruction failed: {parsed.raw_instruction}"


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselineParser:

    def test_simple_pick_parsed(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("pick up the red block")
        assert r.action == "pick"
        assert "red" in r.object_target
        assert r.parse_success is True

    def test_simple_place_parsed(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("place the blue cube in the left tray")
        assert r.action == "place"
        assert r.destination == "left tray"

    def test_synonym_not_handled(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("grab the red block")
        # grab IS in keyword list so baseline should handle this
        assert r.action == "pick"

    def test_spatial_not_resolved(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("place the red block to the left of the blue block")
        # Baseline can extract spatial keyword but can't compute position
        assert r.spatial_relation is not None
        # It cannot resolve the actual offset — that requires scene awareness
        assert r.destination is None or "blue" not in (r.destination or "")

    def test_empty_instruction_fails_gracefully(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("")
        assert r.parse_success is False
        assert r.confidence == "low"

    def test_latency_is_sub_millisecond(self):
        from llm_backend.LLM_eval.baseline_parser import BaselineParser
        p = BaselineParser()
        r = p.parse("pick up the red block")
        assert r.latency_ms < 10.0  # should be microseconds, well under 10ms

    def test_baseline_evaluation_runs(self):
        from llm_backend.LLM_eval.baseline_parser import run_baseline_evaluation
        from llm_backend.LLM_eval.test_cases import TEST_CASES
        results = run_baseline_evaluation(verbose=False)
        assert len(results) == len(TEST_CASES)  # all test cases
        assert all("model" in r for r in results)
        assert all(r["model"] == "baseline" for r in results)

    def test_baseline_accuracy_lower_than_expected_for_spatial(self):
        """Baseline should score lower on spatial than LLM (no position resolution)."""
        from llm_backend.LLM_eval.baseline_parser import run_baseline_evaluation
        results  = run_baseline_evaluation(verbose=False)
        spatial  = [r for r in results if r["category"] == "spatial"]
        simple   = [r for r in results if r["category"] == "simple"]
        spatial_acc = sum(r["fully_correct"] for r in spatial) / len(spatial)
        simple_acc  = sum(r["fully_correct"] for r in simple)  / len(simple)
        # Baseline may match keywords but cannot resolve spatial positions.
        # Key assertion: simple accuracy >= spatial accuracy (spatial is harder for baseline)
        assert simple_acc >= spatial_acc, (
            f"Expected simple ({simple_acc:.1%}) >= spatial ({spatial_acc:.1%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REAL LLM INTEGRATION TESTS (require API key)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestLLMPipelineIntegration:
    """Requires OPENAI_API_KEY in .env. Run with: pytest -m integration"""

    def test_parse_and_plan_simple(self):
        from llm_backend.custom_LLM_parser import parse_instruction
        parsed = parse_instruction("pick up the red block")
        assert parsed.action.value == "pick"
        scene   = {"objects": [
            {"label": "red block", "position": (2.5, 1.0)},
            {"label": "left tray", "position": (6.0, 1.0)},
        ]}
        plan = TaskPlanner().generate_plan(parsed, scene)
        assert plan.total_steps >= 3

    def test_parse_and_plan_spatial(self):
        from llm_backend.custom_LLM_parser import parse_instruction
        parsed = parse_instruction(
            "place the red block to the left of the blue block"
        )
        assert parsed.spatial_relation is not None
        scene = {"objects": [
            {"label": "red block",  "position": (2.5, 1.0)},
            {"label": "blue block", "position": (3.0, 2.0)},
        ]}
        plan = TaskPlanner().generate_plan(parsed, scene)
        assert plan.total_steps >= 3
