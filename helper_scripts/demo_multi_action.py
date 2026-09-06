"""
demo_multi_action.py
--------------------
S5-3 evidence script — Multi-Action Command Support (Sprint 5).

Runs a set of multi-action instructions through the real pipeline stages
    LLM parse  ->  task plan  ->  execution (MockRobot)
and prints a per-instruction report suitable for attaching as sprint evidence.

MockRobot and a fixture scene are used so the script runs without PyBullet;
the parsing and planning stages are the real ones used by main.py.

Usage:
    python helper_scripts/demo_multi_action.py
    python helper_scripts/demo_multi_action.py > documentation/sprint5_multi_action_evidence.txt
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from llm_backend.custom_LLM_parser import parse_multi_instruction
from task_planner.planner import TaskPlanner
from simulation_backend.mock_robot import MockRobot
from simulation_backend.executor import Executor

SEP = "=" * 74

SCENE = {
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

INSTRUCTIONS = [
    ("MA01", "Move the green block to the left tray and then move the yellow block to the right tray."),
    ("MA02", "Locate the red block, then move the blue block to the right tray."),
    ("MA03", "Locate the yellow block then move the green block to the workstation."),
    ("MA04", "Move the red block to the left tray, then the blue block to the right tray, then locate the green block."),
    ("MA05", "Take the red block to the left tray then take the yellow block to the right tray and finally locate the blue block."),
    ("MA06", "Pick up the red block and place it in the left tray, then pick up the blue block and place it in the right tray."),
    ("MA07", "Move the red block to the left tray; move the yellow block to the workstation."),
    ("SA01", "Pick up the red block and place it in the left tray."),
    ("SA02", "Grab the blue block then drop it near the workstation."),
    ("NEG1", "Move the purple block to the left tray then move the green block to the right tray."),
    ("NEG2", "Pick up the red block, then place the blue block in the right tray."),
]


def run_case(case_id: str, instruction: str) -> bool:
    print(f"\n{SEP}\n  {case_id}  {instruction}\n{SEP}")

    planner = TaskPlanner()

    # -- Stage 1: LLM parse ----------------------------------------------------
    t0 = time.perf_counter()
    try:
        parsed_set = parse_multi_instruction(instruction)
    except Exception as e:
        print(f"  [1] PARSE FAILED — {e}")
        return False
    parse_ms = (time.perf_counter() - t0) * 1000

    print(f"  [1] Parse        : multi_action={parsed_set.is_multi_action} "
          f"actions={parsed_set.action_count}  ({parse_ms:.0f}ms)")
    for i, segment in enumerate(parsed_set.segments, 1):
        print(f"        segment {i} : \"{segment}\"")
    for i, a in enumerate(parsed_set.actions, 1):
        print(f"        action  {i} : {a.action.value:<7} "
              f"object='{a.object_target}' dest='{a.destination or '-'}' "
              f"spatial='{a.spatial_relation or '-'}' conf={a.confidence.value}")

    # -- Stage 3: Task planning ------------------------------------------------
    try:
        if parsed_set.is_multi_action:
            plan = planner.plan_multi_step(parsed_set.actions, SCENE)
        else:
            plan = planner.generate_plan(parsed_set.primary, SCENE)
    except ValueError as e:
        print(f"  [3] PLAN FAILED SAFELY — {e}")
        return False

    print(f"  [3] Plan         : {plan.total_steps} steps")
    for cmd in plan.commands:
        print(f"        {cmd.summary()}")

    # -- Stage 4: Execution ----------------------------------------------------
    robot = MockRobot()
    robot.load_scene(SCENE)
    result = Executor(robot).execute(plan, verbose=False)

    status = "SUCCESS" if result.success else "FAILED"
    print(f"  [4] Execution    : {status} — "
          f"{result.steps_completed}/{plan.total_steps} steps "
          f"({result.total_latency_ms:.0f}ms)")
    if not result.success:
        print(f"        failed step {result.failed_step}: {result.failed_reason}")

    return result.success


def main() -> None:
    print(f"{SEP}\n  S5-3 Multi-Action Command Support — evidence run")
    print(f"  LLM backend : {os.getenv('LLM_BACKEND', 'openai')}")
    print(f"  Robot       : MockRobot   Scene: {len(SCENE['objects'])} objects")
    print(SEP)

    passed = sum(run_case(cid, text) for cid, text in INSTRUCTIONS)

    print(f"\n{SEP}")
    print(f"  {passed}/{len(INSTRUCTIONS)} cases executed successfully "
          f"(NEG1/NEG2 are expected to fail safely)")
    print(SEP)


if __name__ == "__main__":
    main()
