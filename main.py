"""
main.py
-------
Wires all modules together:
    User instruction
        → LLM parse          (llm_backend/custom_LLM_parser.py)
        → Vision lookup      (simulation_backend/vision/scene_representation.py)
        → Task plan          (task_planner/planner.py)
        → Execution          (simulation_backend/executor.py)
        → Feedback           (inline validation)

All stages are logged via tracker.py with a unique task_id.
LLM backend is controlled by LLM_BACKEND in .env — not a CLI flag.

Vision uses the real simulation camera/detector stack.  If VISION_DETECTOR is
unset, production defaults to VISION_DETECTOR=yolo.

Usage:
    # Single instruction (real vision)
    python main.py "pick up the red block and place it in the left tray"

    # Single instruction (explicit live simulation)
    USE_LIVE_SIMULATION=true python main.py "pick up the red block"

    # Interactive mode
    python main.py --interactive

    # Suppress output
    python main.py --quiet "locate the yellow block"
"""
import os
import sys
import argparse
import logging
import time
import logging
logger = logging.getLogger(__name__)

os.environ["PYDANTIC_DISABLE_PLUGINS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Module imports ─────────────────────────────────────────────────────────────
from llm_backend.custom_LLM_parser import parse_instruction, parse_multi_instruction
from llm_backend.schema            import (
    ParsedInstruction, MultiActionInstruction, ConfidenceLevel,
)
from llm_backend.tracker           import PipelineTracker
from task_planner.planner          import TaskPlanner
from simulation_backend.vision.scene_representation import get_current_scene
from simulation_backend.mock_robot import MockRobot
from simulation_backend.executor   import Executor
from simulation_backend.action_schema import plan_to_commands
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

SEP = "═" * 60

# ── Real vision defaults ──────────────────────────────────────────────────────
os.environ.setdefault("VISION_DETECTOR", "yolo")
_USE_LIVE = os.getenv("USE_LIVE_SIMULATION", "true").strip().lower() == "true"


def _get_scene_and_robot(sim=None, verbose: bool = True):
    """
    Return (scene_dict, robot_instance) from the real vision pipeline.

    If a Simulation instance is provided, reuse it so Stage 2 and execution
    share the same live workspace.  Otherwise get_current_scene() creates a
    temporary headless Simulation and returns its real camera/detector scene.

    Args:
        sim     : Simulation instance, or None to create a temporary real-vision scene.
        verbose : Whether to print the detection summary table.

    Returns:
        (scene dict, robot instance)
    """
    if sim is not None:
        scene = get_current_scene(verbose=verbose, sim=sim)
        robot = sim.get_robot()
    else:
        scene = get_current_scene(verbose=verbose)
        robot = MockRobot()
    return scene, robot


def _execution_timeout_for_robot(robot) -> float:
    """
    Return the per-command execution timeout for the active robot.

    MockRobot commands are near-instant, but real KUKA PyBullet pick/place
    sequences include multiple IK moves, physics stepping, and settling time.
    Keep the safety timeout, with an environment override for final tuning.
    """
    if getattr(robot, "model_name", "") == "kuka_iiwa" or type(robot).__name__ == "KukaIIWA":
        return float(os.getenv("KUKA_EXECUTION_TIMEOUT", "15.0"))
    return float(os.getenv("EXECUTOR_TIMEOUT_SECONDS", "5.0"))


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    instruction: str,
    verbose:     bool = True,
    tracker:     PipelineTracker | None = None,
    sim=None,
) -> dict:
    """
    Run the full pipeline for a single instruction.

    Args:
        instruction : Natural language task instruction.
        verbose     : Print progress to stdout.
        tracker     : PipelineTracker instance for cross-domain logging.
        sim         : Optional Simulation instance. Pass None to let the
                      vision adapter create a temporary real-vision scene.

    Returns:
        Result dict — keys: success, task_id, parsed, plan, execution.
    """
    if tracker is None:
        tracker = PipelineTracker()

    _backend = os.getenv("LLM_BACKEND", "openai")

    # ── Register task ──────────────────────────────────────────────────────────
    task_id = tracker.new_task(instruction, model=_backend)

    vision_label = "REAL"

    if verbose:
        print(f"\n{SEP}")
        print(f"  PIPELINE START")
        print(f"  Instruction : {instruction}")
        print(f"  Model       : {_backend}")
        print(f"  Vision      : {vision_label}")
        print(f"  Task ID     : {task_id}")
        print(SEP)

    result = {
        "success":    False,
        "task_id":    task_id,
        "parsed":     None,
        "parsed_set": None,
        "plan":       None,
        "execution":  None,
    }

    # ══ STAGE 1: LLM PARSE ════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [1/5] LLM Parse ({_backend})")

    try:
        t0        = time.perf_counter()
        # S5-3: one instruction may contain several sequential actions.
        # parse_multi_instruction() always returns a MultiActionInstruction —
        # a single-action command simply comes back with one action.
        parsed_set = parse_multi_instruction(instruction)
        lat        = (time.perf_counter() - t0) * 1000

        parsed = parsed_set.primary          # back-compat for single-action code
        result["parsed"]     = parsed
        result["parsed_set"] = parsed_set

        tracker.record(
            task_id, "llm_parse", status="success",
            payload={
                "is_multi_action": parsed_set.is_multi_action,
                "action_count":    parsed_set.action_count,
                "segments":        parsed_set.segments,
                "actions":         [a.model_dump(mode="json") for a in parsed_set.actions],
            },
            latency_ms=lat,
        )

        if verbose:
            if parsed_set.is_multi_action:
                print(f"       Multi-action: YES — {parsed_set.action_count} actions")
                for i, a in enumerate(parsed_set.actions, 1):
                    print(f"         {i}. {a.action.value:<7} "
                          f"object='{a.object_target}' "
                          f"dest='{a.destination or '—'}' "
                          f"spatial='{a.spatial_relation or '—'}' "
                          f"({a.confidence.value})")
            else:
                print(f"       Action      : {parsed.action.value}")
                print(f"       Object      : {parsed.object_target}")
                print(f"       Destination : {parsed.destination or '—'}")
                print(f"       Spatial     : {parsed.spatial_relation or '—'}")
                print(f"       Confidence  : {parsed.confidence.value}")
            print(f"       Latency     : {lat:.0f}ms")

        if parsed_set.confidence == ConfidenceLevel.LOW:
            if verbose:
                print(f"\n  ⚠  Low confidence — instruction may be ambiguous")
                print(f"     Notes: {parsed_set.notes or parsed.notes}")
            tracker.record(task_id, "feedback", status="retry",
                           payload={"reason": "low_confidence",
                                    "notes": parsed_set.notes or parsed.notes})
            result["success"] = False
            tracker.complete_task(task_id, success=False)
            return result

    except Exception as e:
        tracker.record(task_id, "llm_parse", status="failed", error=str(e))
        tracker.complete_task(task_id, success=False)
        if verbose:
            print(f"       ✗ LLM parse failed: {e}")
        return result

    # ══ STAGE 2: VISION LOOKUP ════════════════════════════════════════════════
    if verbose:
        print(f"\n  [2/5] Vision Lookup  [{vision_label}]")

    try:
        t0 = time.perf_counter()
        scene, robot = _get_scene_and_robot(sim, verbose=verbose)
        lat = (time.perf_counter() - t0) * 1000
        objects = scene.get("objects", [])

        if not objects:
            message = (
                "No objects detected in the current scene. "
                "Check the real vision simulation before planning."
            )
            print(f"       ⚠ {message}")

        tracker.record(
            task_id, "vision_lookup", status="success",
            payload={
                "object_count": len(objects),
                "objects": [o.get("label") for o in objects],
                "source": vision_label,
            },
            latency_ms=lat,
        )
        
        if verbose:
            print(f"       Objects in scene: {[o.get('label') for o in objects]}")
            print(f"       Latency         : {lat:.0f}ms")

        # Show detection bounding boxes — DIRECT + live simulation mode only.
        # GUI mode already has PyBullet's own 3D window; skip the popup there
        # so only one window appears instead of two.
        if _USE_LIVE and sim is not None and os.getenv("SIMULATION_MODE", "DIRECT").upper() != "GUI":
            _show_detection_window(sim)

    except FileNotFoundError as e:
        message = f"Scene file missing: {e}"
        tracker.record(task_id, "vision_lookup", status="failed", error=message)
        tracker.complete_task(task_id, success=False)
        tracker.save()
        if verbose:
            print(f"       ✗ Vision lookup failed: {message}")
        return result

    except Exception as e:
        tracker.record(task_id, "vision_lookup", status="failed", error=str(e))
        tracker.complete_task(task_id, success=False)
        tracker.save()
        if verbose:
            print(f"       ✗ Vision lookup failed: {e}")
        return result

    # ══ STAGE 3: TASK PLANNING ════════════════════════════════════════════════
    if verbose:
        print(f"\n  [3/5] Task Planning")

    try:
        planner = TaskPlanner()
        t0      = time.perf_counter()
        # S5-3: route multi-action instructions through plan_multi_step() so
        # every action is planned, in order, into one continuous ActionPlan.
        if parsed_set.is_multi_action:
            plan = planner.plan_multi_step(parsed_set.actions, scene, task_id=task_id)
        else:
            plan = planner.generate_plan(parsed, scene, task_id=task_id)
        lat     = (time.perf_counter() - t0) * 1000

        result["plan"] = plan

        tracker.record(
            task_id, "task_plan", status="success",
            payload={
                "steps":        plan.total_steps,
                "commands":     [c.command_type.value for c in plan.commands],
                "action_count": parsed_set.action_count,
                "multi_action": parsed_set.is_multi_action,
            },
            latency_ms=lat,
        )
        if verbose:
            if parsed_set.is_multi_action:
                print(f"       Actions planned : {parsed_set.action_count}")
            print(f"       Steps generated : {plan.total_steps}")
            for cmd in plan.commands:
                print(f"       {cmd.summary()}")

    except ValueError as e:
        tracker.record(task_id, "task_plan", status="failed", error=str(e))
        tracker.complete_task(task_id, success=False)
        if verbose:
            print(f"       ✗ Planning failed: {e}")
        return result

    # ══ STAGE 4: EXECUTION ════════════════════════════════════════════════════
    robot_label = type(robot).__name__
    if verbose:
        print(f"\n  [4/5] Execution  [{robot_label}]")

    try:
        robot.load_scene(scene)
        executor = Executor(
            robot,
            tracker=tracker,
            task_id=task_id,
            timeout_seconds=_execution_timeout_for_robot(robot),
        )
        exec_res = executor.execute(plan, verbose=verbose)

        result["execution"] = exec_res

        if not exec_res.success:
            tracker.complete_task(task_id, success=False)
            result["success"] = False
            return result

    except Exception as e:
        tracker.record(task_id, "execution", status="failed", error=str(e))
        tracker.complete_task(task_id, success=False)
        if verbose:
            print(f"       ✗ Execution error: {e}")
        return result

    # ══ STAGE 5: FEEDBACK ════════════════════════════════════════════════════
    if verbose:
        print(f"  [5/5] Feedback & Validation")

    tracker.record(
        task_id, "feedback", status="success",
        payload={
            "steps_completed": exec_res.steps_completed,
            "total_steps":     plan.total_steps,
            "latency_ms":      exec_res.total_latency_ms,
        },
    )
    tracker.complete_task(task_id, success=True)
    result["success"] = True

    if verbose:
        print(f"       ✓ Task completed — {exec_res.steps_completed}/{plan.total_steps} steps")
        print(f"\n{SEP}")
        print(f"  PIPELINE COMPLETE  ✓  Task ID: {task_id}")
        print(SEP)
        tracker.print_task(task_id)

    tracker.save()
    return result


# ── Interactive mode ───────────────────────────────────────────────────────────




def _show_detection_window(sim) -> None:
    """
    Capture one camera frame and display it with detection bounding boxes.

    Called at the end of Stage 2 (Vision Lookup) when:
        - USE_LIVE_SIMULATION=true
        - SIMULATION_MODE=DIRECT (skipped in GUI mode, which already has
          PyBullet's own 3D window — avoids showing two windows at once)

    What is shown:
        - Bounding boxes from the active primary detector (colour / YOLO)
          drawn by detector.draw_detections() — each detector uses its own
          colour scheme (cyan for colour, yellow for YOLO)
        - Ground truth labels for every registered object derived from the
          PyBullet segmentation mask — white dot + "label (x, y, z)"
        - A bottom info bar showing detector name and object count

    Behaviour:
        - Opens an 800x600 OpenCV window
        - Blocks until any key is pressed
        - Closes the window and returns — pipeline continues to Stage 3

    Args:
        sim : Simulation instance (provides camera, detector, registry)
    """
    import cv2
    import numpy as np
    import pybullet as p

    WINDOW = "Stage 2 — Detection  (press any key to continue)"

    try:
        camera   = sim.camera
        detector = sim.detector
        registry = sim.registry

        # ── Capture one frame ─────────────────────────────────────────────
        frame   = camera.capture()
        display = frame.bgr.copy()

        # ── Primary detector bounding boxes ───────────────────────────────
        detections = []
        if detector is not None:
            try:
                detections = detector.detect(frame)
                display    = detector.draw_detections(display, detections)
            except Exception as e:
                logger.debug(f"[detection window] Detector error: {e}")

        # ── Ground truth labels from segmentation mask ────────────────────
        for entry in registry.all_entries():
            mask = (frame.seg == entry.body_id)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            try:
                pos, _ = p.getBasePositionAndOrientation(
                    entry.body_id, physicsClientId=sim.client
                )
                lbl = f"{entry.label} ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            except Exception:
                lbl = entry.label
            cv2.circle(display, (cx, cy), 3, (200, 200, 200), -1)
            cv2.putText(
                display, lbl, (cx + 5, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

        # ── Bottom info bar ───────────────────────────────────────────────
        h, w      = display.shape[:2]
        bar_h     = 40
        overlay   = display.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

        det_name  = detector.name if detector else "ground_truth"
        det_count = len(detections)
        cv2.putText(
            display,
            f"Detector: {det_name}   Objects detected: {det_count}   "
            f"Press any key to continue...",
            (10, h - bar_h + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (0, 210, 255), 1, cv2.LINE_AA,
        )

        # ── Show and wait ─────────────────────────────────────────────────
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 800, 600)
        cv2.imshow(WINDOW, display)
        cv2.waitKey(0)
        cv2.destroyWindow(WINDOW)
        # Flush pending window-close messages so Windows doesn't flag the
        # app as "Not Responding" for a few seconds after the popup closes.
        for _ in range(4):
            cv2.waitKey(1)

    except Exception as e:
        logger.warning(f"[detection window] Could not display: {e}")


def _hold_simulation_open(sim) -> None:
    """
    Keep the PyBullet GUI window open after the pipeline completes.

    Steps physics at real-time rate so gravity / settling stays active
    in the 3D window. Exits when the user types Q + Enter in the terminal.
    No OpenCV window — the PyBullet 3D view is the only display.
    """
    import pybullet as p
    import time as _time
    import threading

    print(f"\n{'═'*60}")
    print(f"  Pipeline complete — PyBullet window open.")
    print(f"  Type Q + Enter in this terminal to quit.")
    print(f"{'═'*60}\n")

    quit_flag = threading.Event()

    def _wait_for_q():
        while not quit_flag.is_set():
            try:
                line = input().strip().lower()
                if line in ("q", "quit", "exit", ""):
                    quit_flag.set()
            except EOFError:
                quit_flag.set()
                break

    listener = threading.Thread(target=_wait_for_q, daemon=True)
    listener.start()

    try:
        while not quit_flag.is_set():
            p.stepSimulation(physicsClientId=sim.client)
            _time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("  Closing simulation.")





def run_interactive(sim=None) -> None:
    tracker  = PipelineTracker()
    _backend = os.getenv("LLM_BACKEND", "openai")
    vision_label = "REAL"

    print(f"\n{SEP}")
    print("  Multimodal LLM — Industrial Task Planning Pipeline")
    print(f"  Model: {_backend}  |  Vision: {vision_label}")
    print("  Type 'quit' to exit  |  'status' for summary  |  'reset' to reset scene")
    print(SEP + "\n")

    while True:
        try:
            instruction = input("Instruction: ").strip()
            if not instruction:
                continue
            if instruction.lower() in ("quit", "exit", "q"):
                tracker.print_summary()
                print("Goodbye!")
                break
            if instruction.lower() == "status":
                tracker.print_summary()
                continue
            if instruction.lower() == "reset" and sim is not None:
                sim.reset()
                print("  Scene reset to initial positions.\n")
                continue
            run_pipeline(instruction, verbose=True, tracker=tracker, sim=sim)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            tracker.print_summary()
            break
        except Exception as e:
            print(f"  ✗ Pipeline error: {e}")

robot_type = os.getenv("ROBOT_BACKEND", "mock").lower()

if robot_type == "ros":
    from simulation_backend.robots.ros_robot import ROSRobot
    robot = ROSRobot()
    logger.info("Using ROS2 robot")
elif robot_type == "real":
    from simulation_backend.robots.kuka_robot import KukaRobot
    robot = KukaRobot()
else:
    from simulation_backend.mock_robot import MockRobot
    robot = MockRobot()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Multimodal LLM Industrial Task Planning Pipeline"
    )
    ap.add_argument("instruction", nargs="?", help="Instruction to execute")
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    ap.add_argument("--quiet",       "-q", action="store_true", help="Suppress verbose output")
    ap.add_argument("--live",        "-l", action="store_true",
                    help="Force live simulation (overrides USE_LIVE_SIMULATION env var)")
    args = ap.parse_args()

    # --live flag overrides the env var
    if args.live:
        os.environ["USE_LIVE_SIMULATION"] = "true"
        _USE_LIVE = True  # noqa: F811

    # Start simulation if live mode is requested
    sim = None
    if _USE_LIVE and (args.interactive or args.instruction):
        try:
            from simulation_backend.simulation import Simulation
            sim = Simulation()
            print(f"  Simulation started — {len(sim.registry)} objects loaded.")
        except Exception as e:
            print(f"  ✗ Failed to start simulation: {e}")
            print("  Real vision will be retried during Stage 2; no static scene will be used.")
            sim = None

    try:
        if args.interactive:
            run_interactive(sim=sim)
        elif args.instruction:
            run_pipeline(args.instruction, verbose=not args.quiet, sim=sim)
            # After single-instruction pipeline, keep PyBullet open in GUI mode
            # so the user can inspect the result. Exit only when Q is pressed.
            if sim is not None and os.getenv("SIMULATION_MODE", "DIRECT").upper() == "GUI":
                _hold_simulation_open(sim)
        else:
            ap.print_help()
    finally:
        if sim is not None:
            sim.disconnect()
