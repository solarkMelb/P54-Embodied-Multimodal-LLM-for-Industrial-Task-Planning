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
        if sim is not None and os.getenv("SIMULATION_MODE", "DIRECT").upper() != "GUI":
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


# ── GUI presentation (S5-3 demo) ───────────────────────────────────────────────

def _style_gui(sim) -> None:
    """
    Turn the PyBullet debug window into something presentable.

    PyBullet's GUI mode opens its ExampleBrowser: side panels, a Params pane and
    three synthetic-camera preview boxes (RGB, depth, segmentation). They are
    useful while debugging the vision module and only clutter a demo of the
    task pipeline, so they are switched off here and the camera is framed on
    the workspace instead of the default far-away view.

    GUI mode only. Every call is guarded — a visualiser that refuses a setting
    must never take the pipeline down with it.
    """
    import pybullet as p

    def _try(fn):
        """Apply one visual setting; ignore it if this build refuses it."""
        try:
            fn()
        except Exception as e:
            logger.debug(f"[gui] setting skipped: {e}")

    try:
        c = sim.client

        # Panels and synthetic-camera previews off.
        for flag in (p.COV_ENABLE_GUI,
                     p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            _try(lambda f=flag: p.configureDebugVisualizer(f, 0, physicsClientId=c))

        # Depth cues: shadows are what stop the scene reading as a flat diagram.
        _try(lambda: p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=c))
        _try(lambda: p.configureDebugVisualizer(
            lightPosition=[2.6, -2.2, 3.4], physicsClientId=c))
        _try(lambda: p.configureDebugVisualizer(
            shadowMapResolution=4096, shadowMapWorldSize=4,
            shadowMapIntensity=0.75, physicsClientId=c))

        # ── Palette ───────────────────────────────────────────────────────
        # A dark studio set. The workspace objects are the data the viewer has
        # to read, so everything that is not an object — backdrop, floor, table,
        # walls — is pushed down in value until the coloured blocks are the
        # brightest thing in the frame.
        # Value ladder, darkest to lightest. Each element has to separate from
        # the one behind it, or the silhouette disappears — the table read as
        # invisible when it sat too close in value to the floor under it.
        #   backdrop  <  floor  <  table  <  trays  <  blocks
        BACKDROP   = [0.095, 0.105, 0.135]          # darkest: recedes completely
        FLOOR      = [0.185, 0.200, 0.235]          # dark, but reads as a surface
        TABLE_TOP  = [0.520, 0.395, 0.270, 1.0]     # warm wood, clearly above the floor
        TRAY_BODY  = [0.740, 0.760, 0.800, 1.0]     # cool light grey: separates by hue too
        WALL_GLASS = [0.55, 0.68, 0.85, 0.07]       # a hint of an edge, nothing more
        WORKSTN    = [0.30, 0.32, 0.37, 0.35]       # translucent: stops hiding the red block

        _try(lambda: p.configureDebugVisualizer(
            rgbBackground=BACKDROP, physicsClientId=c))

        # Shadows carry the depth, but on a dark set a heavy shadow turns to
        # mud — keep them soft.
        _try(lambda: p.configureDebugVisualizer(
            shadowMapResolution=4096, shadowMapWorldSize=4,
            shadowMapIntensity=0.45, physicsClientId=c))

        def _close(a, b, tol=0.06):
            return all(abs(x - y) <= tol for x, y in zip(a[:3], b[:3]))

        WOOD_CFG = (0.76, 0.60, 0.42)               # table colour in scene_config.yaml

        for body in range(p.getNumBodies(physicsClientId=c)):
            try:
                uid  = p.getBodyUniqueId(body, physicsClientId=c)
                name = p.getBodyInfo(uid, physicsClientId=c)[1].decode(errors="replace").lower()
                shapes = p.getVisualShapeData(uid, physicsClientId=c)
            except Exception:
                continue

            # plane.urdf ships a checkerboard texture that tiles into the
            # distance and reads as a flat diagram. Strip it, keep the floor.
            if "plane" in name or "floor" in name:
                _try(lambda i=uid: p.changeVisualShape(
                    i, -1, textureUniqueId=-1, rgbaColor=FLOOR + [1.0],
                    physicsClientId=c))
                continue

            for shape in shapes:
                rgba = shape[7]
                link = shape[1]
                if 0.3 <= rgba[3] < 0.95:            # the perimeter walls
                    _try(lambda i=uid, l=link: p.changeVisualShape(
                        i, l, rgbaColor=WALL_GLASS, physicsClientId=c))
                elif _close(rgba, WOOD_CFG):         # the table
                    _try(lambda i=uid, l=link: p.changeVisualShape(
                        i, l, rgbaColor=TABLE_TOP, physicsClientId=c))

        try:
            for entry in sim.registry.all_entries():
                low = entry.label.lower()
                # The workstation sits dead centre and hides the red block.
                if "workstation" in low:
                    _try(lambda i=entry.body_id: p.changeVisualShape(
                        i, -1, rgbaColor=WORKSTN, physicsClientId=c))
                # Trays are the drop-off targets, so they have to read against
                # the warm table — cool and light does both jobs at once.
                elif "tray" in low:
                    _try(lambda i=entry.body_id: p.changeVisualShape(
                        i, -1, rgbaColor=TRAY_BODY, physicsClientId=c))
        except Exception:
            pass

        p.resetDebugVisualizerCamera(
            cameraDistance=float(os.getenv("GUI_CAM_DISTANCE", "0.95")),
            cameraYaw=float(os.getenv("GUI_CAM_YAW", "84")),
            cameraPitch=float(os.getenv("GUI_CAM_PITCH", "-27")),
            cameraTargetPosition=[0.45, 0.0, 0.10],
            physicsClientId=c,
        )
    except Exception as e:
        logger.debug(f"[gui] Could not style the visualiser: {e}")


def _label_objects(sim, previous: list | None = None, focus: set | None = None) -> list:
    """
    Name the objects the viewer actually needs to read.

    Labelling all seven objects at once produces noise, not information: the
    labels collide, and nothing tells the eye which objects the instruction is
    about. Only the objects named in the instruction are labelled, and each
    label is tinted to its object so the association is immediate.

    Args:
        sim:      Simulation instance.
        previous: Label ids from an earlier call, replaced in place.
        focus:    Object labels to show. None shows every block and tray.

    Returns:
        The list of debug-text ids, to pass back on the next refresh.
    """
    import pybullet as p

    # Bright tints: on a dark set, a label has to out-value the surface it
    # floats over, and each one carries its object's hue so the eye pairs them
    # before it reads the word.
    TINT = {
        "red block":    [1.00, 0.46, 0.42],
        "blue block":   [0.48, 0.70, 1.00],
        "green block":  [0.42, 0.93, 0.53],
        "yellow block": [1.00, 0.84, 0.36],
    }
    TRAY = [0.74, 0.78, 0.84]

    ids: list = []
    previous = previous or []
    try:
        for entry in sim.registry.all_entries():
            label = entry.label
            low   = label.lower()
            if "workstation" in low:          # never named in an instruction
                continue
            if focus and low not in focus:
                continue

            pos, _ = p.getBasePositionAndOrientation(entry.body_id, physicsClientId=sim.client)
            if "tray" in low:
                dx, dy, dz, colour, size = 0.0, -0.17, 0.02, TRAY, 0.95
            else:
                dx, dy, dz, colour, size = 0.0, 0.0, 0.15, TINT.get(low, [0.90, 0.92, 0.95]), 1.05

            kwargs = dict(textColorRGB=colour, textSize=size, physicsClientId=sim.client)
            if len(ids) < len(previous):
                kwargs["replaceItemUniqueId"] = previous[len(ids)]
            ids.append(p.addUserDebugText(
                label, [pos[0] + dx, pos[1] + dy, pos[2] + dz], **kwargs))

        # Clear any labels left over from a longer previous set.
        for stale in previous[len(ids):]:
            try:
                p.removeUserDebugItem(stale, physicsClientId=sim.client)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[gui] Could not draw object labels: {e}")
    return ids


def _banner(sim, lines: list[tuple[str, float, list]], previous: list | None = None) -> list:
    """
    Draw the caption block above the workspace.

    Each entry is (text, size, colour) so the three lines carry a hierarchy —
    title, instruction, outcome — instead of reading as one flat block.
    """
    import pybullet as p

    ids: list = []
    previous = previous or []
    try:
        z = 0.52
        for i, (text, size, colour) in enumerate(lines):
            kwargs = dict(textColorRGB=colour, textSize=size, physicsClientId=sim.client)
            if i < len(previous):
                kwargs["replaceItemUniqueId"] = previous[i]
            ids.append(p.addUserDebugText(text, [0.02, 0.30, z], **kwargs))
            z -= 0.055 + 0.022 * size
    except Exception as e:
        logger.debug(f"[gui] Could not draw banner: {e}")
    return ids


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
            try:
                p.stepSimulation(physicsClientId=sim.client)
            except p.error:
                # The user closed the PyBullet window instead of typing Q.
                # The physics server is gone, so stop stepping and exit
                # cleanly rather than raising a traceback after a run that
                # already completed successfully.
                print("  PyBullet window closed.")
                break
            _time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Print the camera the user ended on, so a view found by dragging with
        # the mouse can be pinned in .env and reproduced on the next run.
        try:
            cam  = p.getDebugVisualizerCamera(physicsClientId=sim.client)
            yaw, pitch, dist = cam[8], cam[9], cam[10]
            print(f"\n  Camera position for this view — paste into .env to keep it:")
            print(f"    GUI_CAM_DISTANCE={dist:.2f}")
            print(f"    GUI_CAM_YAW={yaw:.0f}")
            print(f"    GUI_CAM_PITCH={pitch:.0f}\n")
        except Exception:
            pass
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
    args = ap.parse_args()

    sim = None
    if args.interactive or args.instruction:
        try:
            from simulation_backend.simulation import Simulation
            sim = Simulation()
            print(f"  Simulation started — {len(sim.registry)} objects loaded.")
            if os.getenv("SIMULATION_MODE", "DIRECT").upper() == "GUI":
                _style_gui(sim)
                sim._demo_labels = _label_objects(sim)
                sim._demo_banner = _banner(sim, [
                    ("P54   Multi-action command support", 1.45, [0.95, 0.96, 0.98]),
                    ("Natural language  ->  vision  ->  planner  ->  KUKA", 0.95, [0.60, 0.65, 0.73]),
                ])
        except Exception as e:
            print(f"  ✗ Failed to start simulation: {e}")
            print("  Real vision will be retried during Stage 2; no static scene will be used.")
            sim = None

    try:
        if args.interactive:
            run_interactive(sim=sim)
        elif args.instruction:
            gui = sim is not None and os.getenv("SIMULATION_MODE", "DIRECT").upper() == "GUI"
            if gui:
                shown = args.instruction if len(args.instruction) <= 58 \
                    else args.instruction[:55].rstrip() + "..."
                sim._demo_banner = _banner(sim, [
                    ("P54   Multi-action command support", 1.45, [0.95, 0.96, 0.98]),
                    (f'"{shown}"', 1.0, [0.66, 0.71, 0.78]),
                    ("running...", 1.05, [0.60, 0.65, 0.73]),
                ], getattr(sim, "_demo_banner", None))

            res = run_pipeline(args.instruction, verbose=not args.quiet, sim=sim)

            # After a single-instruction run, keep PyBullet open in GUI mode so
            # the result can be inspected and screenshotted. Refresh the labels
            # first so every block is named where it actually ended up.
            if gui:
                plan = res.get("plan")
                parsed_set = res.get("parsed_set")
                actions = getattr(parsed_set, "action_count", 1)
                steps = getattr(plan, "total_steps", 0)
                status = "COMPLETE" if res.get("success") else "FAILED"
                # Label only what the instruction actually touched, so the
                # final frame points at the result instead of naming everything.
                focus = set()
                for cmd in getattr(plan, "commands", []) or []:
                    if cmd.target_object:
                        focus.add(cmd.target_object.lower())
                sim._demo_labels = _label_objects(
                    sim, getattr(sim, "_demo_labels", None), focus=focus or None)
                sim._demo_banner = _banner(sim, [
                    ("P54   Multi-action command support", 1.45, [0.95, 0.96, 0.98]),
                    (f'"{shown}"', 1.0, [0.66, 0.71, 0.78]),
                    (f"{actions} actions   {steps} steps   {status}", 1.25,
                     [0.36, 0.94, 0.56] if res.get("success") else [1.00, 0.45, 0.40]),
                ], getattr(sim, "_demo_banner", None))
                _hold_simulation_open(sim)
        else:
            ap.print_help()
    finally:
        if sim is not None:
            sim.disconnect()
