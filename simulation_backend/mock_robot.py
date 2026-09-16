"""
mock_robot.py
-------------

Simulates robot behaviour without requiring PyBullet.
Maintains internal state (position, held object, object map) and responds
to commands by updating that state and returning success/failure responses.

Designed as a drop-in implementation of the same RobotBase interface used
by the real PyBullet robots (FrankaPanda, KukaIIWA in simulation_backend/robots/).
Selected via ROBOT_MODEL=mock in .env (the default) — no pipeline code
changes needed to switch between MockRobot and a real robot.

Usage:
    from simulation_backend.mock_robot import MockRobot

    robot = MockRobot()
    robot.load_scene(scene)

    result = robot.move_to(2.5, 1.0)
    result = robot.pick("red block")
    result = robot.place("left tray")

    robot.print_state()
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Response model ─────────────────────────────────────────────────────────────
@dataclass
class CommandResult:
    success:    bool
    command:    str
    message:    str
    latency_ms: float = 0.0
    state:      dict  = field(default_factory=dict)

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} [{self.command}] {self.message} ({self.latency_ms:.0f}ms)"


# ── Mock robot ─────────────────────────────────────────────────────────────────
class MockRobot:
    """
    Simulates a robot arm in a 2D industrial workspace.

    State:
        position    Current (x, y) position of the robot arm
        held_object Name of the object currently held (None if empty)
        object_map  Dict of object_name → {"position": (x,y), "held": bool}
        workspace   (width, height) of the workspace

    All commands return a CommandResult with success/failure and a message.
    """

    def __init__(
        self,
        workspace: tuple = (10.0, 10.0),
        move_speed: float = 1.0,
        simulate_latency: bool = True,
    ):
        self.workspace        = workspace
        self.move_speed       = move_speed
        self.simulate_latency = simulate_latency

        # Internal state
        self._position:    tuple          = (0.0, 0.0)
        self._held_object: Optional[str]  = None
        self._object_map:  dict           = {}
        self._command_log: list           = []

    # ── Scene loading ─────────────────────────────────────────────────────────

    def load_scene(self, scene: dict) -> None:
        """
        Load a scene representation into the robot's world model.

        Args:
            scene: Scene dict from vision module.
                   Expected format:
                   {
                       "objects": [
                           {"label": "red block",  "position": (2.5, 1.0)},
                           {"label": "left tray",  "position": (4.0, 0.5)},
                       ]
                   }
        """
        self._object_map = {}
        for obj in scene.get("objects", []):
            label    = obj.get("label") or obj.get("name", "unknown")
            position = obj.get("position", (0.0, 0.0))
            if isinstance(position, (list, tuple)):
                pos = (float(position[0]), float(position[1]))
            elif isinstance(position, dict):
                pos = (float(position.get("x", 0)), float(position.get("y", 0)))
            else:
                pos = (0.0, 0.0)

            self._object_map[label.lower()] = {
                "label":    label,
                "position": pos,
                "held":     False,
            }
        logger.info(f"Scene loaded: {len(self._object_map)} objects")

    # ── Robot commands ────────────────────────────────────────────────────────

    def move_to(self, x: float, y: float, z: float = 0.0) -> CommandResult:
        """
        Move robot arm to absolute position (x, y).

        z is accepted to keep the signature identical to RobotBase and the
        PyBullet robots, but MockRobot tracks the arm in 2D only, so it is
        ignored rather than stored.
        """
        start = time.perf_counter()

        # Boundary check
        if not (0 <= x <= self.workspace[0] and 0 <= y <= self.workspace[1]):
            return CommandResult(
                success=False,
                command="move_to",
                message=f"Position ({x}, {y}) is outside workspace bounds {self.workspace}",
                latency_ms=self._elapsed(start),
            )

        old_pos     = self._position
        self._position = (x, y)
        latency     = self._elapsed(start)

        result = CommandResult(
            success=True,
            command="move_to",
            message=f"Moved from {old_pos} to ({x}, {y})",
            latency_ms=latency,
            state=self._get_state(),
        )
        self._log(result)
        return result

    def move_to_object(self, object_name: str) -> CommandResult:
        """Move robot arm to the position of a named object."""
        start = time.perf_counter()
        obj   = self._find_object(object_name)

        if obj is None:
            return CommandResult(
                success=False,
                command="move_to_object",
                message=f"Object '{object_name}' not found in scene",
                latency_ms=self._elapsed(start),
            )

        pos = obj["position"]
        self._position = pos
        result = CommandResult(
            success=True,
            command="move_to_object",
            message=f"Moved to '{object_name}' at {pos}",
            latency_ms=self._elapsed(start),
            state=self._get_state(),
        )
        self._log(result)
        return result

    def pick(self, object_name: str) -> CommandResult:
        """Pick up a named object."""
        start = time.perf_counter()

        # Already holding something
        if self._held_object is not None:
            return CommandResult(
                success=False,
                command="pick",
                message=f"Already holding '{self._held_object}'. Place it first.",
                latency_ms=self._elapsed(start),
            )

        obj = self._find_object(object_name)
        if obj is None:
            return CommandResult(
                success=False,
                command="pick",
                message=f"Object '{object_name}' not found in scene",
                latency_ms=self._elapsed(start),
            )

        if obj["held"]:
            return CommandResult(
                success=False,
                command="pick",
                message=f"Object '{object_name}' is already being held",
                latency_ms=self._elapsed(start),
            )

        # Move to object first if not already there
        obj_pos = obj["position"]
        if self._position != obj_pos:
            self._position = obj_pos

        obj["held"] = True
        self._held_object = object_name

        result = CommandResult(
            success=True,
            command="pick",
            message=f"Picked up '{object_name}' at {obj_pos}",
            latency_ms=self._elapsed(start),
            state=self._get_state(),
        )
        self._log(result)
        return result

    def place(self, location_name: str) -> CommandResult:
        """Place the currently held object at a named location."""
        start = time.perf_counter()

        if self._held_object is None:
            return CommandResult(
                success=False,
                command="place",
                message="Not holding any object. Pick something first.",
                latency_ms=self._elapsed(start),
            )

        loc = self._find_object(location_name)
        if loc is None:
            return CommandResult(
                success=False,
                command="place",
                message=f"Location '{location_name}' not found in scene",
                latency_ms=self._elapsed(start),
            )

        loc_pos = loc["position"]
        placed_object = self._held_object

        # Update state
        if placed_object in self._object_map:
            self._object_map[placed_object.lower()]["position"] = loc_pos
            self._object_map[placed_object.lower()]["held"]     = False

        self._position    = loc_pos
        self._held_object = None

        result = CommandResult(
            success=True,
            command="place",
            message=f"Placed '{placed_object}' at '{location_name}' {loc_pos}",
            latency_ms=self._elapsed(start),
            state=self._get_state(),
        )
        self._log(result)
        return result

    def locate(self, object_name: str) -> CommandResult:
        """Find and return the position of a named object."""
        start = time.perf_counter()
        obj   = self._find_object(object_name)

        if obj is None:
            return CommandResult(
                success=False,
                command="locate",
                message=f"Object '{object_name}' not found in scene",
                latency_ms=self._elapsed(start),
            )

        result = CommandResult(
            success=True,
            command="locate",
            message=f"Found '{object_name}' at {obj['position']}",
            latency_ms=self._elapsed(start),
            state={"position": obj["position"]},
        )
        self._log(result)
        return result

    # ── State inspection ──────────────────────────────────────────────────────

    def get_position(self) -> tuple:
        return self._position

    def get_held_object(self) -> Optional[str]:
        return self._held_object

    def get_object_map(self) -> dict:
        return dict(self._object_map)

    def get_command_log(self) -> list:
        return list(self._command_log)

    def print_state(self) -> None:
        """Pretty print current robot state."""
        print(f"\n{'─'*50}")
        print(f"  MockRobot State")
        print(f"{'─'*50}")
        print(f"  Position:    {self._position}")
        print(f"  Holding:     {self._held_object or 'nothing'}")
        print(f"  Objects in scene:")
        for name, obj in self._object_map.items():
            held_str = " [HELD]" if obj["held"] else ""
            print(f"    {obj['label']:<20} @ {obj['position']}{held_str}")
        print(f"{'─'*50}\n")

    def reset(self) -> None:
        """Reset robot to initial state."""
        self._position    = (0.0, 0.0)
        self._held_object = None
        self._object_map  = {}
        self._command_log = []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_object(self, name: str) -> Optional[dict]:
        """Case-insensitive object lookup."""
        return self._object_map.get(name.lower())

    def _get_state(self) -> dict:
        return {
            "position":    self._position,
            "held_object": self._held_object,
        }

    def _log(self, result: CommandResult) -> None:
        self._command_log.append({
            "command":    result.command,
            "success":    result.success,
            "message":    result.message,
            "latency_ms": result.latency_ms,
        })

    @staticmethod
    def _elapsed(start: float) -> float:
        return (time.perf_counter() - start) * 1000