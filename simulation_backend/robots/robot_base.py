"""
robot_base.py
-------------
Abstract base class for all robot arm controllers in the simulation pipeline.

Every concrete robot — Franka Panda, Kuka IIWA, or any future model (e.g.
UR5, not yet implemented) — must inherit from RobotBase and implement the
five abstract command methods.

Design goals:
    1. One interface.  Executor calls move_to(), move_to_object(), pick(),
       place(), and locate() without knowing which robot is active.
       Swap robots by instantiating a different subclass — zero changes
       to Executor, TaskPlanner, or main.py.

    2. One return type.  All commands return CommandResult (imported from
       mock_robot.py so the existing dataclass is reused across real and
       mock robots). Executor consumes CommandResult directly.

    3. Stateful scene model.  load_scene() must be called before any command.
       Subclasses store a local object map so they can resolve object names
       to 3-D positions without querying the vision module on every call.

    4. PyBullet-backed.  Real subclasses receive a physics_client ID and
       a robot body_id from PyBullet. They use p.setJointMotorControl2() /
       p.calculateInverseKinematics() to move the real simulated arm.
       MockRobot does NOT inherit from this class; it is the no-PyBullet
       stand-in used while real robots are being built.

    5. Gripper delegation.  Subclasses hold a GripperBase instance and
       delegate open/close to it inside pick() and place().

Hierarchy:
    RobotBase        (this file — abstract)
        └─ FrankaPanda       (robots/Franka_panda.py)
        └─ KukaIIWA          (robots/Kuka_IIWA.py)
        └─ UniversalRobotUR5 (robots/Universal_Robot_ur5.py)

Selecting a robot at runtime:
    Instantiate the desired subclass and pass it to Executor.
    In future, a factory function keyed on ROBOT_MODEL in .env
    can be added here following the same pattern as get_detector().

Usage (pipeline):
    from simulation_backend.robots.Franka_panda import FrankaPanda
    from simulation_backend.executor import Executor

    robot = FrankaPanda(physics_client=client, body_id=robot_id, config=cfg)
    robot.load_scene(scene)
    executor = Executor(robot)
    result = executor.execute(plan)

Usage (implementing a new robot):
    from simulation_backend.robots.robot_base import RobotBase
    from simulation_backend.mock_robot import CommandResult

    class MyRobot(RobotBase):
        model_name = "my_robot"

        def move_to(self, x, y, z=0.0):
            ...
            return CommandResult(success=True, command="move_to", message="...")

        def move_to_object(self, object_name):
            ...

        def pick(self, object_name):
            ...

        def place(self, location_name):
            ...

        def locate(self, object_name):
            ...
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from simulation_backend.mock_robot import CommandResult

logger = logging.getLogger(__name__)


class RobotBase(ABC):
    """
    Abstract base class for all real PyBullet robot arm controllers.

    Subclasses must:
        1. Set a class-level `model_name` string used in logs and repr.
           e.g. "franka_panda", "kuka_iiwa", "universal_robot_ur5"
        2. Implement the five abstract command methods:
               move_to(), move_to_object(), pick(), place(), locate()
        3. Call super().__init__(physics_client, body_id, config) from __init__.
        4. Implement reset() to return the arm to its home configuration.

    Subclasses may:
        - Override load_scene() if they need extra processing beyond
          populating self._object_map (e.g. caching 3-D goal positions).
        - Override get_position() / get_held_object() if tracking state
          differently (e.g. reading joint positions from PyBullet directly).
        - Add robot-specific methods (e.g. set_joint_positions(), home()).
        - Store a GripperBase instance and call it inside pick()/place().

    State maintained by this base class:
        _position     : (x, y, z) of the TCP in world coordinates (metres)
        _held_object  : label of the object currently grasped, or None
        _object_map   : {label.lower(): {"label": str, "position": (x,y,z), "held": bool}}
        _command_log  : list of logged command result dicts (for debugging)
    """

    #: Human-readable model identifier. Override in every subclass.
    model_name: str = "base_robot"

    def __init__(
        self,
        physics_client: int,
        body_id:        int,
        config:         dict = None,
    ) -> None:
        """
        Args:
            physics_client : PyBullet client ID from p.connect().
            body_id        : PyBullet body ID of the loaded robot URDF.
            config         : Optional robot-specific config dict
                             (e.g. from scene_config.yaml or .env overrides).
                             Subclasses read their own keys from this dict.
        """
        self._client   = physics_client
        self._body_id  = body_id
        self._cfg      = config or {}

        # Internal state — mirrored from MockRobot so Executor works identically
        self._position:    tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._held_object: Optional[str]              = None
        self._object_map:  dict                       = {}
        self._command_log: list                       = []

        logger.info(f"[{self.model_name}] Robot controller initialised "
                    f"(body_id={body_id}, client={physics_client}).")

    # ── Scene loading ──────────────────────────────────────────────────────────

    def load_scene(self, scene: dict) -> None:
        """
        Populate the robot's local object map from the vision scene dict.

        Called by main.py / Executor before the first command is issued.
        Subclasses may override to cache additional derived data (e.g.
        pre-computed IK solutions for every object position).

        Args:
            scene : Planner-compatible scene dict:
                    {
                        "objects": [
                            {"label": "red block",  "position": [x, y]   or (x, y, z)},
                            {"label": "left tray",  "position": [x, y]},
                        ]
                    }
                    Position may be 2-D (x, y) or 3-D (x, y, z).
                    If z is absent, it defaults to 0.0.
        """
        self._object_map = {}
        for obj in scene.get("objects", []):
            label    = obj.get("label") or obj.get("name", "unknown")
            position = obj.get("position", (0.0, 0.0, 0.0))

            if isinstance(position, (list, tuple)):
                if len(position) == 2:
                    pos3d = (float(position[0]), float(position[1]), 0.0)
                else:
                    pos3d = (float(position[0]), float(position[1]), float(position[2]))
            elif isinstance(position, dict):
                pos3d = (
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0)),
                )
            else:
                pos3d = (0.0, 0.0, 0.0)

            self._object_map[label.lower()] = {
                "label":    label,
                "position": pos3d,
                "held":     False,
            }

        logger.info(f"[{self.model_name}] Scene loaded: {len(self._object_map)} objects.")

    # ── Abstract robot commands (must implement) ───────────────────────────────

    @abstractmethod
    def move_to(self, x: float, y: float, z: float = 0.0) -> CommandResult:
        """
        Move the robot TCP to absolute world coordinates (x, y, z).

        Implementation guide:
            1. Validate (x, y, z) against workspace bounds.
               Return failure CommandResult if out of bounds.
            2. Compute inverse kinematics:
                   joint_positions = p.calculateInverseKinematics(
                       self._body_id, end_effector_link_index,
                       targetPosition=[x, y, z], physicsClientId=self._client
                   )
            3. Send joint targets:
                   for i, jp in enumerate(joint_positions):
                       p.setJointMotorControl2(
                           self._body_id, i, p.POSITION_CONTROL,
                           targetPosition=jp, physicsClientId=self._client
                       )
            4. Step simulation until convergence or timeout.
            5. Update self._position = (x, y, z).
            6. Return CommandResult(success=True/False, command="move_to", ...).

        Args:
            x, y, z : Target TCP position in world metres.
                      z defaults to 0.0 (table surface level).

        Returns:
            CommandResult with success, message, and latency_ms.
        """
        ...

    @abstractmethod
    def move_to_object(self, object_name: str) -> CommandResult:
        """
        Move the robot TCP to the position of a named scene object.

        Implementation guide:
            1. Look up position via self._find_object(object_name).
               Return failure if not found.
            2. Add an approach offset above the object (e.g. z += grasp_height).
            3. Call self.move_to(x, y, z + offset) or implement the IK directly.
            4. Return the CommandResult.

        Args:
            object_name : Case-insensitive label matching the scene object
                          (e.g. "red block", "left tray").

        Returns:
            CommandResult with success, message, and latency_ms.
        """
        ...

    @abstractmethod
    def pick(self, object_name: str) -> CommandResult:
        """
        Grasp the named object with the gripper.

        Implementation guide:
            1. Return failure if already holding an object (self._held_object).
            2. Look up the object via self._find_object(object_name).
            3. Move to the pre-grasp pose above the object.
            4. Open gripper:  self._gripper.open()  (if gripper attached).
            5. Descend to grasp pose.
            6. Close gripper: self._gripper.close().
            7. Optionally: create a PyBullet constraint to attach the object
               to the end-effector link so it moves with the arm:
                   p.createConstraint(
                       self._body_id, end_effector_link,
                       object_body_id, -1,
                       p.JOINT_FIXED, [0,0,0], [0,0,0.05], [0,0,0]
                   )
            8. Update state: obj["held"] = True, self._held_object = object_name.
            9. Return CommandResult.

        Args:
            object_name : Label of the object to grasp.

        Returns:
            CommandResult with success, message, and latency_ms.
        """
        ...

    @abstractmethod
    def place(self, location_name: str) -> CommandResult:
        """
        Release the currently held object at the named location.

        Implementation guide:
            1. Return failure if not holding anything (self._held_object is None).
            2. Look up destination position via self._find_object(location_name).
            3. Move to position above the destination.
            4. Descend to release height.
            5. Open gripper: self._gripper.open().
            6. Remove PyBullet constraint if one was created in pick().
            7. Retract arm to a safe height.
            8. Update state: held obj position → dest, self._held_object = None.
            9. Return CommandResult.

        Args:
            location_name : Label of the destination (e.g. "left tray").

        Returns:
            CommandResult with success, message, and latency_ms.
        """
        ...

    @abstractmethod
    def locate(self, object_name: str) -> CommandResult:
        """
        Confirm the object exists in the scene and return its position.

        This is a read-only operation — it must NOT move the robot arm.

        Implementation guide:
            1. Look up via self._find_object(object_name).
            2. Optionally: query the live PyBullet scene or vision module
               to verify the object is still at its expected position.
            3. Return CommandResult with the position in the state dict.

        Args:
            object_name : Label of the object to find.

        Returns:
            CommandResult(success=True, state={"position": (x, y, z)})
            or CommandResult(success=False) if the object is not in the scene.
        """
        ...

    # ── Optional hooks (may override) ─────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """
        Return the robot arm to its home/resting configuration.

        Called between tasks or at pipeline startup to establish a known state.
        Subclasses should:
            1. Reset all joints to their home positions using setJointMotorControl2.
            2. Open the gripper.
            3. Release any active PyBullet constraints.
            4. Clear self._held_object and reset self._position to home.

        This method is abstract (not optional) because a real robot absolutely
        must be able to home safely. MockRobot has its own reset() that clears
        internal state without any PyBullet calls.
        """
        ...

    # ── State inspection (same interface as MockRobot) ─────────────────────────

    def get_position(self) -> tuple[float, float, float]:
        """Return the current TCP position (x, y, z) in world metres."""
        return self._position

    def get_held_object(self) -> Optional[str]:
        """Return the label of the currently held object, or None."""
        return self._held_object

    def get_object_map(self) -> dict:
        """Return a copy of the local object map."""
        return dict(self._object_map)

    def get_command_log(self) -> list:
        """Return a copy of the command log."""
        return list(self._command_log)

    def print_state(self) -> None:
        """Pretty-print the current robot state to stdout."""
        print(f"\n{'─'*55}")
        print(f"  {self.__class__.__name__} State  (body_id={self._body_id})")
        print(f"{'─'*55}")
        print(f"  Position : {self._position}")
        print(f"  Holding  : {self._held_object or 'nothing'}")
        print(f"  Scene objects:")
        for name, obj in self._object_map.items():
            held_str = " [HELD]" if obj["held"] else ""
            print(f"    {obj['label']:<22} @ {obj['position']}{held_str}")
        print(f"{'─'*55}\n")

    # ── Shared helpers for subclasses ──────────────────────────────────────────

    def _find_object(self, name: str) -> Optional[dict]:
        """
        Case-insensitive partial-match object lookup in the local scene map.

        Tries exact match first, then partial containment in both directions.

        Args:
            name : Object label to search for (e.g. "red block", "red").

        Returns:
            Object dict {"label", "position", "held"} or None if not found.
        """
        name_lower = name.lower()

        # Exact match
        if name_lower in self._object_map:
            return self._object_map[name_lower]

        # Partial match — name is a substring of a known label
        for key, obj in self._object_map.items():
            if name_lower in key or key in name_lower:
                return obj

        return None

    def _get_state(self) -> dict:
        """Return a snapshot of the robot's current state dict."""
        return {
            "position":    self._position,
            "held_object": self._held_object,
        }

    def _log(self, result: CommandResult) -> None:
        """Append a CommandResult to the internal command log."""
        self._command_log.append({
            "command":    result.command,
            "success":    result.success,
            "message":    result.message,
            "latency_ms": result.latency_ms,
        })

    @staticmethod
    def _elapsed(start: float) -> float:
        """Return milliseconds elapsed since `start` (from time.perf_counter())."""
        return (time.perf_counter() - start) * 1000

    def _workspace_bounds(self) -> tuple[tuple, tuple]:
        """
        Return the robot's reachable workspace bounds as ((x_min, y_min, z_min),
        (x_max, y_max, z_max)) in world metres.

        Default returns generous bounds — subclasses should override with the
        actual kinematic reach of their specific robot model.

        Returns:
            ((x_min, y_min, z_min), (x_max, y_max, z_max))
        """
        return ((-2.0, -2.0, -0.5), (2.0, 2.0, 2.0))

    def _within_bounds(self, x: float, y: float, z: float = 0.0) -> bool:
        """
        Check whether a target position is within the robot's workspace bounds.

        Args:
            x, y, z : Target TCP position in world metres.

        Returns:
            True if the position is reachable, False otherwise.
        """
        (x_min, y_min, z_min), (x_max, y_max, z_max) = self._workspace_bounds()
        return (x_min <= x <= x_max and
                y_min <= y <= y_max and
                z_min <= z <= z_max)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"model='{self.model_name}', "
                f"body_id={self._body_id}, "
                f"client={self._client})")
