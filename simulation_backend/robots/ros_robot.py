"""
ros_robot.py
------------
ROS2 robot implementation for Swinburne physical robots.

Publishes commands to ROS2 topics and subscribes to feedback.
Compatible with MoveIt2 action servers used by Swinburne's robot setup.

Not currently wired into the live pipeline: main.py reads ROBOT_BACKEND at
module level and imports ROSRobot when it is "ros", but that code path
builds an unused module-level `robot` and is never consulted by
run_pipeline() or Simulation._load_robot() (which selects the robot via
ROBOT_MODEL instead). Import and instantiate this class directly until
that wiring exists.
"""
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ROSRobot:
    """
    Robot implementation using ROS2.
    Sends commands to the robot via ROS2 topics/actions.
    """

    def __init__(self):
        self._ros_available = self._init_ros()
        self._held_object: Optional[str] = None
        self._position = (0.0, 0.0, 0.0)
        self._scene: dict = {}

    def _init_ros(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node
            rclpy.init()
            self._node = Node("p54_pipeline_node")
            logger.info("[ROSRobot] ROS2 node initialised")
            return True
        except ImportError:
            logger.warning("[ROSRobot] rclpy not available — commands will be logged only")
            return False
        except Exception as e:
            logger.warning(f"[ROSRobot] ROS2 init failed: {e}")
            return False

    def load_scene(self, scene: dict) -> None:
        self._scene = {
            obj["label"]: obj["position"]
            for obj in scene.get("objects", [])
        }
        logger.info(f"[ROSRobot] Scene loaded: {len(self._scene)} objects")

    def locate(self, object_name: str) -> dict:
        """Confirm object exists in scene via vision module."""
        name_lower = object_name.lower()
        for label, pos in self._scene.items():
            if name_lower in label.lower():
                logger.info(f"[ROSRobot] Located '{object_name}' at {pos}")
                return {"label": label, "position": pos}
        raise ValueError(f"Object '{object_name}' not found in scene.")

    def move_to(self, position: tuple) -> bool:
        """Send MoveIt2 move command to robot."""
        x, y = position[0], position[1]
        z = 0.3  # approach height

        if self._ros_available:
            self._publish_move_command(x, y, z)
        else:
            logger.info(f"[ROSRobot] MOVE to ({x:.3f}, {y:.3f}, {z:.3f}) [ROS not available — logged only]")

        self._position = (x, y, z)
        time.sleep(0.1)  # simulate command acknowledgment
        return True

    def pick(self, object_name: str) -> bool:
        """Send gripper close command."""
        if self._held_object:
            raise ValueError(f"Already holding '{self._held_object}'. Place it first.")

        if self._ros_available:
            self._publish_gripper_command("close")
        else:
            logger.info(f"[ROSRobot] PICK '{object_name}' [ROS not available — logged only]")

        self._held_object = object_name
        return True

    def place(self, object_name: str) -> bool:
        """Send gripper open command."""
        if self._ros_available:
            self._publish_gripper_command("open")
        else:
            logger.info(f"[ROSRobot] PLACE '{object_name}' [ROS not available — logged only]")

        self._held_object = None
        return True

    def reset(self) -> None:
        """Send robot to home position."""
        if self._ros_available:
            self._publish_move_command(0.0, 0.0, 0.5)
        self._held_object = None
        self._position = (0.0, 0.0, 0.0)

    # ── ROS2 publishers ────────────────────────────────────────────────────────

    def _publish_move_command(self, x: float, y: float, z: float) -> None:
        try:
            from geometry_msgs.msg import PoseStamped
            import rclpy

            publisher = self._node.create_publisher(
                PoseStamped, "/p54/target_pose", 10
            )
            msg = PoseStamped()
            msg.header.frame_id = "base_link"
            msg.header.stamp    = self._node.get_clock().now().to_msg()
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.w = 1.0

            publisher.publish(msg)
            logger.info(f"[ROSRobot] Published move to ({x:.3f}, {y:.3f}, {z:.3f})")

        except Exception as e:
            logger.error(f"[ROSRobot] Move publish failed: {e}")

    def _publish_gripper_command(self, state: str) -> None:
        try:
            from std_msgs.msg import String

            publisher = self._node.create_publisher(
                String, "/p54/gripper_command", 10
            )
            msg = String()
            msg.data = state  # "open" or "close"
            publisher.publish(msg)
            logger.info(f"[ROSRobot] Gripper command: {state}")

        except Exception as e:
            logger.error(f"[ROSRobot] Gripper publish failed: {e}")

    def get_state(self) -> dict:
        return {
            "position":    self._position,
            "held_object": self._held_object,
            "ros_active":  self._ros_available,
        }