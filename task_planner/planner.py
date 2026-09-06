"""
task_planner/planner.py
-----------------------
PB7 (Sprint 2) + PB7-SP + PB7-MULTI (Sprint 3)

Sprint 3 additions:
    - Spatial relation handling: "left of", "right of", "near", "on top of",
      "next to" → calculates offset position relative to reference object
    - Multi-step instruction handling: detects sequential two-action instructions
      and generates a compound plan (e.g. pick A then move B)

Usage:
    from task_planner.planner import TaskPlanner
    planner = TaskPlanner()
    plan    = planner.generate_plan(parsed_instruction, scene)
    plan.print_plan()
"""

import copy
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_backend.schema import ParsedInstruction, ActionType
from simulation_backend.action_schema import ActionPlan, RobotCommand, CommandType, Position

logger = logging.getLogger(__name__)

# ── Spatial offset map ─────────────────────────────────────────────────────────
# Maps spatial relation strings to (dx, dy) offsets in workspace units.
# Applied relative to the reference object's position.
SPATIAL_OFFSETS: dict[str, tuple[float, float]] = {
    "left of":    (-1.5,  0.0),
    "left":       (-1.5,  0.0),
    "right of":   ( 1.5,  0.0),
    "right":      ( 1.5,  0.0),
    "near":       ( 0.8,  0.8),
    "next to":    ( 1.2,  0.0),
    "on top of":  ( 0.0,  0.0),   # same x,y; height handled by real sim
    "in front of":( 0.0, -1.5),
    "behind":     ( 0.0,  1.5),
    "in":         ( 0.0,  0.0),   # inside container → use container position
}

DEFAULT_OFFSET = (1.0, 0.0)  # fallback when relation not in map


# ── Scene helpers ──────────────────────────────────────────────────────────────

def _find_in_scene(scene: dict, query: str) -> dict | None:
    """Case-insensitive partial match lookup in scene object list."""
    query_lower = query.lower()
    for obj in scene.get("objects", []):
        label = (obj.get("label") or obj.get("name", "")).lower()
        if query_lower in label or label in query_lower:
            pos = obj.get("position", (0.0, 0.0))
            if isinstance(pos, (list, tuple)):
                position = (float(pos[0]), float(pos[1]))
            elif isinstance(pos, dict):
                position = (float(pos.get("x", 0)), float(pos.get("y", 0)))
            else:
                position = (0.0, 0.0)
            return {"label": obj.get("label") or obj.get("name"), "position": position}
    return None


def _apply_spatial_offset(
    ref_position: tuple[float, float],
    spatial_relation: str | None,
) -> tuple[float, float]:
    """
    Calculate a target position by applying a spatial offset to a reference position.

    Args:
        ref_position:     (x, y) of the reference object
        spatial_relation: string like "left of", "near", "right of"

    Returns:
        (x, y) of the computed target position
    """
    if not spatial_relation:
        return ref_position

    relation_lower = spatial_relation.lower().strip()
    dx, dy = SPATIAL_OFFSETS.get(relation_lower, DEFAULT_OFFSET)
    return (ref_position[0] + dx, ref_position[1] + dy)


# ── Task planner ───────────────────────────────────────────────────────────────

class TaskPlanner:
    """
    Rule-based task planner — Sprint 2 + Sprint 3.

    Sprint 3 additions:
    - _resolve_destination(): uses spatial_relation to compute offset positions
    - _plan_multi_step(): handles compound two-action instructions
    - generate_plan(): detects multi-step and routes accordingly
    """

    def generate_plan(
        self,
        parsed: ParsedInstruction,
        scene: dict,
        task_id: str | None = None,
    ) -> ActionPlan:
        """
        Generate a step-by-step ActionPlan from instruction and scene.

        Handles:
        - Simple single-action instructions (pick, place, move, locate)
        - Spatial relation instructions ("left of", "near", "right of" etc.)
        - Multi-step instructions ("pick A then move B")

        Raises:
            ValueError: If required objects are not found in the scene
        """
        action = parsed.action.value
        logger.info(f"Planning: action={action}, object={parsed.object_target}, "
                    f"destination={parsed.destination}, spatial={parsed.spatial_relation}")

        # Detect multi-step: action=pick AND destination present AND spatial_relation
        # indicates a compound instruction (pick A and place it relative to B)
        if (action == ActionType.PICK.value and
                parsed.destination and
                parsed.spatial_relation and
                parsed.spatial_relation.lower() not in ("in",)):
            steps = self._plan_pick_with_spatial(parsed, scene)

        elif action == ActionType.PICK.value:
            steps = self._plan_pick(parsed, scene)

        elif action == ActionType.PLACE.value:
            steps = self._plan_place(parsed, scene)

        elif action == ActionType.MOVE.value:
            steps = self._plan_move(parsed, scene)

        elif action == ActionType.LOCATE.value:
            steps = self._plan_locate(parsed, scene)

        else:
            raise ValueError(f"Unknown action type: {action}")

        return ActionPlan(
            task_id=task_id,
            instruction=parsed.raw_instruction,
            commands=steps,
        )

    # ── Destination resolver ───────────────────────────────────────────────────

    def _resolve_destination(
        self,
        parsed: ParsedInstruction,
        scene: dict,
    ) -> tuple[str, Position]:
        """
        Resolve the destination position, applying spatial offset if present.

        Returns:
            (destination_label, Position) — the label and computed position
        """
        dest_name = parsed.destination or "right tray"
        dest = _find_in_scene(scene, dest_name)

        if dest is None:
            raise ValueError(
                f"Destination '{dest_name}' not found in scene. "
                f"Available: {[o.get('label') for o in scene.get('objects', [])]}"
            )

        raw_pos = dest["position"]

        if parsed.spatial_relation:
            computed = _apply_spatial_offset(raw_pos, parsed.spatial_relation)
            logger.info(f"Spatial offset applied: '{parsed.spatial_relation}' "
                        f"to {raw_pos} → {computed}")
            pos = Position(x=computed[0], y=computed[1])
            label = f"{dest_name} [{parsed.spatial_relation}]"
        else:
            pos   = Position(x=raw_pos[0], y=raw_pos[1])
            label = dest_name

        return label, pos

    # ── Planning methods ───────────────────────────────────────────────────────

    def _plan_pick(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """Pick only, or pick + place at named destination (no spatial offset)."""
        steps    = []
        step_n   = 1
        obj_name = parsed.object_target

        obj = _find_in_scene(scene, obj_name)
        if obj is None:
            raise ValueError(
                f"Object '{obj_name}' not found in scene. "
                f"Available: {[o.get('label') for o in scene.get('objects', [])]}"
            )

        obj_pos = Position(x=obj["position"][0], y=obj["position"][1])

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Confirm '{obj_name}' in scene at {obj['position']}",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=obj_name, target_position=obj_pos,
            description=f"Navigate arm to '{obj_name}'",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PICK,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        step_n += 1

        if parsed.destination:
            dest_label, dest_pos = self._resolve_destination(parsed, scene)
            steps.append(RobotCommand(
                step=step_n, command_type=CommandType.MOVE,
                target_object=parsed.destination, target_position=dest_pos,
                description=f"Navigate to '{dest_label}'",
            ))
            step_n += 1
            steps.append(RobotCommand(
                step=step_n, command_type=CommandType.PLACE,
                target_object=parsed.destination,
                description=f"Place '{obj_name}' at '{dest_label}'",
            ))

        return steps

    def _plan_pick_with_spatial(
        self, parsed: ParsedInstruction, scene: dict
    ) -> list[RobotCommand]:
        """
        PB7-SP: Pick + place with spatial offset destination.
        e.g. "place the red block to the left of the blue block"
        → moves to blue block position − spatial offset, not to a named tray.
        """
        steps    = []
        step_n   = 1
        obj_name = parsed.object_target

        obj = _find_in_scene(scene, obj_name)
        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_pos = Position(x=obj["position"][0], y=obj["position"][1])

        # Resolve offset destination
        dest_label, dest_pos = self._resolve_destination(parsed, scene)

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Confirm '{obj_name}' at {obj['position']}",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=obj_name, target_position=obj_pos,
            description=f"Navigate to '{obj_name}'",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PICK,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=parsed.destination, target_position=dest_pos,
            description=f"Navigate to computed position {dest_label} "
                        f"({dest_pos.x:.1f}, {dest_pos.y:.1f})",
        ))
        step_n += 1
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PLACE,
            target_object=parsed.destination,
            description=f"Place '{obj_name}' {parsed.spatial_relation} '{parsed.destination}'",
        ))

        return steps

    def _plan_place(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """Locate → move → pick → move to destination (with optional spatial offset) → place."""
        steps    = []
        step_n   = 1
        obj_name = parsed.object_target

        obj = _find_in_scene(scene, obj_name)
        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_pos            = Position(x=obj["position"][0], y=obj["position"][1])
        dest_label, dest_pos = self._resolve_destination(parsed, scene)

        steps.append(RobotCommand(step=step_n, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Confirm '{obj_name}' exists at {obj['position']}"))
        step_n += 1
        steps.append(RobotCommand(step=step_n, command_type=CommandType.MOVE,
            target_object=obj_name, target_position=obj_pos,
            description=f"Navigate to '{obj_name}'"))
        step_n += 1
        steps.append(RobotCommand(step=step_n, command_type=CommandType.PICK,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'"))
        step_n += 1
        steps.append(RobotCommand(step=step_n, command_type=CommandType.MOVE,
            target_object=parsed.destination or "destination", target_position=dest_pos,
            description=f"Navigate to '{dest_label}'"))
        step_n += 1
        steps.append(RobotCommand(step=step_n, command_type=CommandType.PLACE,
            target_object=parsed.destination or "destination",
            description=f"Release '{obj_name}' at '{dest_label}'"))

        return steps

    def _plan_move(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        return self._plan_place(parsed, scene)

    def _plan_locate(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        obj_name = parsed.object_target
        obj      = _find_in_scene(scene, obj_name)
        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")
        return [RobotCommand(
            step=1, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Find '{obj_name}' at {obj['position']}",
        )]

    def plan_multi_step(
        self,
        instructions: list[ParsedInstruction],
        scene: dict,
        task_id: str | None = None,
    ) -> ActionPlan:
        """
        PB7-MULTI: Generate a compound plan for multiple sequential instructions.

        Each instruction produces its own sub-plan. Commands are renumbered
        sequentially so the executor runs them as a single continuous plan.

        Args:
            instructions: List of ParsedInstruction objects (in order)
            scene:        Scene dict — shared across all sub-plans
            task_id:      Optional tracker task_id

        Returns:
            Single ActionPlan with all steps from all sub-plans combined

        Example:
            instructions = [
                ParsedInstruction(action=pick, object="red block", ...),
                ParsedInstruction(action=pick, object="blue block", destination="right tray", ...),
            ]
        """
        all_commands = []
        step_offset  = 0

        # S5-3: plan each action against a WORKING COPY of the scene that is
        # updated after every sub-plan. Without this, action 2 would still be
        # planned against action 1's starting positions, so an instruction like
        # "move the red block to the left tray then move it to the right tray"
        # would target a block that is no longer there.
        working_scene = copy.deepcopy(scene)

        # S5-3: the robot has ONE gripper. If an action picks something up and
        # never puts it down, the next action cannot pick anything else. Catch
        # that here, at plan time, with a clear reason — instead of letting the
        # executor fail halfway through a partially executed plan.
        held_object: str | None = None
        held_by_action: int = 0

        for i, parsed in enumerate(instructions):
            try:
                sub_plan = self.generate_plan(parsed, working_scene, task_id=None)
            except ValueError as e:
                # Fail safely with a clear reason naming the offending action,
                # instead of silently dropping it or executing a partial plan.
                raise ValueError(
                    f"Action {i + 1}/{len(instructions)} "
                    f"('{parsed.raw_instruction}') could not be planned: {e}"
                ) from e

            picks  = sum(1 for c in sub_plan.commands
                         if c.command_type == CommandType.PICK)
            places = sum(1 for c in sub_plan.commands
                         if c.command_type == CommandType.PLACE)

            if held_object and picks:
                raise ValueError(
                    f"Action {i + 1}/{len(instructions)} "
                    f"('{parsed.raw_instruction}') needs the gripper, but the "
                    f"robot is still holding '{held_object}' from action "
                    f"{held_by_action}. Give action {held_by_action} a "
                    f"destination, or place '{held_object}' before this action."
                )

            for cmd in sub_plan.commands:
                new_cmd = cmd.model_copy(
                    update={"step": cmd.step + step_offset}
                )
                all_commands.append(new_cmd)
            step_offset += len(sub_plan.commands)

            if picks > places:
                held_object    = parsed.object_target
                held_by_action = i + 1
            elif places:
                held_object    = None
                held_by_action = 0

            self._apply_plan_to_scene(working_scene, parsed, sub_plan)
            logger.info(f"Sub-plan {i+1}: {len(sub_plan.commands)} steps added")

        combined_instruction = " | ".join(p.raw_instruction for p in instructions)

        return ActionPlan(
            task_id=task_id,
            instruction=combined_instruction,
            commands=all_commands,
        )

    # -- Predicted scene state (S5-3) -------------------------------------------

    @staticmethod
    def _apply_plan_to_scene(scene: dict, parsed: ParsedInstruction, sub_plan) -> None:
        """
        Update a working scene dict to reflect where a sub-plan leaves its object.

        Called between sub-plans in plan_multi_step() so later actions are
        planned against the predicted workspace state rather than the original
        one. Only moves that actually end in a PLACE change anything; LOCATE and
        bare PICK leave the scene untouched.

        Args:
            scene:    Working scene dict — mutated in place.
            parsed:   The instruction this sub-plan came from.
            sub_plan: The ActionPlan generated for it.
        """
        from simulation_backend.action_schema import CommandType

        if not any(c.command_type == CommandType.PLACE for c in sub_plan.commands):
            return

        # The final MOVE before the PLACE carries the drop-off position.
        final_position = None
        for cmd in sub_plan.commands:
            if cmd.command_type == CommandType.MOVE and cmd.target_position is not None:
                final_position = cmd.target_position
        if final_position is None:
            return

        query = (parsed.object_target or "").lower()
        for obj in scene.get("objects", []):
            label = (obj.get("label") or obj.get("name") or "").lower()
            if query and (query in label or label in query):
                obj["position"] = (final_position.x, final_position.y)
                logger.info(
                    f"Predicted scene update: '{label}' -> "
                    f"({final_position.x:.2f}, {final_position.y:.2f})"
                )
                return
