# Technical Notes

Deeper reference material that doesn't belong in the day-to-day setup guide: how spatial
relations and multi-action instructions are actually resolved, the reasoning behind the
main architectural choices, and the project's sprint-by-sprint progress. Start with
[`README.md`](README.md) for setup and usage — this file is for anyone extending or
reviewing the implementation.

---

## Spatial Relation Handling

The task planner (`task_planner/planner.py`) resolves positional instructions to offsets
in **metres**, applied relative to a reference object's position, matching the units used
throughout `simulation_backend/scene_config.yaml`.

Axis convention is the workspace's own: **+X** points away from the robot base, **+Y** is
the robot's left — this is why `left tray` sits at `y=+0.45` and `right tray` at
`y=-0.45` in the default scene. Left/right offsets therefore move along Y; front/behind
move along X. Clearance from the reference object's centre is `0.15 m` (blocks are `0.05 m`
across).

| Relation | Offset (dx, dy) | Example |
|---|---|---|
| left of | (0.0, +0.15) | "place the red block to the left of the blue block" |
| right of | (0.0, −0.15) | "move the green block to the right of the workstation" |
| near | (−0.10, +0.10) | "put the yellow block near the workstation" |
| next to | (0.0, +0.15) | "place it next to the blue block" |
| on top of | (0.0, 0.0) | "stack the red block on top of the blue block" |
| in front of | (−0.15, 0.0) | "move it in front of the workstation" |
| behind | (+0.15, 0.0) | "place it behind the workstation" |
| in / into / onto / at | (0.0, 0.0) | "place the block in the left tray" — uses the container's exact position |

Unrecognised relations fall back to `DEFAULT_OFFSET = (0.0, +0.15)`.

---

## Multi-Action Command Handling

A single instruction may contain several sequential actions. They are split, parsed,
planned and executed in the order they were written.

```bash
python main.py "move the green block to the left tray and then move the yellow block to the right tray"
```

```
[1/5] LLM Parse (openai)
      Multi-action: YES — 2 actions
        1. move    object='green block'  dest='left tray'   (high)
        2. move    object='yellow block' dest='right tray'  (high)
[3/5] Task Planning
      Actions planned : 2
      Steps generated : 10
```

**How it works**

| Stage | Module | Behaviour |
|---|---|---|
| Split | `llm_backend/multi_action.py` | `split_instruction()` breaks the sentence into ordered single-action segments. Deterministic — no API call. |
| Parse | `llm_backend/custom_LLM_parser.py` | `parse_multi_instruction()` parses each segment with the existing `parse_instruction()` (cache, retries, synonym mapping and edge cases all still apply) and returns a `MultiActionInstruction`. |
| Plan | `task_planner/planner.py` | `plan_multi_step()` builds one continuous `ActionPlan` with sequentially renumbered steps. |
| Execute | `simulation_backend/executor.py` | Unchanged — it already executes an `ActionPlan` step by step. |

**Splitting is object-driven, not connector-driven.** A connector only starts a new
action if the segment after it names its own workspace object:

| Instruction | Actions | Why |
|---|---|---|
| "pick up the red block **and** place it in the left tray" | 1 | one pick-and-place |
| "grab the blue block **then** drop **it** near the workstation" | 1 | "it" refers back to the blue block |
| "move the green block to the left tray **and then** move the **yellow block** to the right tray" | 2 | a second object is named |
| "move the red block to the left tray, **then the blue block** to the right tray" | 2 | verb ellipsis — "move" is inherited |

Connectors recognised: `and then`, `then`, `after that`, `afterwards`, `followed by`,
`finally`, `;`, `, and`, and a comma directly before a verb. Bare `next` is excluded on
purpose — "next to the blue block" is a spatial relation, not a sequence.

**State tracking between actions.** `plan_multi_step()` plans each action against a
working copy of the scene that is updated after every sub-plan, so "move the red block to
the left tray then move it to the right tray" targets the block where action 1 left it.

It also tracks the gripper. The robot has one gripper, so an action that picks an object
up and never puts it down blocks the next action. That is caught at plan time with a
clear reason rather than failing halfway through execution:

```
Action 2/2 ('place the blue block in the right tray') needs the gripper, but the
robot is still holding 'red block' from action 1. Give action 1 a destination,
or place 'red block' before this action.
```

**Evidence**

```bash
python helper_scripts/demo_multi_action.py                 # 9 valid + 2 safe-fail cases
pytest tests/test_multi_action.py -v -m "not integration"  # 33 tests, no API key
pytest tests/test_multi_action.py -v                       # + 9 live-LLM tests
```

Recorded output: `documentation/sprint5_multi_action_evidence.txt`

---

## Key Design Decisions

**Rule-based task planner** — Deterministic, zero API cost, fully testable without
external dependencies, and sufficient for the constrained pick-and-place simulation
environment. An LLM-based planner can be substituted in future iterations.

**MockRobot** — Implements the same `RobotBase`-shaped interface as the real robots, with
no PyBullet arm dependency, so tests and fast dev iterations don't need a real robot.
Real robot execution (Franka, KUKA) is selected via `ROBOT_MODEL` in `.env` — the
`Executor` code is identical either way.

**Pydantic schemas** — `ParsedInstruction`, `RobotCommand`, `ActionPlan` enforce strict
interface contracts between modules. Validation errors surface at module boundaries
rather than deep in pipeline logic.

**LLM_BACKEND in .env** — Model selection is a deployment-time decision. Each team member
sets their own key and model. The codebase is model-agnostic.

**Baseline parser** — Rule-based keyword matching with no LLM provides a research
comparison point. Demonstrates that LLMs provide 30–40pp accuracy gains on spatial,
synonym, and ambiguous instruction categories.

---

## Project Progress

| Phase | Deliverables |
|---|---|
| Sprint 1 | LLM parser, schema, prompts, edge cases, multi-model evaluation framework |
| Sprint 2 | Task planner, mock robot, executor, action schema, 5-stage pipeline, tracker |
| Sprint 3 | Spatial relation handling, multi-step planning, baseline parser, full evaluation suite |
| Sprint 4 | Live PyBullet vision (YOLO/colour + ground-truth fallback), Franka/KUKA real robot execution, end-to-end regression testing |
| Sprint 5 | Fine-tuned YOLOv8 integration, multi-action command support, Docker + local-LLM (Ollama) environment, cross-platform reproducibility validation |
