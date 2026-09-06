# Multimodal LLM for Industrial Task Planning

**COS40005 Computing Technology Project B — Capstone**  
Swinburne University of Technology × ARENA2036 × University of Stuttgart

---

## Overview

An intelligent simulation-based robotic system that interprets natural language instructions and executes corresponding tasks in a simulated industrial environment. The system integrates Large Language Models (LLMs), computer vision, and rule-based task planning into a modular five-stage pipeline.

```
User instruction
    → [1] LLM Parse        extract action, object, destination, spatial relation
    → [2] Vision Lookup    identify objects and positions in the workspace
    → [3] Task Planning    generate step-by-step robot action sequence
    → [4] Execution        send commands to robot (MockRobot / real simulation)
    → [5] Feedback         validate completion, log result, retry on failure
```

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Minh Hoang Duong | 104487115 | Team Member (Visualization) |
| Lakshit Bansal | 105028858 | Team Member (Vision Module) |
| Ved Jay Makhijani | 104762184 | Team Leader |
| Dinith Thejana | 105231766 | Team Member (Simulation Backend) |
| Kaveesha Dharmadasa | 105271678 | Team Member (Documentation / Scene Representation) |

**Supervisors:** Prof. Prem Prakash Jayaraman · Prof. Boris Eisenbart · Muhammad Saeed  
**Industry Partner:** ARENA2036 / University of Stuttgart

---

## Project Structure

```
P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning/
│
├── main.py                              ← Pipeline entry point
├── benchmark.py                         ← LLM parse latency benchmark
├── conftest.py                          ← Pytest configuration
├── pytest.ini                           ← Test markers
├── requirements.txt                     ← Dependencies
├── README.md
├── .env                                 ← API keys / config (never committed)
├── .env.example                         ← Template — copy to .env
│
├── llm_backend/                         ← LLM instruction parser
│   ├── __init__.py
│   ├── custom_LLM_parser.py             ← parse_instruction() / parse_multi_instruction()
│   ├── multi_action.py                  ← S5-3 multi-action instruction splitter
│   ├── schema.py                        ← ParsedInstruction / MultiActionInstruction models
│   ├── prompts.py                       ← System prompt + 6 few-shot examples
│   ├── edge_cases.py                    ← Empty/vague/synonym handling
│   ├── tracker.py                       ← Cross-domain pipeline task tracker
│   ├── hello_world.py                   ← API/backend connection test
│   └── backends/                        ← Per-model API implementations
│       ├── openai_backend.py            ← GPT-4o via OpenAI API
│       ├── gemini_backend.py            ← Gemini via Google API
│       ├── deepseek_backend.py          ← DeepSeek via OpenAI-compatible API
│       └── huggingface_backend.py       ← Local HuggingFace models (no API key)
│
├── llm_backend/LLM_eval/                ← Multi-model evaluation
│   ├── comparison_report.py             ← Full evaluation report runner
│   ├── evaluator.py                     ← Runs models against test cases
│   ├── metrics.py                       ← 10 metrics per model per category
│   ├── test_cases.py                    ← 25 labelled test cases, 6 categories
│   ├── model_registry.py                ← Model loader for evaluation
│   ├── baseline_parser.py               ← Rule-based parser (no LLM) for comparison
│   ├── eval_report.py                   ← End-to-end + baseline evaluation runner
│   ├── evaluation_metrics.csv           ← Generated — metrics output
│   └── evaluation_results.json          ← Generated — raw per-case results
│
├── task_planner/                        ← Task planning module
│   ├── __init__.py
│   └── planner.py                       ← Rule-based planner with spatial relations
│
├── simulation_backend/                  ← Execution + live vision module
│   ├── __init__.py
│   ├── action_schema.py                 ← RobotCommand, ActionPlan Pydantic schemas
│   ├── mock_robot.py                    ← MockRobot simulator (no PyBullet required)
│   ├── executor.py                      ← Runs ActionPlan step by step
│   ├── simulation.py                    ← Owns the PyBullet session; picks robot via ROBOT_MODEL
│   ├── display_scene.py                 ← Standalone live detection-window viewer
│   ├── scene_config.yaml                ← Workspace/object/robot layout config
│   ├── URDF_DOCUMENTATION.md            ← URDF asset authorship & licensing notes
│   ├── assets/block_urdf/               ← Custom table/tray/block/workstation URDFs
│   │
│   ├── simulation_environment/          ← PyBullet scene construction
│   │   ├── workspace.py                 ← Table/floor/walls
│   │   ├── object_loader.py             ← Loads objects from scene_config.yaml
│   │   ├── object_registry.py           ← Maps PyBullet body_ids to labels
│   │   └── scene_builder.py             ← Detector output → planner scene dict
│   │
│   ├── vision/                          ← Live vision stack
│   │   ├── scene_representation.py      ← get_current_scene() — Stage 2 entry point
│   │   ├── camera.py                    ← PyBullet camera capture
│   │   ├── detection_base.py            ← Abstract detector interface
│   │   ├── ground_truth.py              ← Exact-position fallback detector
│   │   ├── detection_implementation/    ← colour_detector.py, yolo_detector.py
│   │   └── detection_weight/            ← Cached YOLO weights (downloaded on first run)
│   │
│   └── robots/                          ← Real robot implementations
│       ├── robot_base.py                ← Abstract RobotBase interface
│       ├── Franka_panda.py              ← ROBOT_MODEL=franka
│       ├── Kuka_IIWA.py                 ← ROBOT_MODEL=kuka
│       └── gripper/                     ← franka_hand.py, gripper_base.py
│
├── fine_tuning/                         ← Model fine-tuning artifacts (datasets, training runs, weights)
│
├── helper_scripts/                      ← Standalone utility scripts, run independently of the main pipeline
│
├── tests/                               ← Test suite (145 tests total)
│   ├── test_llm_module.py               ← 40 tests (28 unit + 12 integration)
│   ├── test_2.py                        ← 35 tests (24 unit + 11 integration)
│   ├── test_sprint2.py                  ← 38 unit tests
│   ├── test_multi_action.py             ← 43 tests (34 unit + 9 integration) — S5-3
│   ├── integration_tests.py             ← 31 tests (29 unit + 2 integration)
│   └── test_real_vision_adapter.py      ← 1 unit test
│
└── documentation/                       ← Reports and docs
    ├── vision_framework_comparison.md
    ├── tool_recommendations.md
    ├── P54_Evaluation_Report.pdf
    └── simulation_backend_diagrams.xml
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MinhWorkingAI/P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning.git
cd P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Torch is pulled in by `transformers`; if you need GPU support, install the CUDA-specific torch wheel
*before* `pip install -r requirements.txt` (see comments in `requirements.txt`).

### 4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and fill in the values you need. Only the vars for your chosen `LLM_BACKEND` are required:
```
# Controls which LLM the pipeline uses
LLM_BACKEND=openai      # openai | gemini | deepseek | huggingface

# OpenAI (GPT-4o)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# Google Gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite

# DeepSeek
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_MODEL=deepseek-chat

# HuggingFace — runs a local model, no API key or internet needed after first download
HF_MODEL=Qwen/Qwen2.5-7B-Instruct

# Vision / simulation / robot (see .env.example for the full list)
USE_LIVE_SIMULATION=true       # false falls back to a static JSON scene in drafts/
SIMULATION_MODE=DIRECT         # DIRECT (headless) | GUI (visual debug window)
VISION_DETECTOR=yolo           # empty (ground truth only) | colour | yolo
ROBOT_MODEL=mock               # mock | franka | kuka  (ur5 not yet implemented)
```

Each team member uses their own `.env` with their own keys. The `.env` file is in `.gitignore` and is never committed.

---

## Running the Pipeline

### Single instruction
```bash
python main.py "pick up the red block and place it in the left tray"
```

### Interactive mode
```bash
python main.py --interactive
```
Type any instruction at the prompt. Type `status` to see the tracker summary. Type `reset` to reset the scene. Type `quit` to exit.

### Quiet mode (minimal output)
```bash
python main.py --quiet "locate the yellow block"
```

### Force live simulation
```bash
python main.py --live "pick up the red block"
```
Overrides `USE_LIVE_SIMULATION` for this run regardless of what's in `.env`.

### Switch model without changing code
Set `LLM_BACKEND` in your `.env`:
```
LLM_BACKEND=gemini
```
Then run normally — no code change needed.

### Test all 6 instruction categories
```bash
# Simple
python main.py "pick up the red block and place it in the left tray"

# Spatial
python main.py "place the red block to the left of the blue block"

# Synonym
python main.py "grab the yellow block and drop it in the right tray"

# Multi-step / spatial
python main.py "move the blue block near the workstation"

# Ambiguous — exits gracefully at Stage 1
python main.py "put that thing over there"

# Edge case — all caps normalised
python main.py "PICK UP THE RED BLOCK AND PLACE IT IN THE LEFT TRAY"
```

---

## Running Tests

### All unit tests (no API key or PyBullet required)
```bash
pytest tests/ -v -m "not integration"
```
Expected: **120 passed, 25 deselected**

### Integration-style tests that still don't need an API key
```bash
pytest tests/integration_tests.py -v -m "not integration"
```

### Full test suite including real LLM calls (requires API key)
```bash
pytest tests/ -v
```
145 tests total (120 unit + 25 marked `integration`, spread across `test_llm_module.py`, `test_2.py`, and `integration_tests.py`).

### Single test class
```bash
pytest tests/integration_tests.py::TestSpatialRelationPlanning -v
pytest tests/test_sprint2.py::TestMockRobot -v
```

---

## Running the Evaluation

### Baseline only (no API key needed — instant)
```bash
cd llm_backend/LLM_eval
python eval_report.py --baseline-only
```

### Full evaluation across all models (requires API keys)
```bash
cd llm_backend/LLM_eval
python eval_report.py
```

### Specific models only
```bash
python eval_report.py --models openai gemini
```

### Export results to CSV and JSON
```bash
python eval_report.py --export
```
Outputs: `evaluation_metrics.csv` and `evaluation_results.json`

### Multi-model comparison report
```bash
cd llm_backend/LLM_eval
python comparison_report.py
```

---

## Evaluation Categories

| Category | Cases | Description |
|---|---|---|
| simple | 5 | Basic single-action instructions |
| spatial | 5 | Positional relationships (left of, near, on top of) |
| synonym | 5 | Non-standard action words (grab, drop, find) |
| multi_step | 3 | Instructions implying two sequential actions |
| ambiguous | 3 | Vague or underspecified instructions |
| edge_case | 4 | Unknown objects, formatting variations, boundaries |

---

## Pipeline Stages

### Stage 1 — LLM Parse (`llm_backend/custom_LLM_parser.py`)
Sends the instruction to GPT-4o / Gemini / DeepSeek / a local HuggingFace model (selected via `LLM_BACKEND`) with a structured system prompt and 6 few-shot examples. Returns `ParsedInstruction` with action, object, destination, spatial relation, and confidence. Handles empty, vague, and synonym edge cases before calling the model.

### Stage 2 — Vision Lookup (`simulation_backend/vision/scene_representation.py`)
`get_current_scene()` captures the live PyBullet workspace through `simulation_backend/simulation.py`. Detection priority per object: primary detector (YOLO or colour threshold, if `VISION_DETECTOR` is set) first, then ground truth (exact PyBullet positions) as fallback for anything the detector missed. Fails fast with a `RuntimeError` if any object registered in the workspace is missing from the detected scene. If `USE_LIVE_SIMULATION=false`, falls back to a static JSON scene in `drafts/` instead.

### Stage 3 — Task Planning (`task_planner/planner.py`)
Rule-based planner combining `ParsedInstruction` and the scene map into an ordered `ActionPlan`. Generates `locate → move → pick → move → place` sequences. Spatial offset handling — "left of", "right of", "near", "next to", "on top of", "in front of", "behind" — computes offset positions relative to reference objects.

### Stage 4 — Execution (`simulation_backend/`)
`Executor` runs each `RobotCommand` sequentially, stops on first failure, and returns an `ExecutionResult`. The robot is selected by `ROBOT_MODEL` in `.env` — `mock` (default, no PyBullet arm), `franka`, or `kuka` — all implementing the same `RobotBase` interface, so switching robots is a config change, not a code change. `ur5` is not yet implemented and falls back to `MockRobot`.

### Stage 5 — Feedback (`llm_backend/tracker.py`)
Validates task completion, logs all 5 stages to `task_log.json` with a unique `task_id`. Triggers retry flag on failure or low confidence.

---

## Example Output

```
════════════════════════════════════════════════════════════
  PIPELINE START
  Instruction : pick up the red block and place it in the left tray
  Model       : openai
  Vision      : REAL
  Task ID     : d3c4f72a
════════════════════════════════════════════════════════════

  [1/5] LLM Parse (openai)
       Action      : pick
       Object      : red block
       Destination : left tray
       Spatial     : in
       Confidence  : high
       Latency     : 2926ms

  [2/5] Vision Lookup  [REAL]
       Objects in scene: ['red block', 'blue block', ..., 'left tray', 'right tray']

  [3/5] Task Planning
       Steps generated : 5
       Step 1: LOCATE 'red block'
       Step 2: MOVE 'red block' → (2.5, 1.0)
       Step 3: PICK 'red block'
       Step 4: MOVE 'left tray' → (6.0, 1.0)
       Step 5: PLACE 'left tray'

  [4/5] Execution  [MockRobot]
       ✓ Plan completed successfully in 0ms

  [5/5] Feedback & Validation
       ✓ Task completed — 5/5 steps

════════════════════════════════════════════════════════════
  PIPELINE COMPLETE  ✓  Task ID: d3c4f72a
════════════════════════════════════════════════════════════
```

---

## Spatial Relation Handling

The task planner supports positional instructions using offset-based spatial reasoning:

| Relation | Offset (dx, dy) | Example |
|---|---|---|
| left of | (−1.5, 0.0) | "place the red block to the left of the blue block" |
| right of | (+1.5, 0.0) | "move the green block to the right of the workstation" |
| near | (+0.8, +0.8) | "put the yellow block near the workstation" |
| next to | (+1.2, 0.0) | "place it next to the blue block" |
| on top of | (0.0, 0.0) | "stack the red block on top of the blue block" |
| in front of | (0.0, −1.5) | "move it in front of the workstation" |
| behind | (0.0, +1.5) | "place it behind the workstation" |
| in | (0.0, 0.0) | "place the block in the left tray" |

---

## Multi-Action Command Handling  *(S5-3, Sprint 5)*

A single instruction may contain several sequential actions. They are split,
parsed, planned and executed in the order they were written.

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

**Splitting is object-driven, not connector-driven.** A connector only starts a
new action if the segment after it names its own workspace object:

| Instruction | Actions | Why |
|---|---|---|
| "pick up the red block **and** place it in the left tray" | 1 | one pick-and-place |
| "grab the blue block **then** drop **it** near the workstation" | 1 | "it" refers back to the blue block |
| "move the green block to the left tray **and then** move the **yellow block** to the right tray" | 2 | a second object is named |
| "move the red block to the left tray, **then the blue block** to the right tray" | 2 | verb ellipsis — "move" is inherited |

Connectors recognised: `and then`, `then`, `after that`, `afterwards`,
`followed by`, `finally`, `;`, `, and`, and a comma directly before a verb.
Bare `next` is excluded on purpose — "next to the blue block" is a spatial
relation, not a sequence.

**State tracking between actions**

`plan_multi_step()` plans each action against a working copy of the scene that
is updated after every sub-plan, so "move the red block to the left tray then
move it to the right tray" targets the block where action 1 left it.

It also tracks the gripper. The robot has one gripper, so an action that picks
an object up and never puts it down blocks the next action. That is caught at
plan time with a clear reason rather than failing halfway through execution:

```
Action 2/2 ('place the blue block in the right tray') needs the gripper, but the
robot is still holding 'red block' from action 1. Give action 1 a destination,
or place 'red block' before this action.
```

**Evidence**

```bash
python helper_scripts/demo_multi_action.py        # 9 valid + 2 safe-fail cases
pytest tests/test_multi_action.py -v -m "not integration"   # 34 tests, no API key
pytest tests/test_multi_action.py -v                        # + 9 live-LLM tests
```

Recorded output: `documentation/sprint5_multi_action_evidence.txt`

---

## Test Coverage

| File | Tests | Unit (no API) | Marked `integration` (needs API) |
|---|---|---|---|
| `tests/test_llm_module.py` | 40 | 28 | 12 |
| `tests/test_2.py` | 35 | 24 | 11 |
| `tests/test_sprint2.py` | 38 | 38 | 0 |
| `tests/integration_tests.py` | 31 | 29 | 2 |
| `tests/test_real_vision_adapter.py` | 1 | 1 | 0 |
| **Total** | **145** | **120** | **25** |

---

## Key Design Decisions

**Rule-based task planner** — Deterministic, zero API cost, fully testable without external dependencies, and sufficient for the constrained pick-and-place simulation environment. An LLM-based planner can be substituted in future iterations.

**MockRobot** — Implements the same `RobotBase`-shaped interface as the real robots, with no PyBullet arm dependency, so tests and fast dev iterations don't need a real robot. Real robot execution (Franka, KUKA) is selected via `ROBOT_MODEL` in `.env` — the `Executor` code is identical either way.

**Pydantic schemas** — `ParsedInstruction`, `RobotCommand`, `ActionPlan` enforce strict interface contracts between modules. Validation errors surface at module boundaries rather than deep in pipeline logic.

**LLM_BACKEND in .env** — Model selection is a deployment-time decision. Each team member sets their own key and model. The codebase is model-agnostic.

**Baseline parser** — Rule-based keyword matching with no LLM provides a research comparison point. Demonstrates that LLMs provide 30–40pp accuracy gains on spatial, synonym, and ambiguous instruction categories.

---

## Project Progress

| Phase | Deliverables |
|---|---|
| Sprint 1 | LLM parser, schema, prompts, edge cases, multi-model evaluation framework |
| Sprint 2 | Task planner, mock robot, executor, action schema, 5-stage pipeline, tracker |
| Sprint 3 | Spatial relation handling, multi-step planning, baseline parser, full evaluation suite |
| Final | Live PyBullet vision (YOLO/colour + ground-truth fallback), Franka/KUKA real robot execution, end-to-end regression testing |

---

## Literature

- Ahn et al. — *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (SayCan)
- Driess et al. — *PaLM-E: An Embodied Multimodal Language Model*
- Radford et al. — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Yao et al. — *ReAct: Synergizing Reasoning and Acting in Language Models*
- Bode et al. — *A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics* (arXiv 2410.22997)

---

## Contact

| Supervisor | Email |
|---|---|
| Muhammad Saeed | msaeed@swin.edu.au |
| Prof. Boris Eisenbart | beisenbart@swin.edu.au |
| Prof. Prem Prakash Jayaraman | pjayaraman@swin.edu.au |
