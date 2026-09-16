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

For spatial-relation internals, multi-action command handling, design rationale, and sprint-by-sprint progress, see [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

---

## Team Member

| Name | Student ID | Role |
|---|---|---|
| Minh Hoang Duong | 104487115 | Team Member (Code Auditor) |
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
├── conftest.py                          ← Pytest configuration
├── pytest.ini                           ← Test markers
├── requirements.txt                     ← Dependencies
├── README.md
├── TECHNICAL_NOTES.md                   ← Design rationale, feature internals, sprint history
├── .env                                 ← API keys / config (never committed)
├── .env.example                         ← Template — copy to .env
│
├── Dockerfile, Dockerfile.lab           ← Container images (dev / Swinburne lab ROS2)
├── docker-compose.yml                   ← Pipeline + Ollama (dev)
├── docker-compose.gpu.yml               ← Opt-in NVIDIA GPU override
├── docker-compose.lab.yml               ← Lab ROS2 + physical-robot stack
├── .dockerignore
│
├── llm_backend/                         ← LLM instruction parser
│   ├── __init__.py
│   ├── custom_LLM_parser.py             ← parse_instruction() / parse_multi_instruction()
│   ├── multi_action.py                  ← Multi-action instruction splitter
│   ├── schema.py                        ← ParsedInstruction / MultiActionInstruction models
│   ├── prompts.py                       ← System prompt + few-shot examples
│   ├── edge_cases.py                    ← Empty/vague/synonym handling
│   ├── tracker.py                       ← Cross-domain pipeline task tracker
│   ├── cache.py                         ← Disk-based LLM response cache
│   └── backends/                        ← Per-model API implementations
│       ├── openai_backend.py            ← GPT-4o via OpenAI API
│       ├── gemini_backend.py            ← Gemini via Google API
│       ├── deepseek_backend.py          ← DeepSeek via OpenAI-compatible API
│       └── ollama_backend.py            ← Local Ollama models (no API key)
│
├── llm_backend/LLM_eval/                ← Multi-model evaluation
│   ├── comparison_report.py             ← Full evaluation report runner
│   ├── evaluator.py                     ← Runs models against test cases
│   ├── metrics.py                       ← 10 metrics per model per category
│   ├── test_cases.py                    ← 25 labelled test cases, 6 categories
│   ├── model_registry.py                ← Model loader for evaluation
│   ├── baseline_parser.py               ← Rule-based parser (no LLM) for comparison
│   └── eval_report.py                   ← End-to-end + baseline evaluation runner
│
├── task_planner/                        ← Task planning module
│   ├── __init__.py
│   └── planner.py                       ← Rule-based planner with spatial relations
│
├── simulation_backend/                  ← Execution + live vision module
│   ├── __init__.py
│   ├── action_schema.py                 ← RobotCommand, ActionPlan Pydantic schemas
│   ├── mock_robot.py                    ← MockRobot simulator (no PyBullet arm required)
│   ├── executor.py                      ← Runs ActionPlan step by step
│   ├── simulation.py                    ← Owns the PyBullet session; picks robot via ROBOT_MODEL
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
│   │   └── detection_weight/            ← Stock YOLO weights (fine-tuned weights live in fine_tuning/)
│   │
│   └── robots/                          ← Real robot implementations
│       ├── robot_base.py                ← Abstract RobotBase interface
│       ├── Franka_panda.py              ← ROBOT_MODEL=franka
│       ├── Kuka_IIWA.py                 ← ROBOT_MODEL=kuka
│       └── gripper/                     ← franka_hand.py, gripper_base.py
│
├── fine_tuning/                         ← YOLO fine-tuning artifacts (datasets, training runs, weights)
│
├── helper_scripts/                      ← Standalone utility scripts, run independently of the main pipeline
│
├── tests/                               ← Test suite (152 tests total)
│   ├── test_llm_module.py               ← 40 tests (28 unit + 12 integration)
│   ├── test_sprint2.py                  ← 38 unit tests
│   ├── test_multi_action.py             ← 42 tests (33 unit + 9 integration)
│   ├── integration_tests.py             ← 31 tests (29 unit + 2 integration)
│   └── test_real_vision_adapter.py      ← 1 unit test
│
└── documentation/                       ← Reports, evaluation artifacts, and design docs
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MinhWorkingAI/P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning.git
cd P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning
```

### 2. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and fill in the values for your chosen `LLM_BACKEND`:
```
# Controls which LLM the pipeline uses
LLM_BACKEND=ollama      # openai | gemini | deepseek | ollama

# OpenAI (GPT-4o)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# Google Gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite

# DeepSeek
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_MODEL=deepseek-chat

# Ollama — runs a local model, no API key. Requires the Ollama app installed
# and running (ollama.com), and the model pulled once: `ollama pull qwen2.5:7b`
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# Vision / simulation / robot (see .env.example for the full list)
SIMULATION_MODE=DIRECT         # DIRECT (headless) | GUI (visual debug window)
VISION_DETECTOR=yolo           # empty (ground truth only) | colour | yolo
ROBOT_MODEL=mock               # mock | franka | kuka  (ur5 not yet implemented)
```

Each team member uses their own `.env` with their own keys. The `.env` file is in `.gitignore` and is never committed.

---

## Running the Pipeline via Docker

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
docker compose run --rm p54 python main.py "pick up the red block and place it in the left tray"
```

**Notes**
- **Headless only.** PyBullet's GUI window needs a display the container doesn't have — `SIMULATION_MODE` is forced to `DIRECT` inside the container regardless of `.env`.
- **CPU by default.** For NVIDIA GPU passthrough (Windows/Linux with an NVIDIA GPU and Docker Desktop or native Docker), add the GPU override:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm p54 python main.py "..."
  ```
  Not available on Apple Silicon — there is no NVIDIA GPU to pass through.
- **Reproducibility.** The image builds for whichever platform runs it — no hardcoded architecture — and has been validated on both Windows/amd64 and macOS/Apple Silicon.
- `docker-compose.lab.yml` + `Dockerfile.lab` target the Swinburne lab's ROS2 + physical-robot setup and are not for general use.

---

## Running the Pipeline Natively

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Torch is pulled in by `ultralytics` (YOLO); if you need GPU support, install the CUDA-specific torch wheel *before* this step (see comments in `requirements.txt`).

### 3. Run
```bash
python main.py "pick up the red block and place it in the left tray"      # single instruction
python main.py --interactive                                              # interactive mode
python main.py --quiet "locate the yellow block"                          # minimal output
```
In interactive mode: type any instruction at the prompt, `status` for the tracker summary, `reset` to reset the scene, `quit` to exit.

### Switch model without changing code
Set `LLM_BACKEND` in `.env` (`openai | gemini | deepseek | ollama`), then run normally.

### Instruction categories
```bash
python main.py "pick up the red block and place it in the left tray"                                       # simple
python main.py "place the red block to the left of the blue block"                                         # spatial
python main.py "grab the yellow block and drop it in the right tray"                                       # synonym
python main.py "move the green block to the left tray and then move the yellow block to the right tray"    # multi-action
python main.py "put that thing over there"                                                                 # ambiguous — exits gracefully at Stage 1
python main.py "PICK UP THE RED BLOCK AND PLACE IT IN THE LEFT TRAY"                                        # edge case — all caps normalised
```

---

## Pipeline Stages

### Stage 1 — LLM Parse (`llm_backend/custom_LLM_parser.py`)
Sends the instruction to GPT-4o / Gemini / DeepSeek / a local Ollama model (selected via `LLM_BACKEND`) with a structured system prompt and few-shot examples. Returns `ParsedInstruction` with action, object, destination, spatial relation, and confidence. Handles empty, vague, and synonym edge cases before calling the model. Instructions with more than one action are split and parsed as a `MultiActionInstruction` — see `TECHNICAL_NOTES.md`.

### Stage 2 — Vision Lookup (`simulation_backend/vision/scene_representation.py`)
`get_current_scene()` captures the live PyBullet workspace through `simulation_backend/simulation.py`. Detection priority per object: primary detector (YOLO or colour threshold, if `VISION_DETECTOR` is set) first, then ground truth (exact PyBullet positions) as fallback for anything the detector missed. Fails fast with a `RuntimeError` if any object registered in the workspace is missing from the detected scene.

### Stage 3 — Task Planning (`task_planner/planner.py`)
Rule-based planner combining `ParsedInstruction` and the scene map into an ordered `ActionPlan`. Generates `locate → move → pick → move → place` sequences per action, and chains several actions into one plan for multi-action instructions. Spatial relations ("left of", "near", "behind", ...) resolve to offset positions relative to a reference object — see `TECHNICAL_NOTES.md` for the full offset table.

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
       Step 2: MOVE 'red block' → (0.45, -0.20)
       Step 3: PICK 'red block'
       Step 4: MOVE 'left tray' → (0.65, 0.45)
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

## Running Tests

### All unit tests (no API key or PyBullet required)
```bash
pytest tests/ -v -m "not integration"
```
Expected: **129 passed, 23 deselected**

### Integration-style tests that still don't need an API key
```bash
pytest tests/integration_tests.py -v -m "not integration"
```

### Full test suite including real LLM calls (requires API key)
```bash
pytest tests/ -v
```
152 tests total (129 unit + 23 marked `integration`), spread across `test_llm_module.py`, `test_sprint2.py`, `test_multi_action.py`, `integration_tests.py`, and `test_real_vision_adapter.py`.

### Single test class
```bash
pytest tests/integration_tests.py::TestSpatialRelationPlanning -v
pytest tests/test_sprint2.py::TestMockRobot -v
```

---

## Test Coverage

| File | Tests | Unit (no API) | Marked `integration` (needs API) |
|---|---|---|---|
| `tests/test_llm_module.py` | 40 | 28 | 12 |
| `tests/test_sprint2.py` | 38 | 38 | 0 |
| `tests/test_multi_action.py` | 42 | 33 | 9 |
| `tests/integration_tests.py` | 31 | 29 | 2 |
| `tests/test_real_vision_adapter.py` | 1 | 1 | 0 |
| **Total** | **152** | **129** | **23** |

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
