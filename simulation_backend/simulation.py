"""
simulation_backend/simulation.py
---------------------------------
Simulation orchestrator for the P54 pipeline.

Owns the full PyBullet session lifecycle:
    - connects PyBullet (DIRECT for pipeline, GUI for visual debug)
    - applies physics settings from scene_config.yaml
    - builds the workspace (table, floor, walls)
    - loads all scene objects via ObjectLoader → ObjectRegistry
    - creates Camera, GroundTruth, SceneBuilder
    - optionally creates a primary detector via get_detector()

Public interface:
    get_live_scene() -> dict
        Captures one camera frame, runs detector (if active) and ground
        truth fallback, prints a detection summary, and returns the
        planner-compatible scene dict:
        {"objects": [{"label": str, "position": (x, y)}, ...]}

    get_robot() -> RobotBase | MockRobot
        Returns the active robot instance ready for Executor — MockRobot,
        FrankaPanda, or KukaIIWA, selected by ROBOT_MODEL in .env.

    reset() -> None
        Resets all object positions to their scene_config.yaml defaults.

    disconnect() -> None
        Cleanly disconnects from PyBullet. Always call this on exit.

Usage (from main.py):
    from simulation_backend.simulation import Simulation

    sim = Simulation()
    scene = sim.get_live_scene()
    robot = sim.get_robot()
    robot.load_scene(scene)
    ...
    sim.reset()
    sim.disconnect()

Environment variables:
    VISION_DETECTOR=yolo        activate YOLOv8 detector
    VISION_DETECTOR=colour      activate colour HSV detector
    SIMULATION_MODE=DIRECT      DIRECT (headless) or GUI (visual window)
"""

import os
import logging
import yaml
import pybullet as p
import pybullet_data

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simulation_backend",
    "scene_config.yaml",
)

_CONFIG_PATH_ALT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene_config.yaml",
)


class Simulation:
    """
    Manages the full PyBullet simulation session for the pipeline.

    Phase 1: GroundTruth vision only, MockRobot execution.
    Phase 2: Optional primary detector (YOLO / colour) with ground truth fallback.
    Phase 3 will add real robot URDF loading and RobotBase subclass selection.
    """

    def __init__(
        self,
        config_path: str = None,
        mode: str = None,
    ) -> None:
        """
        Connect to PyBullet and build the full simulation scene.

        Args:
            config_path : Path to scene_config.yaml.
                          Defaults to simulation_backend/scene_config.yaml.
            mode        : "DIRECT" (headless, for pipeline) or "GUI" (window).
                          Overridden by SIMULATION_MODE env var if set.
                          Defaults to "DIRECT".
        """
        self._cfg          = self._load_config(config_path)
        self._client       = self._connect(mode)
        self._registry     = None
        self._loader       = None
        self._workspace    = None
        self._camera       = None
        self._ground_truth = None
        self._builder      = None
        self._detector     = None
        self._robot        = None

        self._build_scene()
        logger.info("Simulation initialised successfully.")

    # ── Public interface ───────────────────────────────────────────────────────

    def get_live_scene(self, verbose: bool = True) -> dict:
        """
        Capture the current state of the scene and return the planner dict.

        Detection priority per object:
            1. Primary detector (YOLO / colour) — if VISION_DETECTOR is set
            2. GroundTruth — always runs as fallback for any object the
               primary detector missed

        When verbose=True (default), prints a detection summary table showing
        every object, its source (yolo / ground_truth), confidence, and position.

        Args:
            verbose : Print detection summary to stdout. Default True.

        Returns:
            {
                "objects": [
                    {"label": "red block",  "position": [0.45, -0.20, 0.05]},
                    ...
                ]
            }
        """
        p.stepSimulation(physicsClientId=self._client)

        # Run primary detector (empty list if none configured)
        detector_results = []
        if self._detector is not None:
            try:
                frame = self._camera.capture()
                detector_results = self._detector.detect(frame)
            except Exception as e:
                logger.warning(f"Detector failed, falling back to ground truth: {e}")
                if verbose:
                    print(f"       ⚠  Detector error: {e}")
                    print(f"       ↳  Falling back to ground truth for all objects.")

        # Ground truth always runs — fills in anything the detector missed
        gt_results = self._ground_truth.get_all()

        # Build the rich scene dict
        rich_scene    = self._builder.build(
            detector_results=detector_results,
            segmentation_results=None,
            ground_truth_results=gt_results,
        )
        planner_scene = self._builder.to_planner_format(rich_scene)

        # ── Detection summary printout ─────────────────────────────────────
        if verbose:
            detector_name = self._detector.name if self._detector else "none"
            detected_by   = {d.label: d for d in detector_results}
            total         = len(rich_scene.get("detected_objects", []))

            print(f"       Detector        : {detector_name}")
            print(f"       Objects found   : {total}")
            print(f"       {'Label':<20} {'Source':<14} {'Conf':>6}  {'Position (x, y)'}")
            print(f"       {'─'*20} {'─'*14} {'─'*6}  {'─'*20}")

            for obj in rich_scene.get("detected_objects", []):
                label  = obj["label"]
                source = obj["source"]
                conf   = obj["confidence"]
                coords = obj["position"]["coordinates_3d"]
                x, y   = coords["x"], coords["y"]
                source_tag = f"[{source}]"
                print(f"       {label:<20} {source_tag:<14} {conf:>6.2f}  ({x:.3f}, {y:.3f})")

        return planner_scene

    def get_robot(self):
        """
        Return the active robot instance — MockRobot, FrankaPanda, or
        KukaIIWA — selected by ROBOT_MODEL in .env (see _load_robot()).
        """
        return self._robot

    def reset(self) -> None:
        """Reset all objects to their scene_config.yaml starting positions."""
        if self._loader is not None:
            self._loader.reset_positions()
            logger.info("Scene reset to initial positions.")
        if self._robot is not None:
            self._robot.reset()

    def disconnect(self) -> None:
        """Disconnect from PyBullet. Always call on exit."""
        try:
            p.disconnect(physicsClientId=self._client)
            logger.info("PyBullet disconnected.")
        except Exception as e:
            logger.warning(f"Disconnect error: {e}")

    @property
    def registry(self):
        return self._registry

    @property
    def client(self) -> int:
        return self._client

    @property
    def camera(self):
        return self._camera

    @property
    def detector(self):
        return self._detector

    @property
    def builder(self):
        return self._builder

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def robot(self):
        return self._robot

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _connect(self, mode: str = None) -> int:
        env_mode      = os.getenv("SIMULATION_MODE", "DIRECT").upper()
        resolved_mode = (mode or env_mode).upper()
        pybullet_mode = p.GUI if resolved_mode == "GUI" else p.DIRECT
        client        = p.connect(pybullet_mode)

        physics_cfg   = self._cfg.get("physics", {})
        gravity       = physics_cfg.get("gravity", [0.0, 0.0, -9.81])
        p.setGravity(*gravity, physicsClientId=client)

        timestep = physics_cfg.get("timestep", 0.00416)
        p.setTimeStep(timestep, physicsClientId=client)

        solver_iter = physics_cfg.get("solver_iterations", 150)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iter,
            physicsClientId=client,
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

        # Disable PyBullet's internal real-time stepping so all physics
        # is driven exclusively by explicit p.stepSimulation() calls.
        # This gives the robot controller full control over timing.
        p.setRealTimeSimulation(0, physicsClientId=client)

        logger.info(f"PyBullet connected — mode={resolved_mode} client={client}")
        return client

    def _build_scene(self) -> None:
        from simulation_backend.simulation_environment.workspace       import Workspace
        from simulation_backend.simulation_environment.object_loader   import ObjectLoader
        from simulation_backend.simulation_environment.object_registry import ObjectRegistry
        from simulation_backend.simulation_environment.scene_builder   import SceneBuilder
        from simulation_backend.vision.ground_truth                    import GroundTruth
        from simulation_backend.vision.camera                          import Camera
        from simulation_backend.vision.detection_base                  import get_detector
        ws_cfg  = self._cfg.get("workspace", {})
        obj_cfg = self._cfg.get("objects",   [])
        cam_cfg = self._cfg.get("camera",    {})

        self._registry  = ObjectRegistry()

        self._workspace = Workspace(self._client, ws_cfg)
        self._workspace.build()
        logger.info("Workspace built.")

        self._loader = ObjectLoader(self._client, obj_cfg, self._registry)
        self._loader.load_all()
        logger.info(f"Loaded {len(self._registry)} objects into registry.")

        self._camera       = Camera(physics_client=self._client, config=cam_cfg)
        self._ground_truth = GroundTruth(self._client, self._registry)
        self._builder      = SceneBuilder(self._registry)

        # Inject physics_client into config so detectors that need live
        # PyBullet positions (colour_detector) can call p.getBasePositionAndOrientation
        detector_cfg = dict(self._cfg)
        detector_cfg["_physics_client"] = self._client
        self._detector = get_detector(registry=self._registry, config=detector_cfg)
        if self._detector:
            logger.info(f"Primary detector active: {self._detector}")
        else:
            logger.info("No primary detector — using ground truth only.")

        self._robot = self._load_robot(self._registry)

    def _load_robot(self, registry):
        """
        Instantiate the robot controller selected by ROBOT_MODEL in .env.

        Options:
            mock   -> MockRobot (default, no URDF, no physics arm)
            franka -> FrankaPanda (loads panda.urdf from pybullet_data)
            kuka   -> KukaIIWA (loads kuka_iiwa/model.urdf from pybullet_data)
            ur5    -> UniversalRobotUR5 [future]

        Returns:
            Robot instance (MockRobot or RobotBase subclass).
        """
        from simulation_backend.mock_robot import MockRobot

        robot_model = os.getenv("ROBOT_MODEL", "mock").strip().lower()

        if robot_model == "franka":
            try:
                import pybullet_data
                from simulation_backend.robots.Franka_panda import FrankaPanda

                urdf_path = os.path.join(
                    pybullet_data.getDataPath(),
                    "franka_panda", "panda.urdf",
                )
                body_id = p.loadURDF(
                    urdf_path,
                    basePosition    = [0.0, 0.0, 0.0],
                    useFixedBase    = True,
                    physicsClientId = self._client,
                )
                robot = FrankaPanda(
                    physics_client = self._client,
                    body_id        = body_id,
                    registry       = registry,
                )
                robot.reset()
                logger.info(
                    f"Robot: FrankaPanda loaded "
                    f"(body_id={body_id}, base=[0.0, 0.0, 0.0])."
                )
                return robot

            except Exception as e:
                logger.error(
                    f"Failed to load FrankaPanda: {e}. "
                    f"Falling back to MockRobot."
                )
                return MockRobot()

        elif robot_model == "kuka":
            try:
                import pybullet_data
                from simulation_backend.robots.Kuka_IIWA import KukaIIWA

                urdf_path = os.path.join(
                    pybullet_data.getDataPath(),
                    "kuka_iiwa", "model.urdf",
                )
                body_id = p.loadURDF(
                    urdf_path,
                    basePosition=[0.0, 0.0, 0.0],
                    useFixedBase=True,
                    physicsClientId=self._client,
                )
                robot = KukaIIWA(
                    physics_client=self._client,
                    body_id=body_id,
                    registry=registry,
                )
                robot.reset()
                logger.info(
                    f"Robot: KukaIIWA loaded "
                    f"(body_id={body_id}, base=[0.0, 0.0, 0.0])."
                )
                return robot

            except Exception as e:
                logger.error(
                    f"Failed to load KukaIIWA: {e}. "
                    f"Falling back to MockRobot."
                )
                return MockRobot()

        elif robot_model == "ur5":
            logger.warning(
                f"ROBOT_MODEL={robot_model} is not yet implemented. "
                f"Falling back to MockRobot."
            )
            return MockRobot()

        else:
            robot = MockRobot()
            logger.info("Robot: MockRobot (ROBOT_MODEL=mock or unset).")
            return robot

    @staticmethod
    def _load_config(config_path: str = None) -> dict:
        if config_path and os.path.exists(config_path):
            path = config_path
        elif os.path.exists(_CONFIG_PATH):
            path = _CONFIG_PATH
        elif os.path.exists(_CONFIG_PATH_ALT):
            path = _CONFIG_PATH_ALT
        else:
            search_dirs = [
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_backend"),
                os.getcwd(),
                os.path.join(os.getcwd(), "simulation_backend"),
            ]
            path = None
            for d in search_dirs:
                candidate = os.path.join(d, "scene_config.yaml")
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(
                    "scene_config.yaml not found. "
                    "Run from the project root or pass config_path explicitly."
                )

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        logger.info(f"Config loaded from: {path}")
        return cfg
