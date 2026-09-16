"""
detection_base.py
-----------------
Abstract base class for all object detectors in the simulation vision pipeline.

Every concrete detector — colour threshold, YOLO, or any future model (e.g.
SAM, open-vocabulary, depth-based — not yet implemented) — must inherit
from DetectorBase and implement the single abstract method: detect().

Design goals:
    1. One interface.  SceneBuilder calls detect() without
       knowing which backend is active. Swap detectors by changing VISION_DETECTOR
       in .env — zero code changes in the rest of the pipeline.

    2. One output type.  All detectors return list[Detection] (defined in
       scene_builder.py). SceneBuilder.build() consumes that list directly.

    3. Stateless detection calls.  detect() receives everything it needs in
       the call arguments. Detectors may hold model weights or thresholds as
       instance state, but must not mutate shared pipeline state.

    4. Graceful degradation.  If a detector cannot find an object it returns
       an empty list for that object — it does not raise. Downstream falls
       through to segmentation, then ground truth.

Hierarchy:
    DetectorBase  (this file — abstract)
        └─ ColourDetector     (detection_implementation/colour_detector.py)
        └─ YOLODetector       (detection_implementation/yolo_detector.py)  [future]
        └─ GroundedSAMDetector (detection_implementation/gsam_detector.py) [future]

Selecting a detector at runtime:
    Set VISION_DETECTOR in .env to the detector's registry key:
        VISION_DETECTOR=colour          → ColourDetector
        VISION_DETECTOR=yolo            → YOLODetector
        VISION_DETECTOR=grounded_sam    → GroundedSAMDetector
        (unset / empty)                 → no primary detector; ground truth used

    The factory function get_detector() (bottom of this file) reads
    VISION_DETECTOR and returns the appropriate instance, or None.

Usage (pipeline):
    from simulation_backend.vision.detection_base import DetectorBase, get_detector

    detector: DetectorBase | None = get_detector(registry)

    if detector:
        detections = detector.detect(frame)
    else:
        detections = []

Usage (implementing a new detector):
    from simulation_backend.vision.detection_base import DetectorBase
    from simulation_backend.simulation_environment.scene_builder import Detection

    class MyDetector(DetectorBase):
        name = "my_detector"

        def detect(self, frame: CameraFrame) -> list[Detection]:
            ...
            return [Detection(body_id=..., label=..., position_3d=..., ...)]
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

# Detection and CameraFrame imported here so subclasses only need one import
from simulation_backend.simulation_environment.scene_builder import Detection
from simulation_backend.vision.camera import CameraFrame
from simulation_backend.simulation_environment.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


# ── Abstract base ──────────────────────────────────────────────────────────────

class DetectorBase(ABC):
    """
    Abstract base class for all object detectors.

    Subclasses must:
        1. Set a class-level `name` string (used as the registry key and
           in log messages, e.g. "colour", "yolo", "grounded_sam").
        2. Implement detect(frame) → list[Detection].
        3. Call super().__init__(registry, config) from their __init__.

    Subclasses may:
        - Override warmup() to load model weights lazily on first use.
        - Override draw_detections() to customise the debug overlay style.
        - Add their own config keys under the detector's name in
          scene_config.yaml (passed in as the config dict).

    All detectors receive the ObjectRegistry so they can resolve body_ids
    for detected objects by label (required to populate Detection.body_id).
    """

    #: Unique key used by the factory and VISION_DETECTOR env var.
    #: Must be overridden in every subclass.
    name: str = "base"

    def __init__(
        self,
        registry: ObjectRegistry,
        config:   dict = None,
    ) -> None:
        """
        Args:
            registry : ObjectRegistry populated by ObjectLoader at startup.
                       Used to resolve labels → body_ids.
            config   : Detector-specific config dict from scene_config.yaml
                       (the section under the detector's name, or empty dict).
        """
        self._registry = registry
        self._cfg      = config or {}
        self._warmed_up = False
        logger.info(f"[{self.name}] Detector initialised.")

    # ── Abstract interface (must implement) ────────────────────────────────────

    @abstractmethod
    def detect(self, frame: CameraFrame) -> list[Detection]:
        """
        Run object detection on one captured frame.

        This is the single mandatory method every detector must implement.
        It must be fast enough to run at the camera's capture rate without
        blocking the main simulation loop. For heavy models (YOLO, SAM),
        consider running this in a background thread and returning the most
        recent cached result.

        Contract:
            - Must return a list (empty if nothing detected — never raise).
            - Each Detection.body_id must be a valid PyBullet body_id
              resolvable via the ObjectRegistry; use _resolve_body_id().
            - Each Detection.source must equal self.name.
            - Confidence should reflect the detector's actual confidence
              (0.0–1.0). Use 1.0 only for ground truth, not heuristics.

        Args:
            frame : CameraFrame from Camera.capture() containing .bgr,
                    .depth (metres), and .seg (PyBullet body_id mask).

        Returns:
            list[Detection] — may be empty if no objects found.
        """
        ...

    # ── Optional hooks (may override) ─────────────────────────────────────────

    def warmup(self) -> None:
        """
        Load model weights, compile TensorRT plans, or do any other
        first-call expensive initialisation.

        Called automatically by detect() on the first frame if not already
        done. Override to add detector-specific warmup logic.

        Example (YOLO):
            def warmup(self) -> None:
                self._model = YOLO(self._cfg.get("weights", "yolov8n.pt"))
                logger.info(f"[{self.name}] YOLO model loaded.")
        """
        self._warmed_up = True

    def draw_detections(
        self,
        frame:      np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Overlay detection bounding boxes and labels on a BGR frame.

        Default implementation draws green rectangles with white label text.
        Override to add confidence bars, colour-coded boxes, etc.

        Args:
            frame      : (H, W, 3) uint8 BGR image to draw on (not modified in place)
            detections : list[Detection] to visualise

        Returns:
            (H, W, 3) uint8 BGR image with overlaid annotations
        """
        canvas = frame.copy()
        for det in detections:
            bb = det.bounding_box_2d
            if bb is None:
                continue

            x1, y1 = int(bb["x_min"]), int(bb["y_min"])
            x2, y2 = int(bb["x_max"]), int(bb["y_max"])

            # Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 220, 0), thickness=2)

            # Label + confidence
            label_text = f"{det.label}  {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 220, 0), -1)
            cv2.putText(
                canvas, label_text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return canvas

    # ── Shared helpers for subclasses ──────────────────────────────────────────

    def _resolve_body_id(self, label: str) -> Optional[int]:
        """
        Look up the PyBullet body_id for a detected label.

        Uses ObjectRegistry.get_by_label() which supports partial,
        case-insensitive matching (e.g. "red" matches "red block").

        Args:
            label : Human-readable object label string

        Returns:
            Integer body_id if found, or None if the label is unknown.
        """
        entry = self._registry.get_by_label(label)
        if entry is None:
            logger.debug(f"[{self.name}] Unknown label — cannot resolve body_id: '{label}'")
            return None
        return entry.body_id

    def _depth_at_bbox_centre(
        self,
        frame: CameraFrame,
        bbox:  dict,
        sample_radius: int = 3,
    ) -> float:
        """
        Estimate the metric depth at the centre of a bounding box.

        Samples a small patch around the box centre and returns the
        median to reduce noise from specular highlights or depth holes.

        Args:
            frame         : CameraFrame containing the metric depth map.
            bbox          : {"x_min", "y_min", "x_max", "y_max"} pixel coords.
            sample_radius : Half-size of the sampling patch (pixels).

        Returns:
            Median depth in metres at the box centre, clamped to [near, far].
        """
        cx = int((bbox["x_min"] + bbox["x_max"]) / 2)
        cy = int((bbox["y_min"] + bbox["y_max"]) / 2)
        r  = max(1, sample_radius)

        h, w = frame.depth.shape
        y1 = max(0, cy - r);  y2 = min(h, cy + r + 1)
        x1 = max(0, cx - r);  x2 = min(w, cx + r + 1)

        patch = frame.depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return float(frame.depth[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0.0
        return float(np.median(valid))

    def _ensure_warmed_up(self) -> None:
        """Call warmup() exactly once before the first detect() call."""
        if not self._warmed_up:
            logger.info(f"[{self.name}] Warming up...")
            self.warmup()
            self._warmed_up = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ── Detector registry & factory ───────────────────────────────────────────────

#: Maps VISION_DETECTOR env var values to (module_path, class_name) tuples.
#: Add new detectors here — no other code changes required.
_DETECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "colour":       ("simulation_backend.vision.detection_implementation.colour_detector",  "ColourDetector"),
    "yolo":         ("simulation_backend.vision.detection_implementation.yolo_detector",    "YOLODetector"),
    "grounded_sam": ("simulation_backend.vision.detection_implementation.gsam_detector",    "GroundedSAMDetector"),
}


def get_detector(
    registry: ObjectRegistry,
    config:   dict = None,
) -> Optional[DetectorBase]:
    """
    Factory function — return the active detector or None.

    Reads the VISION_DETECTOR environment variable to select which
    detector class to instantiate.  If the variable is unset or empty,
    returns None (SceneBuilder will fall through to ground truth).

    Args:
        registry : ObjectRegistry to pass to the detector's __init__.
        config   : Full scene config dict; the detector receives the
                   sub-dict matching its name (e.g. config["colour"]).

    Returns:
        An initialised DetectorBase subclass, or None if no detector is set.

    Raises:
        EnvironmentError : If VISION_DETECTOR is set to an unknown key.
        ImportError      : If the detector's module cannot be imported
                           (missing dependency).

    Example:
        detector = get_detector(registry, cfg)
        if detector:
            detections = detector.detect(frame)
    """
    key = os.getenv("VISION_DETECTOR", "").strip().lower()

    if not key:
        logger.debug("VISION_DETECTOR not set — no primary detector active.")
        return None

    if key not in _DETECTOR_REGISTRY:
        known = ", ".join(sorted(_DETECTOR_REGISTRY))
        raise EnvironmentError(
            f"Unknown VISION_DETECTOR='{key}'. "
            f"Valid options: {known}"
        )

    module_path, class_name = _DETECTOR_REGISTRY[key]

    try:
        import importlib
        module = importlib.import_module(module_path)
        cls    = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import detector '{key}' from '{module_path}': {e}. "
            f"Check that all required dependencies are installed."
        ) from e

    detector_cfg = (config or {}).get(key, {})
    instance     = cls(registry=registry, config=detector_cfg)

    logger.info(f"Detector loaded: {class_name} (key='{key}')")
    return instance


def list_detectors() -> list[str]:
    """Return all registered detector keys (values for VISION_DETECTOR)."""
    return sorted(_DETECTOR_REGISTRY.keys())
