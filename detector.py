from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import shutil
import ssl
import tempfile
import urllib.request

import certifi
import cv2
import mediapipe as mp
import numpy as np

from motion import WaveMotionTracker, WaveResult

MOUTH_LANDMARKS = [13, 14, 78, 308, 61, 291]
MODEL_DIR = Path(__file__).resolve().parent / "assets" / "models"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class DetectorConfig:
    mouth_distance_px: float = 60.0
    wave_speed_px_s: float = 75.0
    wave_dir_changes: int = 2
    wave_min_move_px: float = 4.0
    wave_window_s: float = 1.8
    wave_horizontal_ratio: float = 1.1
    mouth_hold_s: float = 0.25
    wave_hold_s: float = 0.1
    cooldown_s: float = 2.0
    tracking_grace_s: float = 0.3
    face_grace_s: float = 0.8


class KicauDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._ensure_models()

        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        running_mode = vision.RunningMode.VIDEO

        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        self.face_mesh = vision.FaceLandmarker.create_from_options(face_options)
        self.motion = WaveMotionTracker(
            window_seconds=config.wave_window_s,
            min_speed=config.wave_speed_px_s,
            min_dir_changes=config.wave_dir_changes,
            min_move_px=config.wave_min_move_px,
            min_horizontal_ratio=config.wave_horizontal_ratio,
        )
        self._reset_detection_state()

    def _reset_detection_state(self) -> None:
        self.state = "IDLE"
        self.mouth_start: Optional[float] = None
        self.wave_start: Optional[float] = None
        self.cooldown_until: float = 0.0
        self._last_mouth = None
        self._last_face_at = float("-inf")
        self._mouth_radius = self.config.mouth_distance_px
        self._wave_position = None
        self._wave_seen_at = float("-inf")
        self._mouth_seen_at = float("-inf")
        self._wave_active_at = float("-inf")

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Dict:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp * 1000)
        hands_result = self.hands.detect_for_video(image, timestamp_ms)
        face_result = self.face_mesh.detect_for_video(image, timestamp_ms)

        mouth_center = self._get_mouth_center(face_result, width, height)
        if mouth_center is not None:
            self._last_mouth = mouth_center
            self._last_face_at = timestamp
            face_xs = [lm.x * width for lm in face_result.face_landmarks[0]]
            self._mouth_radius = max(
                self.config.mouth_distance_px, (max(face_xs) - min(face_xs)) * 0.45
            )
            # Keep sensitivity comparable when standing closer to/farther from the camera.
            scale = max(0.5, min(2.0, (max(face_xs) - min(face_xs)) / 200.0))
            self.motion.min_speed = self.config.wave_speed_px_s * scale
            self.motion.min_move_px = self.config.wave_min_move_px * scale
        elif timestamp - self._last_face_at <= self.config.face_grace_s:
            # A hand over the mouth can briefly hide the face landmarks.
            mouth_center = self._last_mouth
        left_hand = None
        right_hand = None
        hands = []

        if hands_result.hand_landmarks and hands_result.handedness:
            for landmarks, handedness in zip(
                hands_result.hand_landmarks, hands_result.handedness
            ):
                label = handedness[0].category_name.upper()
                hand_data = self._extract_hand(landmarks, label, width, height)
                hands.append(hand_data)
                if label == "LEFT":
                    left_hand = hand_data
                elif label == "RIGHT":
                    right_hand = hand_data

        # Assign roles by position, so either hand can cover the mouth.
        covering_hand = None
        mouth_distance = None
        if mouth_center is not None and hands:
            covering_hand = min(
                hands, key=lambda hand: self._hand_mouth_distance(hand, mouth_center)
            )
            mouth_distance = self._hand_mouth_distance(covering_hand, mouth_center)
            if mouth_distance > self._mouth_radius:
                covering_hand = None
        mouth_covered = covering_hand is not None
        candidates = [hand for hand in hands if hand is not covering_hand]
        if timestamp - self._wave_seen_at > self.config.tracking_grace_s:
            self.motion.reset()
            self._wave_position = None
        # Follow position rather than handedness: labels can flicker during fast waves.
        waving_hand = min(
            candidates,
            key=lambda hand: self._distance(hand["palm_center"], self._wave_position)
            if self._wave_position is not None else 0,
            default=None,
        )

        wave_result = WaveResult(False, 0.0, 0, 0.0, 0.0)
        if waving_hand:
            position = waving_hand["palm_center"]
            if self._wave_position is not None and self._distance(position, self._wave_position) > width * 0.3:
                self.motion.reset()
            self._wave_position = position
            self._wave_seen_at = timestamp
            rx, ry = waving_hand["wave_point"]
            wave_result = self.motion.update(rx, ry, timestamp)

        triggered, display_state = self._update_state(
            mouth_covered, waving_hand is not None, wave_result, timestamp
        )

        return {
            "mouth_center": mouth_center,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "mouth_covered": mouth_covered,
            "mouth_distance": mouth_distance,
            "mouth_radius": self._mouth_radius,
            "waving_hand_present": waving_hand is not None,
            "wave": wave_result,
            "state": display_state,
            "triggered": triggered,
        }

    def shutdown(self) -> None:
        self.hands.close()
        self.face_mesh.close()

    def _update_state(
        self, mouth_covered: bool, waving_hand_present: bool, wave: WaveResult, now: float
    ) -> Tuple[bool, str]:
        if mouth_covered:
            self._mouth_seen_at = now
        mouth_recent = now - self._mouth_seen_at <= self.config.tracking_grace_s
        if waving_hand_present and wave.is_waving:
            self._wave_active_at = now
        wave_active = now - self._wave_active_at <= self.config.tracking_grace_s
        triggered = False
        display_state = self.state

        if self.state == "IDLE":
            if mouth_recent:
                self.state = "MOUTH_COVERED"
                self.mouth_start = now
                display_state = self.state
        elif self.state == "MOUTH_COVERED":
            if not mouth_recent:
                self.state = "IDLE"
                self.mouth_start = None
            elif (
                self.mouth_start is not None
                and now - self.mouth_start >= self.config.mouth_hold_s
            ):
                self.state = "WAVING"
                self.wave_start = None
                display_state = "MOUTH_COVERED"
        elif self.state == "WAVING":
            if not mouth_recent:
                self.state = "IDLE"
                self.wave_start = None
            else:
                if wave_active:
                    if self.wave_start is None:
                        self.wave_start = now
                    elif now - self.wave_start >= self.config.wave_hold_s:
                        triggered = True
                        display_state = "DANCE_DETECTED"
                        self.state = "COOLDOWN"
                        self.cooldown_until = now + self.config.cooldown_s
                        self.wave_start = None
                else:
                    display_state = "MOUTH_COVERED"
                    self.wave_start = None
        elif self.state == "COOLDOWN":
            if now >= self.cooldown_until:
                self.state = "IDLE"
                self.mouth_start = None
                self.wave_start = None

        if not triggered and self.state == "IDLE":
            display_state = "IDLE"
        return triggered, display_state

    def _hand_mouth_distance(self, hand: Dict, mouth_center) -> float:
        # Fingers often cover the mouth while the palm stays below it.
        return min(
            self._distance(mouth_center, point)
            for point in [hand["palm_center"], *hand["landmarks"]]
        )

    def _get_mouth_center(
        self, face_result, width: int, height: int
    ) -> Optional[Tuple[float, float]]:
        if not face_result.face_landmarks:
            return None
        face_landmarks = face_result.face_landmarks[0]
        xs = [face_landmarks[i].x * width for i in MOUTH_LANDMARKS]
        ys = [face_landmarks[i].y * height for i in MOUTH_LANDMARKS]
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    def _extract_hand(self, landmarks, label: str, width: int, height: int) -> Dict:
        pts = [(lm.x * width, lm.y * height) for lm in landmarks]
        palm_ids = [0, 5, 9, 13, 17]
        palm_x = sum(pts[i][0] for i in palm_ids) / len(palm_ids)
        palm_y = sum(pts[i][1] for i in palm_ids) / len(palm_ids)
        wave_points = [(palm_x, palm_y), *(pts[i] for i in (8, 12, 16, 20))]
        return {
            "label": label,
            "landmarks": pts,
            "wrist": pts[0],
            "palm_center": (palm_x, palm_y),
            "wave_point": (
                sum(p[0] for p in wave_points) / len(wave_points),
                sum(p[1] for p in wave_points) / len(wave_points),
            ),
        }

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ensure_models(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._download_if_missing(HAND_MODEL_PATH, HAND_MODEL_URL)
        self._download_if_missing(FACE_MODEL_PATH, FACE_MODEL_URL)

    @staticmethod
    def _download_if_missing(path: Path, url: str) -> None:
        if path.is_file() and path.stat().st_size > 0:
            return
        context = ssl.create_default_context(cafile=certifi.where())
        temporary_path = None
        try:
            with urllib.request.urlopen(url, context=context, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                    temporary_path = Path(output.name)
                    shutil.copyfileobj(response, output)
            temporary_path.replace(path)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"Failed to download model {path.name}. Check your internet connection "
                "and run: python -m pip install --upgrade certifi"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
