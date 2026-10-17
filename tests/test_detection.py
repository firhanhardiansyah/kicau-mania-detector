import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from kicau.config import DetectorConfig
from kicau.detector import KicauDetector
from kicau.models import ModelPaths
from kicau.motion import WaveMotionTracker, WaveResult


def tracker():
    config = DetectorConfig()
    return WaveMotionTracker(
        config.wave_window_s, config.wave_speed_px_s, config.wave_dir_changes,
        config.wave_min_move_px, config.wave_horizontal_ratio,
    )


def detector_without_models():
    with patch("kicau.detector.ModelStore.ensure", return_value=ModelPaths(Path("hand.task"), Path("face.task"))), \
            patch("kicau.detector.mp.tasks.vision.HandLandmarker.create_from_options"), \
            patch("kicau.detector.mp.tasks.vision.FaceLandmarker.create_from_options"):
        return KicauDetector(DetectorConfig())


class DetectionTests(unittest.TestCase):
    def test_smooth_wave_triggers_state_machine(self):
        detector = detector_without_models()
        motion = tracker()
        triggered = False
        # At 120 fps most individual moves are below the old 4px cutoff.
        for i in range(181):
            now = i / 120
            wave = motion.update(200 + 25 * math.sin(2 * math.pi * 2 * now), 100, now)
            event, _ = detector.gesture.update(True, True, wave, now)
            triggered |= event
        self.assertTrue(triggered)

    def test_stationary_jitter_and_vertical_motion_do_not_wave(self):
        for vertical in (False, True):
            motion = tracker()
            for i in range(90):
                result = motion.update(100 + (i % 2), i * 5 if vertical else 100, i / 60)
                self.assertFalse(result.is_waving)

    def test_fast_landmark_jitter_is_not_a_deliberate_wave(self):
        motion = tracker()
        motion.min_move_px = 2  # Sensitivity used for a smaller/distant face.
        for i in range(90):
            result = motion.update(100 + (3 if i % 2 else -3), 100, i / 60)
            self.assertFalse(result.is_waving)

    def test_missing_mouth_never_triggers(self):
        detector = detector_without_models()
        for i in range(10):
            triggered, state = detector.gesture.update(
                False, True, WaveResult(True, 1, 5, 200, 3), i / 10
            )
            self.assertFalse(triggered)
            self.assertEqual(state, "IDLE")

    def test_either_hand_can_cover_mouth_with_fingertips(self):
        for covering_label in ("Left", "Right"):
            detector = detector_without_models()
            # Palm is too far away, but a fingertip touches the mouth.
            covering = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
            covering[8] = SimpleNamespace(x=0.5, y=0.5)
            waving = [SimpleNamespace(x=0.9, y=0.8) for _ in range(21)]
            other_label = "Right" if covering_label == "Left" else "Left"
            detector.hands = Mock()
            detector.hands.detect_for_video.return_value = SimpleNamespace(
                hand_landmarks=[covering, waving],
                handedness=[[SimpleNamespace(category_name=covering_label)],
                            [SimpleNamespace(category_name=other_label)]],
            )
            detector.face_mesh = Mock()
            detector.face_mesh.detect_for_video.return_value = SimpleNamespace(
                face_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]]
            )
            result = detector.process(np.zeros((480, 640, 3), dtype=np.uint8), 1)
            self.assertTrue(result["mouth_covered"])
            self.assertTrue(result["waving_hand_present"])
            self.assertAlmostEqual(detector._wave_position[0], 576)
            self.assertAlmostEqual(detector.motion.get_points()[0][0], 576)

    def test_reset_drops_old_trajectory(self):
        motion = tracker()
        motion.update(100, 100, 1)
        motion.reset()
        self.assertEqual(motion.get_points(), [])
        self.assertFalse(motion.last_result.is_waving)

    def test_slow_waves_at_different_frame_rates(self):
        for fps in (15, 30, 60, 120):
            motion = tracker()
            detected = False
            for i in range(fps * 2):
                now = i / fps
                result = motion.update(200 + 30 * math.sin(2 * math.pi * now), 100, now)
                detected |= result.is_waving
            self.assertTrue(detected, f"Slow wave missed at {fps} fps")

    def test_short_tracking_dropout_does_not_reset_gesture(self):
        detector = detector_without_models()
        still = WaveResult(False, 0, 0, 0, 0)
        waving = WaveResult(True, 1, 3, 200, 2)
        detector.gesture.update(True, True, still, 0)
        detector.gesture.update(True, True, still, 0.26)
        detector.gesture.update(True, True, waving, 0.30)
        detector.gesture.update(False, False, still, 0.36)
        triggered, state = detector.gesture.update(True, True, waving, 0.42)
        self.assertTrue(triggered)
        self.assertEqual(state, "DANCE_DETECTED")

    def test_long_tracking_dropout_cancels_gesture(self):
        detector = detector_without_models()
        still = WaveResult(False, 0, 0, 0, 0)
        waving = WaveResult(True, 1, 3, 200, 2)
        detector.gesture.update(True, True, still, 0)
        detector.gesture.update(True, True, still, 0.26)
        detector.gesture.update(True, True, waving, 0.30)
        triggered, state = detector.gesture.update(False, False, still, 0.70)
        self.assertFalse(triggered)
        self.assertEqual(state, "IDLE")

    def test_label_changes_and_brief_hand_loss_keep_motion_history(self):
        detector = detector_without_models()
        detector.hands = Mock()
        detector.face_mesh = Mock()
        face = SimpleNamespace(
            face_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]]
        )
        detector.face_mesh.detect_for_video.return_value = face
        covering = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        waving = [SimpleNamespace(x=0.9, y=0.8) for _ in range(21)]
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for now, labels in ((1, ("Left", "Right")), (1.05, ("Right", "Left")),
                            (1.10, ("Left", "Left"))):
            detector.hands.detect_for_video.return_value = SimpleNamespace(
                hand_landmarks=[covering, waving],
                handedness=[[SimpleNamespace(category_name=label)] for label in labels],
            )
            result = detector.process(image, now)
            self.assertTrue(result["mouth_covered"])
            self.assertTrue(result["waving_hand_present"])
        self.assertEqual(len(detector.motion.get_points()), 3)
        detected_hands = detector.hands.detect_for_video.return_value
        detector.hands.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[], handedness=[]
        )
        detector.process(image, 1.15)
        self.assertEqual(len(detector.motion.get_points()), 3)
        detector.hands.detect_for_video.return_value = detected_hands
        detector.process(image, 1.20)
        self.assertEqual(len(detector.motion.get_points()), 4)
        detector.process(image, 1.60)
        self.assertEqual(len(detector.motion.get_points()), 1)

    def test_face_occlusion_cache_expires(self):
        detector = detector_without_models()
        detector.hands = Mock()
        detector.face_mesh = Mock()
        detector.hands.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]],
            handedness=[[SimpleNamespace(category_name="Left")]],
        )
        detector.face_mesh.detect_for_video.return_value = SimpleNamespace(
            face_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]]
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.process(image, 1)
        detector.face_mesh.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        self.assertTrue(detector.process(image, 1.6)["mouth_covered"])
        self.assertFalse(detector.process(image, 1.9)["mouth_covered"])


if __name__ == "__main__":
    unittest.main()
