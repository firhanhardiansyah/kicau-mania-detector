import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from detector import DetectorConfig, KicauDetector
from motion import WaveMotionTracker, WaveResult


def tracker():
    config = DetectorConfig()
    return WaveMotionTracker(
        config.wave_window_s, config.wave_speed_px_s, config.wave_dir_changes,
        config.wave_min_move_px, config.wave_horizontal_ratio,
    )


class DetectionTests(unittest.TestCase):
    def test_smooth_wave_triggers_state_machine(self):
        detector = KicauDetector.__new__(KicauDetector)
        detector.config = DetectorConfig()
        detector.state = "IDLE"
        detector.mouth_start = detector.wave_start = None
        detector.cooldown_until = 0
        motion = tracker()
        triggered = False
        # At 120 fps most individual moves are below the old 4px cutoff.
        for i in range(181):
            now = i / 120
            wave = motion.update(200 + 25 * math.sin(2 * math.pi * 2 * now), 100, now)
            event, _ = detector._update_state(True, True, wave, now)
            triggered |= event
        self.assertTrue(triggered)

    def test_stationary_jitter_and_vertical_motion_do_not_wave(self):
        for vertical in (False, True):
            motion = tracker()
            for i in range(90):
                result = motion.update(100 + (i % 2), i * 5 if vertical else 100, i / 60)
                self.assertFalse(result.is_waving)

    def test_missing_mouth_never_triggers(self):
        detector = KicauDetector.__new__(KicauDetector)
        detector.config = DetectorConfig()
        detector.state = "IDLE"
        for i in range(10):
            triggered, state = detector._update_state(
                False, True, WaveResult(True, 1, 5, 200, 3), i / 10
            )
            self.assertFalse(triggered)
            self.assertEqual(state, "IDLE")

    def test_either_hand_can_cover_mouth_with_fingertips(self):
        for covering_label in ("Left", "Right"):
            detector = KicauDetector.__new__(KicauDetector)
            detector.config = DetectorConfig()
            detector.motion = tracker()
            detector.state = "IDLE"
            detector.mouth_start = detector.wave_start = None
            detector.cooldown_until = 0
            detector._last_mouth = None
            detector._last_face_at = float("-inf")
            detector._mouth_radius = 60
            detector._wave_hand_label = None
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
            self.assertEqual(detector._wave_hand_label, other_label.upper())
            self.assertAlmostEqual(detector.motion.get_points()[0][0], 576)

    def test_reset_drops_old_trajectory(self):
        motion = tracker()
        motion.update(100, 100, 1)
        motion.reset()
        self.assertEqual(motion.get_points(), [])
        self.assertFalse(motion.last_result.is_waving)


if __name__ == "__main__":
    unittest.main()
