import unittest
from unittest.mock import Mock

import numpy as np

from kicau.config import AppConfig
from kicau.motion import WaveResult
from kicau.playback import DancePlayback


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.audio = Mock()
        self.gif = Mock(playing=False)
        self.gif.start.side_effect = lambda now: setattr(self.gif, "playing", True)
        self.gif.stop.side_effect = lambda: setattr(self.gif, "playing", False)
        self.frame = np.ones((10, 10, 4), dtype=np.uint8)
        self.gif.get_frame.return_value = self.frame
        self.playback = DancePlayback(self.audio, self.gif, AppConfig())

    def test_detection_starts_playback_and_stops_after_grace(self):
        data = {"mouth_covered": True, "wave": WaveResult(True, 1, 3, 150, 2), "triggered": True}
        self.assertIs(self.playback.update(data, 10), self.frame)
        self.assertIs(self.playback.update({}, 10.5), self.frame)
        self.assertIsNone(self.playback.update({}, 11.1))
        self.audio.play_once.assert_called_once()
        self.audio.stop.assert_called_once()
        self.gif.start.assert_called_once_with(10)
        self.assertFalse(self.playback.active)

    def test_preview_bypasses_detection_for_configured_duration(self):
        self.playback.preview(10)
        self.assertIs(self.playback.update({}, 14.9), self.frame)
        self.assertIsNone(self.playback.update({}, 16.1))
        self.audio.play_once.assert_called_once()
        self.audio.stop.assert_called_once()

    def test_trigger_survives_raw_tracking_dropout(self):
        # The gesture recognizer can emit an event during tracking grace.
        self.assertIs(self.playback.update({"triggered": True}, 10), self.frame)
        self.assertTrue(self.playback.active)

    def test_idle_does_not_start_audio_or_animation(self):
        self.assertIsNone(self.playback.update({}, 10))
        self.audio.play_once.assert_not_called()
        self.gif.start.assert_not_called()
