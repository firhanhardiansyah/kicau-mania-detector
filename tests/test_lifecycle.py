import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from kicau import app
from kicau.config import AppConfig, DetectorConfig
from kicau.detector import KicauDetector
from kicau.display import Command
from kicau.models import ModelPaths


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, np.zeros((180, 320, 3), dtype=np.uint8))
        self.patches.enter_context(patch.object(app.cv2, "VideoCapture", return_value=self.camera))
        self.detector_type = self.patches.enter_context(patch.object(app, "KicauDetector"))
        self.detector_type.return_value.process.return_value = {"triggered": False}
        self.audio_type = self.patches.enter_context(patch.object(app, "AudioPlayer"))
        self.gif_type = self.patches.enter_context(patch.object(app, "GifPlayer"))
        self.window_type = self.patches.enter_context(patch.object(app, "PreviewWindow"))
        self.window_type.return_value.size = (800, 600)
        self.window_type.return_value.commands.return_value = [Command.QUIT]
        self.application = app.KicauApp(AppConfig())

    def test_camera_failure_releases_capture_without_loading_models(self):
        self.camera.isOpened.return_value = False
        with patch("builtins.print"):
            self.application.run()
        self.camera.release.assert_called_once()
        self.detector_type.assert_not_called()
        self.window_type.assert_not_called()

    def test_model_startup_failure_releases_camera(self):
        self.detector_type.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self.application.run()
        self.camera.release.assert_called_once()

    def test_animation_load_failure_closes_initialized_resources(self):
        self.gif_type.side_effect = OSError("invalid animation")
        with self.assertRaisesRegex(OSError, "invalid animation"):
            self.application.run()
        self.camera.release.assert_called_once()
        self.detector_type.return_value.shutdown.assert_called_once()
        self.audio_type.return_value.shutdown.assert_called_once()

    def test_frame_processing_failure_closes_all_resources(self):
        self.detector_type.return_value.process.side_effect = ValueError("invalid frame")
        with self.assertRaisesRegex(ValueError, "invalid frame"):
            self.application.run()
        self.camera.release.assert_called_once()
        self.detector_type.return_value.shutdown.assert_called_once()
        self.audio_type.return_value.shutdown.assert_called_once()
        self.window_type.return_value.shutdown.assert_called_once()

    def test_shutdown_failure_does_not_skip_other_cleanup(self):
        self.window_type.return_value.shutdown.side_effect = RuntimeError("display shutdown")
        with self.assertRaisesRegex(RuntimeError, "display shutdown"):
            self.application.run()
        self.camera.release.assert_called_once()
        self.detector_type.return_value.shutdown.assert_called_once()
        self.audio_type.return_value.shutdown.assert_called_once()


class DetectorLifecycleTests(unittest.TestCase):
    def test_face_initialization_failure_closes_hand_model(self):
        with ExitStack() as stack:
            stack.enter_context(patch("kicau.detector.ModelStore.ensure", return_value=ModelPaths(Path("hand"), Path("face"))))
            hands = stack.enter_context(patch("kicau.detector.mp.tasks.vision.HandLandmarker.create_from_options"))
            stack.enter_context(patch("kicau.detector.mp.tasks.vision.FaceLandmarker.create_from_options", side_effect=RuntimeError("face init")))
            with self.assertRaisesRegex(RuntimeError, "face init"):
                KicauDetector(DetectorConfig())
            hands.return_value.close.assert_called_once()
