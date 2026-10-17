import unittest

import numpy as np

from kicau.config import AppConfig, OverlayPlacement
from kicau.rendering import FrameRenderer


class RenderingTests(unittest.TestCase):
    def test_six_overlays_are_mirrored_without_mutating_camera_input(self):
        camera = np.full((480, 640, 3), 180, dtype=np.uint8)
        animation = np.full((100, 100, 4), 255, dtype=np.uint8)
        animation[:, :50, :3] = (10, 30, 200)
        animation[:, 50:, :3] = (200, 30, 10)
        output = FrameRenderer(AppConfig()).render(camera, {}, animation, (640, 480))
        self.assertTrue(np.all(camera == 180))
        difference = np.abs(output[:, :320].astype(int) - output[:, 320:][:, ::-1].astype(int))
        self.assertLessEqual(difference.max(), 1)
        for x, y in ((30, 30), (600, 30), (230, 30), (340, 30), (30, 400), (600, 400)):
            self.assertFalse(np.all(output[y, x] == 180), (x, y))
        self.assertTrue(np.all(output[220:260, 250:390] == 180))

    def test_custom_layout_does_not_require_changes_to_application_loop(self):
        config = AppConfig(overlays=(OverlayPlacement(top=True, left=True),))
        camera = np.zeros((480, 640, 3), dtype=np.uint8)
        animation = np.full((40, 40, 4), 255, dtype=np.uint8)
        output = FrameRenderer(config).render(camera, {}, animation, (640, 480))
        self.assertTrue(np.all(output[30, 30] == 255))
        self.assertTrue(np.all(output[450, 600] == 0))
