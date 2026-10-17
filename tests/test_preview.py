import sys
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np
import pygame

from kicau import app, cli


class PreviewTests(unittest.TestCase):
    def test_fullscreen_centering_colors_keyboard_and_debug(self):
        for debug in (False, True):
            with self.subTest(debug=debug), ExitStack() as stack:
                argv = ["main.py"] + (["--debug"] if debug else [])
                stack.enter_context(patch.object(sys, "argv", argv))
                detector = stack.enter_context(patch.object(app, "KicauDetector")).return_value
                detector.process.return_value = {"triggered": False}
                audio = stack.enter_context(patch.object(app, "AudioPlayer")).return_value
                gif = stack.enter_context(patch.object(app, "GifPlayer")).return_value
                gif.get_frame.return_value = np.full((40, 40, 4), 255, dtype=np.uint8)
                camera = np.full((180, 320, 3), (10, 80, 200), dtype=np.uint8)
                cap = Mock()
                cap.isOpened.return_value = True
                cap.read.return_value = (True, camera)
                stack.enter_context(patch.object(app.cv2, "VideoCapture", return_value=cap))
                overlay = stack.enter_context(patch("kicau.rendering.draw_debug_overlay"))
                screen = pygame.Surface((800, 600))
                mode = stack.enter_context(patch.object(pygame.display, "set_mode", return_value=screen))
                for method in ("init", "set_caption", "quit"):
                    stack.enter_context(patch.object(pygame.display, method))
                snapshots = []
                stack.enter_context(patch.object(
                    pygame.display, "flip",
                    side_effect=lambda: snapshots.append(pygame.surfarray.array3d(screen)),
                ))
                stack.enter_context(patch.object(pygame.event, "get", side_effect=[
                    [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)],
                    [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
                ]))
                cli.main()
                mode.assert_called_once_with((0, 0), pygame.FULLSCREEN)
                self.assertEqual(len(snapshots), 2)
                # 16:9 camera on a 4:3 display: precisely 75px above and below.
                first = snapshots[0]
                self.assertTrue(np.all(first[:, :75] == 0))
                self.assertTrue(np.all(first[:, 525:] == 0))
                self.assertTrue(np.all(first[:, 75:525] == (200, 80, 10)))
                self.assertFalse(np.array_equal(snapshots[0], snapshots[1]))
                self.assertEqual(overlay.call_count, 2 if debug else 0)
                audio.play_once.assert_called_once()
                cap.release.assert_called_once()
                detector.shutdown.assert_called_once()
                audio.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
