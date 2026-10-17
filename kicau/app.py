"""Application lifecycle and coordination of detection, playback, and presentation."""

from contextlib import ExitStack
import sys
import time

import cv2

from .animation import GifPlayer
from .audio import AudioPlayer
from .config import AppConfig
from .detector import KicauDetector
from .display import Command, PreviewWindow
from .models import ModelStore
from .playback import DancePlayback
from .rendering import FrameRenderer


def camera_error_message() -> str:
    message = "Failed to open webcam.\n"
    if sys.platform == "darwin":
        message += (
            "macOS: allow camera access in System Settings > Privacy & Security > "
            "Camera for the app running Python (Terminal, iTerm, VS Code, or Codex). "
            "If a permission prompt appeared, choose Allow. Quit and reopen that app, "
            "then run python main.py again.\n"
        )
    return message + "Also check that a camera is connected and available."


class KicauApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> None:
        # Register cleanup immediately after each resource is created. ExitStack
        # still closes the other resources if startup or a shutdown callback fails.
        with ExitStack() as resources:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.config.camera_index, backend)
            resources.callback(camera.release)
            if not camera.isOpened():
                print(camera_error_message())
                return

            detector = KicauDetector(self.config.detector, ModelStore(self.config.asset_dir / "models"))
            resources.callback(detector.shutdown)
            audio = AudioPlayer(str(self.config.music_path))
            resources.callback(audio.shutdown)
            gif = GifPlayer(self.config.gif_path)
            if not gif.loaded:
                print(f"GIF could not be loaded: {self.config.gif_path}")
            window = PreviewWindow(self.config.window_title)
            resources.callback(window.shutdown)
            playback = DancePlayback(audio, gif, self.config)
            renderer = FrameRenderer(self.config)

            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                now = time.monotonic()
                data = detector.process(frame, now)
                if self.config.debug:
                    data["trajectory"] = detector.motion.get_points()
                gif_frame = playback.update(data, now)
                window.show(renderer.render(frame, data, gif_frame, window.size))
                commands = window.commands()
                if Command.QUIT in commands:
                    break
                for command in commands:
                    if command == Command.PREVIEW:
                        playback.preview(now)
