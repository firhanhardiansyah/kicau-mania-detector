"""Coordinate music and animation timing without depending on a display or camera."""

import numpy as np

from .animation import GifPlayer
from .audio import AudioPlayer
from .config import AppConfig


class DancePlayback:
    def __init__(self, audio: AudioPlayer, gif: GifPlayer, config: AppConfig) -> None:
        self.audio = audio
        self.gif = gif
        self.config = config
        self.active = False
        self.last_active = 0.0
        self.preview_until = 0.0

    def preview(self, now: float) -> None:
        self.preview_until = now + self.config.test_duration_s
        self.last_active = now
        self.active = True
        print(f"Testing music and GIF for {self.config.test_duration_s:g} seconds.")
        self.audio.play_once()

    def update(self, data: dict, now: float) -> np.ndarray | None:
        wave = data.get("wave")
        dance_active = bool(data.get("mouth_covered") and wave and wave.is_waving)
        dance_active = dance_active or now < self.preview_until
        if dance_active:
            self.last_active = now
        if data.get("triggered"):
            self.active = True
            self.last_active = now
            print("Dance detected: playing music and GIF.")
            self.audio.play_once()
        if self.active and not dance_active and now - self.last_active > self.config.dance_grace_s:
            self.active = False
            self.audio.stop()
            self.gif.stop()
        if not self.active:
            self.gif.stop()
            return None
        if not self.gif.playing:
            self.gif.start(now)
        return self.gif.get_frame(now)
