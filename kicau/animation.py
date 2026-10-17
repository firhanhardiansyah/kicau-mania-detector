from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageSequence

def remove_black_background(frame: np.ndarray, threshold: int = 35) -> np.ndarray:
    """Remove near-black areas connected to the edges, keeping enclosed dark details."""
    dark = np.max(frame[:, :, :3], axis=2) <= threshold
    # Padding joins background regions touching any of the four edges.
    mask = np.pad(dark.astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(mask, None, (0, 0), 2, flags=4)
    result = frame.copy()
    result[mask[1:-1, 1:-1] == 2, 3] = 0
    return result


class GifPlayer:
    def __init__(self, gif_path: str | Path, transparent_black: bool = True) -> None:
        self.gif_path = gif_path
        self.transparent_black = transparent_black
        self.frames = []
        self.durations = []
        self.total_duration = 0.0
        self.playing = False
        self.start_time = 0.0
        self.loaded = False
        self._load()

    def _load(self) -> None:
        try:
            img = Image.open(self.gif_path)
        except FileNotFoundError:
            self.loaded = False
            return

        with img:
            for frame in ImageSequence.Iterator(img):
                duration_ms = frame.info.get("duration", 100)
                frame_rgba = frame.convert("RGBA")
                frame_np = np.array(frame_rgba)[:, :, [2, 1, 0, 3]]
                if self.transparent_black:
                    frame_np = remove_black_background(frame_np)
                self.frames.append(frame_np)
                self.durations.append(duration_ms / 1000.0)

        self.total_duration = sum(self.durations) if self.durations else 0.0
        self.loaded = bool(self.frames)

    def start(self, now: float) -> None:
        if not self.loaded:
            return
        self.playing = True
        self.start_time = now

    def stop(self) -> None:
        self.playing = False

    def get_frame(self, now: float) -> Optional[np.ndarray]:
        if not self.playing or not self.loaded or self.total_duration <= 0:
            return None
        elapsed = (now - self.start_time) % self.total_duration
        acc = 0.0
        for frame, duration in zip(self.frames, self.durations):
            acc += duration
            if elapsed <= acc:
                return frame
        return self.frames[-1]
