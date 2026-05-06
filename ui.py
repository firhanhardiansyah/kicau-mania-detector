from typing import Dict, Optional, Tuple
import time

import cv2
import numpy as np
from PIL import Image, ImageSequence


def draw_debug_overlay(frame: np.ndarray, data: Dict) -> None:
    if data.get("mouth_center"):
        mx, my = data["mouth_center"]
        cv2.circle(frame, (int(mx), int(my)), 6, (0, 255, 255), -1)

    for hand_key, color in [("left_hand", (0, 255, 0)), ("right_hand", (255, 0, 0))]:
        hand = data.get(hand_key)
        if not hand:
            continue
        for x, y in hand["landmarks"]:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        wx, wy = hand["wrist"]
        cv2.circle(frame, (int(wx), int(wy)), 6, color, 2)

    trajectory = data.get("trajectory", [])
    for i in range(1, len(trajectory)):
        p0 = trajectory[i - 1]
        p1 = trajectory[i]
        cv2.line(
            frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 200, 255), 2
        )

    state = data.get("state", "UNKNOWN")
    wave = data.get("wave")
    status_text = f"STATE: {state}"
    cv2.putText(
        frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    if wave:
        wave_text = f"Wave conf: {wave.confidence:.2f} | changes: {wave.direction_changes} | speed: {wave.avg_speed:.1f}"
        cv2.putText(
            frame,
            wave_text,
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


class GifPlayer:
    def __init__(self, gif_path: str) -> None:
        self.gif_path = gif_path
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

        for frame in ImageSequence.Iterator(img):
            duration_ms = frame.info.get("duration", 100)
            frame_rgb = frame.convert("RGB")
            frame_np = np.array(frame_rgb)[:, :, ::-1]
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


class GifWindow:
    def __init__(self, title: str = "Dancing Cat") -> None:
        self.title = title
        self.is_open = False

    def show(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        cv2.imshow(self.title, frame)
        self.is_open = True

    def hide(self) -> None:
        if self.is_open:
            cv2.destroyWindow(self.title)
            self.is_open = False
