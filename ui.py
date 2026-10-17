from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageSequence


def center_camera_on_screen(frame: np.ndarray, screen_width: int, screen_height: int) -> np.ndarray:
    """Fit the full camera image on a canvas with equal margins on opposite sides."""
    if screen_width <= 0 or screen_height <= 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(screen_width / width, screen_height / height)
    image_width = max(1, min(screen_width, round(width * scale)))
    image_height = max(1, min(screen_height, round(height * scale)))
    image = cv2.resize(
        frame, (image_width, image_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((screen_height, screen_width, 3), dtype=frame.dtype)
    x = (screen_width - image_width) // 2
    y = (screen_height - image_height) // 2
    canvas[y:y + image_height, x:x + image_width] = image
    return canvas


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
    mouth_ok = bool(data.get("mouth_covered"))
    distance = data.get("mouth_distance")
    distance_text = f"{distance:.0f}/{data.get('mouth_radius', 60):.0f}px" if distance is not None else "--"
    lines = [
        f"Mouth: {'OK' if mouth_ok else 'bring either hand closer'} ({distance_text})",
        f"Other hand: {'visible' if data.get('waving_hand_present') else 'not visible'} | Wave: {'OK' if wave and wave.is_waving else 'wave side to side'}",
        "T: test music + GIF (5s) | Q / Esc: quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 80 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)


def draw_gif_overlay(
    frame: np.ndarray, gif_frame: np.ndarray, *, left: bool = False,
    top: bool = False, mirror: bool = False
) -> None:
    """Draw the animation in a corner, optionally mirrored horizontally."""
    if mirror:
        gif_frame = cv2.flip(gif_frame, 1)
    height, width = frame.shape[:2]
    gif_height, gif_width = gif_frame.shape[:2]
    margin = min(16, width // 20, height // 20)
    scale = min(1.0, width * 0.3 / gif_width, height * 0.4 / gif_height)
    overlay_width = max(1, int(gif_width * scale))
    overlay_height = max(1, int(gif_height * scale))
    size = (overlay_width, overlay_height)
    x = margin if left else width - overlay_width - margin
    y = margin if top else height - overlay_height - margin
    target = frame[y:y + overlay_height, x:x + overlay_width]
    if gif_frame.shape[2] == 4:
        source_alpha = gif_frame[:, :, 3].astype(np.float32) / 255.0
        # Resize premultiplied colors so transparent black does not create halos.
        colors = cv2.resize(
            gif_frame[:, :, :3] * source_alpha[:, :, None], size,
            interpolation=cv2.INTER_AREA,
        )
        alpha = cv2.resize(source_alpha, size, interpolation=cv2.INTER_AREA)[:, :, None]
        target[:] = np.clip(np.rint(colors + target * (1.0 - alpha)), 0, 255).astype(np.uint8)
    else:
        target[:] = cv2.resize(gif_frame, size, interpolation=cv2.INTER_AREA)


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
    def __init__(self, gif_path: str, transparent_black: bool = True) -> None:
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
