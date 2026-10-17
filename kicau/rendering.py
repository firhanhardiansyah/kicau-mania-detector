from typing import Dict, Optional

import cv2
import numpy as np
from .config import AppConfig, OverlayPlacement


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
        wave_text = (
            f"Wave conf: {wave.confidence:.2f} | changes: {wave.direction_changes}"
            f" | speed: {wave.avg_speed:.1f}"
        )
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
    distance_text = (
        f"{distance:.0f}/{data.get('mouth_radius', 60):.0f}px"
        if distance is not None else "--"
    )
    hand_status = "visible" if data.get("waving_hand_present") else "not visible"
    wave_status = "OK" if wave and wave.is_waving else "wave side to side"
    lines = [
        f"Mouth: {'OK' if mouth_ok else 'bring either hand closer'} ({distance_text})",
        f"Other hand: {hand_status} | Wave: {wave_status}",
        "T: test music + GIF (5s) | Q / Esc: quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (12, 80 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2,
        )


def draw_gif_overlay(
    frame: np.ndarray, gif_frame: np.ndarray, placement: OverlayPlacement = OverlayPlacement()
) -> None:
    """Draw in a corner or one half of a centered pair, optionally mirrored."""
    if placement.mirror:
        gif_frame = cv2.flip(gif_frame, 1)
    height, width = frame.shape[:2]
    gif_height, gif_width = gif_frame.shape[:2]
    margin = min(16, width // 20, height // 20)
    scale = min(1.0, width * 0.3 / gif_width, height * 0.4 / gif_height)
    if placement.center:
        # Fit the pair between the existing corner GIFs with a small gap.
        scale = min(scale, max(1, width * 0.2 - 2 * margin) / gif_width)
    overlay_width = max(1, int(gif_width * scale))
    overlay_height = max(1, int(gif_height * scale))
    size = (overlay_width, overlay_height)
    x = margin if placement.left else width - overlay_width - margin
    if placement.center:
        if placement.left:
            x = width // 2 - margin // 2 - overlay_width
        else:
            x = (width + 1) // 2 + margin // 2
        x = max(0, min(width - overlay_width, x))
    y = margin if placement.top else height - overlay_height - margin
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


class FrameRenderer:
    """Compose camera, optional diagnostics, and configured animations without GUI calls."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def render(
        self, frame: np.ndarray, data: Dict, gif_frame: np.ndarray | None,
        screen_size: tuple[int, int],
    ) -> np.ndarray:
        display = frame.copy()
        if self.config.debug:
            draw_debug_overlay(display, data)
        if self.config.mirror_preview:
            display = cv2.flip(display, 1)
        if gif_frame is not None:
            for placement in self.config.overlays:
                draw_gif_overlay(display, gif_frame, placement)
        return center_camera_on_screen(display, *screen_size)

