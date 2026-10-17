import argparse
import time
import sys
from pathlib import Path

import cv2
import pygame

from audio import AudioPlayer
from detector import DetectorConfig, KicauDetector
from ui import center_camera_on_screen, draw_debug_overlay, draw_gif_overlay, GifPlayer

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_MUSIC_PATH = str(ASSET_DIR / "music.mp3")
ASSET_GIF_PATH = str(ASSET_DIR / "cat.gif")

MIRROR_PREVIEW = False
DANCE_GRACE_S = 1
WINDOW_TITLE = "Kicau Mania Detector"


def main() -> None:
    parser = argparse.ArgumentParser(description="Kicau Mania Dance Detector")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show detection status, landmarks, and motion trajectory on the camera preview.",
    )
    args = parser.parse_args()
    config = DetectorConfig()

    detector = KicauDetector(config)
    audio = AudioPlayer(ASSET_MUSIC_PATH)
    gif = GifPlayer(ASSET_GIF_PATH)
    dance_live = False
    last_dance_active = 0.0
    preview_until = 0.0
    if not gif.loaded:
        print(f"GIF could not be loaded: {ASSET_GIF_PATH}")

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)

    try:
        if not cap.isOpened():
            print("Failed to open webcam.")
            if sys.platform == "darwin":
                print(
                    "macOS: allow camera access in System Settings > Privacy & "
                    "Security > Camera for the app running Python "
                    "(Terminal, iTerm, VS Code, or Codex). If a permission prompt "
                    "appeared, choose Allow. Quit and reopen that app, then run "
                    "python main.py again."
                )
            print("Also check that a camera is connected and available.")
            return

        pygame.display.init()
        pygame.display.set_caption(WINDOW_TITLE)
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        running = True
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            data = detector.process(frame, now)
            if args.debug:
                data["trajectory"] = detector.motion.get_points()
                draw_debug_overlay(frame, data)

            wave = data.get("wave")
            dance_active = bool(data.get("mouth_covered")) and bool(
                wave and wave.is_waving
            )
            dance_active = dance_active or now < preview_until
            if dance_active:
                last_dance_active = now

            if data["triggered"]:
                dance_live = True
                print("Dance detected: playing music and GIF.")
                audio.play_once()

            if (
                dance_live
                and not dance_active
                and (now - last_dance_active) > DANCE_GRACE_S
            ):
                dance_live = False
                audio.stop()
                gif.stop()

            cat_frame = None
            if dance_live:
                if not gif.playing:
                    gif.start(now)
                cat_frame = gif.get_frame(now)
            else:
                gif.stop()

            display = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
            if cat_frame is not None:
                draw_gif_overlay(display, cat_frame)
                draw_gif_overlay(display, cat_frame, left=True, mirror=True)
            screen_width, screen_height = screen.get_size()
            display = center_camera_on_screen(display, screen_width, screen_height)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pygame.surfarray.blit_array(screen, rgb.swapaxes(0, 1))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    preview_until = now + 5.0
                    last_dance_active = now
                    dance_live = True
                    print("Testing music and GIF for 5 seconds.")
                    audio.play_once()
    finally:
        cap.release()
        pygame.display.quit()
        detector.shutdown()
        audio.shutdown()


if __name__ == "__main__":
    main()
