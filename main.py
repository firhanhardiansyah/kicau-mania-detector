import time

import cv2

from audio import AudioPlayer
from detector import DetectorConfig, KicauDetector
from ui import draw_debug_overlay, GifPlayer, GifWindow

ASSET_MUSIC_PATH = "assets/music.mp3"
ASSET_GIF_PATH = "assets/cat.gif"

MIRROR_PREVIEW = False
DANCE_GRACE_S = 1


def main() -> None:
    config = DetectorConfig(
        mouth_distance_px=60.0,
        wave_speed_px_s=140.0,
        wave_dir_changes=3,
        wave_min_move_px=4.0,
        wave_window_s=1.0,
        wave_horizontal_ratio=1.2,
        mouth_hold_s=0.25,
        wave_hold_s=0.2,
        cooldown_s=2.0,
    )

    detector = KicauDetector(config)
    audio = AudioPlayer(ASSET_MUSIC_PATH)
    gif = GifPlayer(ASSET_GIF_PATH)
    gif_window = GifWindow()
    dance_live = False
    last_dance_active = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            data = detector.process(frame, now)
            data["trajectory"] = detector.motion.get_points()
            draw_debug_overlay(frame, data)

            wave = data.get("wave")
            dance_active = bool(data.get("left_near_mouth")) and bool(
                wave and wave.is_waving
            )
            if dance_active:
                last_dance_active = now

            if data["triggered"]:
                dance_live = True
                audio.play_once()

            if (
                dance_live
                and not dance_active
                and (now - last_dance_active) > DANCE_GRACE_S
            ):
                dance_live = False
                audio.stop()
                gif.stop()
                gif_window.hide()

            if dance_live:
                if not gif.playing:
                    gif.start(now)
                cat_frame = gif.get_frame(now)
                if cat_frame is not None:
                    gif_window.show(cat_frame)
            else:
                gif.stop()
                gif_window.hide()

            display = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
            cv2.imshow("Kicau Mania Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gif_window.hide()
        detector.shutdown()
        audio.shutdown()


if __name__ == "__main__":
    main()
