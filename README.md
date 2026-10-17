# Kicau Mania Dance Detector

Real-time desktop app that detects the Kicau Mania dance with webcam input.

## Setup

Create a virtual environment and install dependencies:

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Assets

Place your files here:

- assets/music.mp3
- assets/cat.gif

## MediaPipe Models

The app uses MediaPipe Tasks models for hands and face mesh. On first run it downloads:

- assets/models/hand_landmarker.task
- assets/models/face_landmarker.task

Ensure you have an internet connection the first time you start the app.
Downloads use the trusted CA certificates supplied by `certifi`. If Python
reports `CERTIFICATE_VERIFY_FAILED`, update it with
`python -m pip install --upgrade certifi`.

## Run

```bash
python main.py
```

The camera preview opens fullscreen with the complete camera image centered
without stretching or cropping. Any unused space is split evenly between
opposite sides. Pygame handles the fullscreen display directly, avoiding the
extra window padding seen with OpenCV on macOS. Press **Q** or **Esc** to exit.

To show detection status, landmarks, and motion trajectory on the camera preview:

```bash
python main.py --debug
```

## Notes

### macOS camera permission

If OpenCV reports `not authorized to capture video` or `Failed to open webcam`,
allow camera access in **System Settings > Privacy & Security > Camera** for the
app used to run Python (such as Terminal, iTerm, VS Code, or Codex). Choose
**Allow** if a permission prompt appears. Quit and reopen that app after granting
permission, then run `python main.py` again. If the app is not listed, run the
command from Terminal to trigger its camera permission request.

### Detection

- Bring either hand close to your mouth (fingers count), then wave the other
  hand side to side a few times. Keep your face and both hands in view.
- Detection tolerates brief tracking interruptions (0.3 seconds), keeps the
  last mouth position for up to 0.8 seconds when obscured, and adapts wave
  sensitivity to face size. Small landmark jitter does not count as a wave.
- With `--debug`, the preview shows mouth proximity, whether the other hand is
  visible, and whether a wave was detected. These overlays are hidden by default.
- Focus the webcam preview window and press **T** to test the music and dancing
  cat for five seconds without performing a gesture. Press **Q** or **Esc** to exit.
- Dancing cats appear in all four corners of the camera preview, with both
  left copies mirrored. An additional mirrored pair sits at the top center,
  sized to fit between the corner GIFs. All six play in sync, preserving the GIF's
  aspect ratio and transparency.
  Black background areas connected to the GIF's edges are made transparent
  automatically; enclosed dark details such as the cat's eyes are preserved.
- The preview is not mirrored by default. Either hand can perform either role.
- Edit `DetectorConfig` in `detector.py` to tune thresholds and cooldowns.
