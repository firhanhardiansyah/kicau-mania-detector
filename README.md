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
- The preview shows mouth proximity, whether the other hand is visible, and
  whether a wave was detected.
- Focus the webcam preview window and press **T** to test the music and dancing
  cat for five seconds without performing a gesture. Press **Q** or **Esc** to exit.
- The preview is not mirrored by default. Either hand can perform either role.
- Edit `DetectorConfig` in `detector.py` to tune thresholds and cooldowns.
