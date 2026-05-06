# Kicau Mania Dance Detector

Real-time desktop app that detects the Kicau Mania dance with webcam input.

## Setup

Create a virtual environment and install dependencies:

```bash
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

## Run

```bash
python main.py
```

## Notes

- The preview is not mirrored by default to preserve accurate left/right hand labels.
- Edit constants in `main.py` to tune thresholds and cooldowns.
