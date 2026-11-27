# GestureSense

**Real-time contactless human-computer interaction using hand gesture recognition.**

Control your computer with hand gestures using only a normal webcam. GestureSense tracks your hand with
[MediaPipe](https://developers.google.com/mediapipe), recognises **15 gestures** with a rule-based classifier, and
turns them into actions such as moving the mouse, clicking, changing slides, adjusting volume and more.
Every session and every recognised gesture is also logged to **MySQL** and **PostgreSQL**.

Built with Python, OpenCV, MediaPipe, PyAutoGUI, MySQL and PostgreSQL. No extra hardware needed.

## Features

- Real-time hand tracking (21 landmarks) from a standard webcam
- 15 gestures recognised with geometric rules on normalised landmarks (independent of hand size and position)
- Smooth mouse control with a mapped active region and cursor smoothing
- Temporal smoothing (majority vote) to avoid accidental triggers
- Session and gesture-event logging to MySQL and/or PostgreSQL (background thread, never slows the video)
- On-screen display with gesture, action, FPS and latency
- Optional tools to record your own landmark dataset and train k-NN / Random Forest classifiers

## Gestures and actions

| Gesture | Hand pose | Action |
|---|---|---|
| POINT | Index finger only | Move mouse cursor |
| PINCH | Thumb touches index | Left click |
| OK_SIGN | Thumb + index circle, other fingers up | Right click |
| THUMBS_UP | Thumb up | Next slide |
| FIST | All fingers folded | Previous slide |
| THUMBS_DOWN | Thumb down | Mute / unmute |
| PEACE | Index + middle, then move hand up/down | Volume up / down |
| THREE_FINGERS | Index + middle + ring, then move hand up/down | Scroll |
| FOUR_FINGERS | Four fingers, thumb folded | Play / pause |
| ROCK | Index + pinky | Screenshot |
| CALL_ME | Thumb + pinky | Switch window (Alt+Tab) |
| L_SHAPE | Thumb + index | Browser back |
| PINKY_ONLY | Pinky only | Show desktop |
| I_LOVE_YOU | Thumb + index + pinky | Toggle full screen (F11) |
| OPEN_PALM | All fingers open | Idle (no action) |

## How it works

```
Webcam -> OpenCV (mirror) -> MediaPipe Hands (21 landmarks) -> normalisation
      -> rule-based classifier -> smoother -> actions (PyAutoGUI)
                                           -> database logger (MySQL / PostgreSQL)
                                           -> on-screen display
```

Performance on a CPU-only laptop: about 17 FPS with roughly 28 ms processing time per frame.

## Installation

Requires **Windows**, a webcam and **Python 3.9 - 3.12** (MediaPipe does not support newer Python versions yet).

```bash
git clone https://github.com/NDbaloch/GestureSense.git
cd GestureSense
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```bash
.venv\Scripts\python.exe main.py                  # run without a database
.venv\Scripts\python.exe main.py --no-actions     # show gestures without controlling the PC
.venv\Scripts\python.exe main.py --db both        # also log to MySQL and PostgreSQL
```

Keys in the video window: **P** pause/resume actions, **Q** or **Esc** quit.
Note: without `--no-actions` the program really controls your mouse and keyboard.

## Database setup (optional)

1. Install MySQL and/or PostgreSQL.
2. Create a file `config_local.py` next to `config.py` with your own credentials (this file is not uploaded to GitHub):

   ```python
   MYSQL = {"host": "localhost", "port": 3306, "user": "root", "password": "YOUR_MYSQL_PASSWORD"}
   POSTGRES = {"host": "localhost", "port": 5432, "user": "postgres", "password": "YOUR_POSTGRES_PASSWORD"}
   ```

3. Create the database and tables once, then run with logging and view the saved data:

   ```bash
   .venv\Scripts\python.exe init_db.py --db both
   .venv\Scripts\python.exe main.py --db both
   .venv\Scripts\python.exe view_report.py --db postgres
   ```

Tables: `sessions` (start/end time, classifier, frames, average FPS and latency) and
`gesture_events` (time, gesture, action, confidence).

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Live application |
| `hand_tracker.py` | MediaPipe Hands wrapper |
| `gesture_classifier.py` | Normalisation, finger states, rule-based classifier, smoothing |
| `actions.py` | Gesture-to-action mapping (PyAutoGUI) |
| `config.py` | Settings and thresholds |
| `database.py`, `init_db.py`, `view_report.py` | MySQL / PostgreSQL logging and reports |
| `collect_data.py`, `train_model.py` | Optional dataset recording and ML training tools |
| `run.bat` | One-click launcher (Windows) |

## Limitations

- One hand, facing the camera, in reasonable lighting
- Fixed geometric thresholds that may need tuning (see `config.py`)
- Volume, window and key actions are designed for Windows

## Author

**Naveed Essa**

