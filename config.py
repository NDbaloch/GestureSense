"""Central settings for GestureSense. Tweak values here instead of editing the code."""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = DATA_DIR / "gestures.csv"
MODEL_FILE = MODEL_DIR / "gesture_model.joblib"

# ---------- Camera ----------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------- MediaPipe ----------
MAX_HANDS = 1
MODEL_COMPLEXITY = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# ---------- Gestures ----------
GESTURES = ["POINT", "PINCH", "OK_SIGN", "THUMBS_UP", "FIST", "THUMBS_DOWN", "PEACE",
            "THREE_FINGERS", "FOUR_FINGERS", "ROCK", "CALL_ME", "L_SHAPE",
            "PINKY_ONLY", "I_LOVE_YOU", "OPEN_PALM"]

# ---------- Rule-based thresholds ----------
THUMB_EXTENDED_DIST = 0.60
PINCH_DIST = 0.30
PINCH_MIN_INDEX_REACH = 1.20
THUMBS_UP_MIN_RISE = 0.30

# ---------- Classification ----------
SMOOTHING_FRAMES = 7
ML_MIN_CONFIDENCE = 0.60

# ---------- Actions ----------
ACTIVE_MARGIN = 0.20
MOUSE_SMOOTHING = 0.30
CLICK_COOLDOWN = 0.50
SLIDE_COOLDOWN = 1.00
VOLUME_GAIN = 1.5
VOLUME_DEADZONE = 0.003

# ---------- Database ----------
DB_BACKEND = "none"
DB_NAME = "gesturesense"

MYSQL = {"host": "localhost", "port": 3306, "user": "root", "password": "YOUR_MYSQL_PASSWORD"}
POSTGRES = {"host": "localhost", "port": 5432, "user": "postgres", "password": "YOUR_POSTGRES_PASSWORD"}

try:
    from config_local import MYSQL, POSTGRES  # noqa: F401
except ImportError:
    pass
