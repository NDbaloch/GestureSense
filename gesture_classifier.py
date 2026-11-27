"""Gesture classification: 15 gestures (rule-based + optional ML)."""
from collections import Counter, deque
from pathlib import Path

import numpy as np

import config

# MediaPipe landmark ids
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

GESTURE_NAMES = [
    "POINT", "PINCH", "OK_SIGN", "THUMBS_UP", "FIST", "THUMBS_DOWN", "PEACE",
    "THREE_FINGERS", "FOUR_FINGERS", "ROCK", "CALL_ME", "L_SHAPE",
    "PINKY_ONLY", "I_LOVE_YOU", "OPEN_PALM",
]


def normalize_landmarks(pixels, handedness=None):
    """Translate wrist to origin and scale by palm size (wrist -> middle MCP).
    Returns a (21, 2) array that is independent of hand position and size."""
    pts = np.asarray(pixels, dtype=np.float32)[:, :2]
    pts = pts - pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    return pts / max(scale, 1e-6)


def _d(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def finger_states(p):
    """Returns (thumb, index, middle, ring, pinky) as booleans (True = extended)."""
    def ext(tip, pip):
        return np.linalg.norm(p[tip]) > np.linalg.norm(p[pip]) * 1.05

    thumb = (_d(p, THUMB_TIP, INDEX_MCP) > 0.8) and (_d(p, THUMB_TIP, PINKY_MCP) > _d(p, THUMB_IP, PINKY_MCP))
    return (thumb, ext(INDEX_TIP, INDEX_PIP), ext(MIDDLE_TIP, MIDDLE_PIP),
            ext(RING_TIP, RING_PIP), ext(PINKY_TIP, PINKY_PIP))


def classify_rule(p):
    """Rule-based classifier on normalized landmarks. Returns a gesture name or 'NONE'."""
    t, i, m, r, k = finger_states(p)

    # Pinch-based gestures
    if _d(p, THUMB_TIP, INDEX_TIP) < 0.3:
        return "OK_SIGN" if (m and r and k) else "PINCH"

    pattern = (t, i, m, r, k)

    if pattern == (True, True, True, True, True):
        return "OPEN_PALM"
    if pattern == (False, True, True, True, True):
        return "FOUR_FINGERS"
    if pattern == (True, True, False, False, True):
        return "I_LOVE_YOU"
    if pattern == (True, True, False, False, False):
        return "L_SHAPE"
    if pattern == (True, False, False, False, True):
        return "CALL_ME"
    if pattern == (False, True, True, True, False):
        return "THREE_FINGERS"
    if (i, m, r, k) == (True, True, False, False):
        return "PEACE"
    if (i, m, r, k) == (True, False, False, True) and not t:
        return "ROCK"
    if (i, m, r, k) == (True, False, False, False) and not t:
        return "POINT"
    if (i, m, r, k) == (False, False, False, True) and not t:
        return "PINKY_ONLY"
    if pattern == (True, False, False, False, False):
        ty, base = p[THUMB_TIP][1], p[INDEX_MCP][1]  # image y grows downward
        if ty < base - 0.3:
            return "THUMBS_UP"
        if ty > base + 0.3:
            return "THUMBS_DOWN"
        return "NONE"
    if pattern == (False, False, False, False, False):
        return "FIST"
    return "NONE"


class GestureSmoother:
    """Majority vote over the last N frames to remove flicker."""

    def __init__(self, window=5, min_votes=3):
        self.buf = deque(maxlen=window)
        self.min_votes = min_votes
        self.current = "NONE"

    def push(self, raw):
        self.buf.append(raw)
        label, votes = Counter(self.buf).most_common(1)[0]
        if votes >= self.min_votes:
            self.current = label
        return self.current


class MLGestureClassifier:
    """Wraps a scikit-learn model trained on the 15 gesture classes."""

    def __init__(self, model_path=None):
        import joblib
        path = Path(model_path or getattr(config, "MODEL_PATH", "models/gesture_model.pkl"))
        if not path.exists():
            raise FileNotFoundError(
                f"ML model not found at {path}. Retrain it with the 15 gesture classes first.")
        self.model = joblib.load(path)

    def predict(self, norm):
        x = np.asarray(norm, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return str(self.model.classes_[idx]), float(proba[idx])
