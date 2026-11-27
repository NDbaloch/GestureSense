"""Thin wrapper around MediaPipe Hands."""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass
class HandData:
    points: np.ndarray      # (21, 2) landmarks normalised to 0..1
    pixels: np.ndarray      # (21, 2) landmarks in pixel units
    handedness: str         # "Left" or "Right" (as seen in the mirrored image)
    landmarks: object       # raw MediaPipe landmarks (used for drawing)


class HandTracker:
    def __init__(self, max_hands=config.MAX_HANDS):
        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """Return HandData for the first detected hand, or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0]
        label = result.multi_handedness[0].classification[0].label
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        pix = pts * np.array([w, h], dtype=np.float32)
        return HandData(points=pts, pixels=pix, handedness=label, landmarks=lm)

    def draw(self, frame, hand):
        self._drawing.draw_landmarks(
            frame,
            hand.landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
