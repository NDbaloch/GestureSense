"""Maps recognised gestures to computer actions (cursor, click, slides, volume, ...)."""
import time
from datetime import datetime

import numpy as np
import pyautogui

import config

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def _xy(points, i):
    """Works for (x, y) tuples/arrays or MediaPipe landmark objects (normalised 0..1)."""
    p = points[i]
    if hasattr(p, "x"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


class ActionController:
    ONE_SHOT_COOLDOWN = 0.8   # seconds between one-shot actions
    CLICK_COOLDOWN = 0.4

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.status = "Idle"
        self.screen_w, self.screen_h = pyautogui.size()
        self.cur = None            # smoothed cursor position
        self.ref_y = None          # reference y for volume / scroll
        self.last_gesture = None
        self.last_fire = {}

    # ------------------------------------------------------------------
    def reset(self):
        self.ref_y = None
        self.last_gesture = None
        self.status = "No hand"

    def _cool(self, name, seconds):
        now = time.time()
        if now - self.last_fire.get(name, 0.0) >= seconds:
            self.last_fire[name] = now
            return True
        return False

    def _vertical_delta(self, y, deadzone):
        """Returns -1 (hand moved up), +1 (down) or 0, re-anchoring after each step."""
        if self.ref_y is None:
            self.ref_y = y
            return 0
        d = y - self.ref_y
        if abs(d) < deadzone:
            return 0
        self.ref_y = y
        return -1 if d < 0 else 1

    # ------------------------------------------------------------------
    def update(self, gesture, points):
        if gesture != self.last_gesture:
            self.ref_y = None
        new_gesture = gesture != self.last_gesture
        self.last_gesture = gesture

        if not self.enabled:
            self.status = "Paused"
            return

        x, y = _xy(points, 8)            # index fingertip
        wy = _xy(points, 0)[1]           # wrist y (used for up/down motion)

        if gesture == "POINT":
            self._move_cursor(x, y)
            self.status = "Moving cursor"
        elif gesture == "PINCH":
            if new_gesture and self._cool("click", self.CLICK_COOLDOWN):
                pyautogui.click()
                self.status = "Left click"
        elif gesture == "OK_SIGN":
            if new_gesture and self._cool("rclick", self.CLICK_COOLDOWN):
                pyautogui.rightClick()
                self.status = "Right click"
        elif gesture == "PEACE":
            step = self._vertical_delta(wy, 0.04)
            if step and self._cool("volume", 0.12):
                pyautogui.press("volumeup" if step < 0 else "volumedown")
                self.status = "Volume up" if step < 0 else "Volume down"
            elif not step:
                self.status = "Volume: move hand up/down"
        elif gesture == "THREE_FINGERS":
            step = self._vertical_delta(wy, 0.02)
            if step:
                pyautogui.scroll(-step * 120)
                self.status = "Scroll down" if step > 0 else "Scroll up"
            else:
                self.status = "Scroll: move hand up/down"
        elif new_gesture:
            self._one_shot(gesture)
        elif gesture in ("OPEN_PALM", "NONE"):
            self.status = "Idle"

    # ------------------------------------------------------------------
    def _move_cursor(self, x, y):
        m = config.ACTIVE_MARGIN
        nx = float(np.clip((x - m) / (1 - 2 * m), 0.0, 1.0))
        ny = float(np.clip((y - m) / (1 - 2 * m), 0.0, 1.0))
        target = np.array([nx * (self.screen_w - 1), ny * (self.screen_h - 1)])
        self.cur = target if self.cur is None else self.cur + 0.35 * (target - self.cur)
        pyautogui.moveTo(int(self.cur[0]), int(self.cur[1]))

    def _one_shot(self, gesture):
        if not self._cool(gesture, self.ONE_SHOT_COOLDOWN):
            return
        if gesture == "THUMBS_UP":
            pyautogui.press("right")
            self.status = "Next slide"
        elif gesture == "FIST":
            pyautogui.press("left")
            self.status = "Previous slide"
        elif gesture == "THUMBS_DOWN":
            pyautogui.press("volumemute")
            self.status = "Mute / unmute"
        elif gesture == "FOUR_FINGERS":
            pyautogui.press("playpause")
            self.status = "Play / pause"
        elif gesture == "ROCK":
            folder = config.RESULTS_DIR / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"
            pyautogui.screenshot(str(path))
            self.status = f"Screenshot saved: {path.name}"
        elif gesture == "CALL_ME":
            pyautogui.hotkey("alt", "tab")
            self.status = "Switch window"
        elif gesture == "L_SHAPE":
            pyautogui.hotkey("alt", "left")
            self.status = "Browser back"
        elif gesture == "PINKY_ONLY":
            pyautogui.hotkey("win", "d")
            self.status = "Show desktop"
        elif gesture == "I_LOVE_YOU":
            pyautogui.press("f11")
            self.status = "Fullscreen toggle"
