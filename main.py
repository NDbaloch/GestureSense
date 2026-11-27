"""GestureSense - real-time contactless HCI using hand gestures (15 gestures).

Run:  python main.py                 (rule-based classifier)
      python main.py --db postgres   (also log sessions/events to PostgreSQL; or mysql / both)
      python main.py --classifier ml (trained ML classifier, needs 15-class model)
      python main.py --no-actions    (display only, does not control the PC)
Keys: P = pause/resume actions, Q or ESC = quit
"""
import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

import config
from actions import ActionController
from database import DatabaseLogger
from gesture_classifier import (GestureSmoother, MLGestureClassifier,
                                classify_rule, normalize_landmarks)
from hand_tracker import HandTracker

LEGEND = [
    "POINT (index)        : move cursor",
    "PINCH (thumb+index)  : left click",
    "OK SIGN              : right click",
    "THUMBS UP            : next slide",
    "FIST                 : previous slide",
    "THUMBS DOWN          : mute / unmute",
    "PEACE + hand up/down : volume",
    "THREE FINGERS + move : scroll",
    "FOUR FINGERS         : play / pause",
    "ROCK (index+pinky)   : screenshot",
    "CALL ME (thumb+pinky): alt+tab",
    "L SHAPE              : browser back",
    "PINKY ONLY           : show desktop",
    "I LOVE YOU           : fullscreen (F11)",
    "OPEN PALM            : idle",
    "P: pause   Q: quit",
]
LINE_H = 17


def parse_args():
    ap = argparse.ArgumentParser(description="GestureSense - hand gesture control")
    ap.add_argument("--classifier", choices=["rule", "ml"], default="rule")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    ap.add_argument("--db", choices=["none", "mysql", "postgres", "both"],
                    default=config.DB_BACKEND, help="log sessions and gesture events to a database")
    ap.add_argument("--no-actions", action="store_true", help="do not control the computer")
    return ap.parse_args()


def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(frame, gesture, conf_text, fps, latency, status, classifier_name):
    h, w = frame.shape[:2]
    m = config.ACTIVE_MARGIN
    cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)),
                  (255, 200, 0), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.rectangle(overlay, (0, h - LINE_H * len(LEGEND) - 10), (300, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture} {conf_text}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Action: {status}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"FPS {fps:.0f} | {latency:.0f} ms | {classifier_name}", (w - 250, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, line in enumerate(LEGEND):
        cv2.putText(frame, line, (6, h - LINE_H * (len(LEGEND) - i) + 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)


def save_summary(fps_list, lat_list, classifier_name):
    if not fps_list:
        return
    text = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}] classifier={classifier_name} "
            f"frames={len(fps_list)} avg_fps={np.mean(fps_list):.1f} "
            f"avg_latency_ms={np.mean(lat_list):.1f} max_latency_ms={np.max(lat_list):.1f}")
    print(text)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "performance_log.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    args = parse_args()
    ml = MLGestureClassifier() if args.classifier == "ml" else None

    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        print("Could not open the webcam. Try --camera 1 or close other apps using it.")
        return 1

    tracker = HandTracker()
    smoother = GestureSmoother()
    actions = ActionController(enabled=not args.no_actions)
    db = DatabaseLogger.open(args.db, "ml" if ml else "rules")
    last_logged = None
    fps_list, lat_list = [], []
    last_time = time.perf_counter()
    fps = 0.0

    print("GestureSense running (15 gestures). Press Q (in the video window) to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received.")
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural

            t0 = time.perf_counter()
            hand = tracker.process(frame)
            conf_text = ""
            conf = None
            if hand is not None:
                norm = normalize_landmarks(hand.pixels, hand.handedness)
                if ml is not None:
                    raw, conf = ml.predict(norm)
                    conf_text = f"({conf:.0%})"
                else:
                    raw = classify_rule(norm)
                gesture = smoother.push(raw)
                actions.update(gesture, hand.points)
                if gesture != last_logged and gesture in config.GESTURES:
                    db.log_event(gesture, actions.status, conf)
                    last_logged = gesture
                tracker.draw(frame, hand)
            else:
                gesture = smoother.push("NONE")
                actions.reset()
                last_logged = None
            latency = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6)) if fps else 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_list.append(fps)
            lat_list.append(latency)

            draw_hud(frame, gesture, conf_text, fps, latency, actions.status,
                     "ML" if ml else "RULES")
            cv2.imshow("GestureSense", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                actions.enabled = not actions.enabled
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        save_summary(fps_list, lat_list, "ml" if ml else "rules")
        if fps_list:
            db.close(len(fps_list), np.mean(fps_list), np.mean(lat_list))
        else:
            db.close(0, 0.0, 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
