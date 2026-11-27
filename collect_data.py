"""Record your own gesture dataset (MediaPipe landmarks + label) into data/gestures.csv.

Keys:  1-6 choose gesture | R start/stop recording | Q quit
Tip:   record each gesture from different distances, angles, lighting and with
       different people (with consent) for a robust model.
"""
import argparse
import csv
import time
from collections import Counter

import cv2

import config
from gesture_classifier import landmarks_to_features, normalize_landmarks
from hand_tracker import HandTracker

HEADER = ["label"] + [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]


def load_counts():
    counts = Counter()
    if config.DATA_FILE.exists():
        with open(config.DATA_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--target", type=int, default=300,
                    help="stop recording automatically after N samples per gesture per session")
    ap.add_argument("--interval", type=float, default=0.05, help="seconds between samples")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    new_file = not config.DATA_FILE.exists()
    counts = load_counts()
    session = Counter()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return 1

    selected = 0
    recording = False
    last_save = 0.0

    with open(config.DATA_FILE, "a", newline="", encoding="utf-8") as f, HandTracker() as tracker:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame)
            label = config.GESTURES[selected]

            if hand is not None:
                tracker.draw(frame, hand)
                if recording and time.time() - last_save >= args.interval:
                    feats = landmarks_to_features(normalize_landmarks(hand.pixels, hand.handedness))
                    writer.writerow([label] + [f"{v:.5f}" for v in feats])
                    counts[label] += 1
                    session[label] += 1
                    last_save = time.time()
                    if session[label] >= args.target:
                        recording = False
                        f.flush()

            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"Gesture [{selected + 1}]: {label}  {'REC' if recording else 'ready'}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "1-6 select | R record/stop | Q quit", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, g in enumerate(config.GESTURES):
                mark = ">" if i == selected else " "
                cv2.putText(frame, f"{mark}{i + 1}. {g}: {counts[g]}", (10, 90 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if hand is None:
                cv2.putText(frame, "No hand detected", (10, 90 + 22 * 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Collect gesture data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                recording = not recording
            if ord("1") <= key <= ord("0") + len(config.GESTURES):
                selected = key - ord("1")
                recording = False
                f.flush()

    cap.release()
    cv2.destroyAllWindows()
    print("Saved samples per gesture:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
