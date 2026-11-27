"""Train and evaluate the gesture classifier on data/gestures.csv.

Compares k-NN and Random Forest (and the rule-based classifier) on a held-out test set,
saves the best ML model to models/gesture_model.joblib and a report to results/.
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import config
from gesture_classifier import classify_rule

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def main():
    if not config.DATA_FILE.exists():
        print("No dataset found. Run collect_data.py first.")
        return 1

    df = pd.read_csv(config.DATA_FILE)
    counts = df["label"].value_counts()
    log("Samples per gesture:")
    log(counts.to_string())
    if len(counts) < 2 or counts.min() < 10:
        print("\nNeed at least 2 gestures with 10+ samples each (aim for 300+ per gesture).")
        return 1

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log(f"\nTrain: {len(X_tr)} samples | Test: {len(X_te)} samples")

    models = {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        pred = model.predict(X_te)
        per_sample_ms = (time.perf_counter() - t0) * 1000 / len(X_te)
        acc = accuracy_score(y_te, pred)
        results[name] = (acc, model, pred)
        log(f"\n=== {name} ===")
        log(f"Accuracy: {acc:.4f} | inference: {per_sample_ms:.3f} ms/sample")
        log(classification_report(y_te, pred, zero_division=0))
        labels = sorted(set(y))
        log("Confusion matrix (rows = true, cols = predicted): " + str(labels))
        log(str(confusion_matrix(y_te, pred, labels=labels)))

    rule_pred = [classify_rule(x.reshape(21, 2)) for x in X_te]
    log(f"\n=== Rule-based classifier ===\nAccuracy: {accuracy_score(y_te, rule_pred):.4f}")

    best_name = max(results, key=lambda n: results[n][0])
    log(f"\nBest ML model: {best_name} ({results[best_name][0]:.4f})")
    log("Note: consecutive video frames are very similar, so this score is optimistic. "
        "For an honest result, also test on a person who was not in the training data.")

    final = models[best_name].fit(X, y)   # refit on all data for deployment
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "training_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved model  -> {config.MODEL_FILE}")
    print(f"Saved report -> {config.RESULTS_DIR / 'training_report.txt'}")
    print("Run it with:  python main.py --classifier ml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
