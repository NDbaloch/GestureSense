"""Print what GestureSense saved in the database (good for the demo / report).

Usage:  python view_report.py --db postgres     (or mysql)
"""
import argparse

import config
from database import BACKENDS, connect


def table(headers, rows):
    rows = [[("" if v is None else v) for v in r] for r in rows]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    if not rows:
        print("(no data yet)")
    print()


def fmt(v, nd=1):
    return round(float(v), nd) if v is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["mysql", "postgres", "both"], required=True)
    args = ap.parse_args()

    for backend in BACKENDS[args.db]:
        print(f"\n========== {backend.upper()} - database '{config.DB_NAME}' ==========\n")
        conn = connect(backend)
        cur = conn.cursor()

        print("Last 5 sessions")
        cur.execute("SELECT id, started_at, ended_at, classifier, frames, avg_fps, avg_latency_ms "
                    "FROM sessions ORDER BY id DESC LIMIT 5")
        table(["id", "started", "ended", "classifier", "frames", "avg_fps", "latency_ms"],
              [(a, str(b)[:19], str(c)[:19] if c else None, d, e, fmt(f), fmt(g))
               for a, b, c, d, e, f, g in cur.fetchall()])

        print("Gesture counts (all sessions)")
        cur.execute("SELECT gesture, COUNT(*) FROM gesture_events GROUP BY gesture "
                    "ORDER BY COUNT(*) DESC")
        table(["gesture", "events"], cur.fetchall())

        print("Last 10 events")
        cur.execute("SELECT event_time, gesture, action, confidence FROM gesture_events "
                    "ORDER BY id DESC LIMIT 10")
        table(["time", "gesture", "action", "confidence"],
              [(str(a)[:23], b, c, fmt(d, 2)) for a, b, c, d in cur.fetchall()])

        cur.close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
