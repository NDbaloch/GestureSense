"""MySQL / PostgreSQL logging for GestureSense.

Tables
  sessions        one row per run of main.py (start/end time, classifier, frames, avg FPS, latency)
  gesture_events  one row each time the recognised gesture changes (gesture, action, confidence)
"""
import queue
import threading
from datetime import datetime

import config

SCHEMA = {
    "mysql": [
        """CREATE TABLE IF NOT EXISTS sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            started_at DATETIME NOT NULL,
            ended_at DATETIME NULL,
            classifier VARCHAR(20) NOT NULL,
            frames INT DEFAULT 0,
            avg_fps FLOAT DEFAULT 0,
            avg_latency_ms FLOAT DEFAULT 0
        ) ENGINE=InnoDB""",
        """CREATE TABLE IF NOT EXISTS gesture_events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT NOT NULL,
            event_time DATETIME(3) NOT NULL,
            gesture VARCHAR(20) NOT NULL,
            action VARCHAR(60),
            confidence FLOAT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        ) ENGINE=InnoDB""",
    ],
    "postgres": [
        """CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            classifier VARCHAR(20) NOT NULL,
            frames INTEGER DEFAULT 0,
            avg_fps REAL DEFAULT 0,
            avg_latency_ms REAL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS gesture_events (
            id SERIAL PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            event_time TIMESTAMP NOT NULL,
            gesture VARCHAR(20) NOT NULL,
            action VARCHAR(60),
            confidence REAL
        )""",
    ],
}

BACKENDS = {"none": [], "mysql": ["mysql"], "postgres": ["postgres"], "both": ["mysql", "postgres"]}


def connect(backend, use_database=True):
    """Open a connection. use_database=False connects to the server only (to create the DB)."""
    if backend == "mysql":
        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError("Run: pip install mysql-connector-python") from exc
        cfg = config.MYSQL
        kwargs = dict(host=cfg["host"], port=cfg["port"], user=cfg["user"], password=cfg["password"])
        if use_database:
            kwargs["database"] = config.DB_NAME
        return mysql.connector.connect(**kwargs)

    if backend == "postgres":
        try:
            import psycopg2
        except ImportError as exc:
            raise RuntimeError("Run: pip install psycopg2-binary") from exc
        cfg = config.POSTGRES
        return psycopg2.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                                password=cfg["password"],
                                dbname=config.DB_NAME if use_database else "postgres")
    raise ValueError(f"Unknown backend: {backend}")


def create_database(backend):
    conn = connect(backend, use_database=False)
    try:
        cur = conn.cursor()
        if backend == "mysql":
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` CHARACTER SET utf8mb4")
            conn.commit()
        else:
            conn.autocommit = True
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{config.DB_NAME}"')
        cur.close()
    finally:
        conn.close()


def create_tables(backend):
    conn = connect(backend)
    try:
        cur = conn.cursor()
        for stmt in SCHEMA[backend]:
            cur.execute(stmt)
        conn.commit()
        cur.close()
    finally:
        conn.close()


class GestureDB:
    """Logs one run of the app into ONE database. Inserts happen in a background thread
    so the video loop never waits for the database."""

    def __init__(self, backend, classifier):
        self.backend = backend
        self.conn = connect(backend)
        self.session_id = self._start_session(classifier)
        self._queue = queue.Queue()
        self._warned = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _start_session(self, classifier):
        cur = self.conn.cursor()
        now = datetime.now()
        if self.backend == "postgres":
            cur.execute("INSERT INTO sessions (started_at, classifier) VALUES (%s, %s) RETURNING id",
                        (now, classifier))
            session_id = cur.fetchone()[0]
        else:
            cur.execute("INSERT INTO sessions (started_at, classifier) VALUES (%s, %s)",
                        (now, classifier))
            session_id = cur.lastrowid
        self.conn.commit()
        cur.close()
        return session_id

    def log_event(self, gesture, action, confidence=None):
        conf = None if confidence is None else float(confidence)
        self._queue.put((datetime.now(), gesture, (action or "")[:60], conf))

    def _run(self):
        cur = self.conn.cursor()
        sql = ("INSERT INTO gesture_events (session_id, event_time, gesture, action, confidence) "
               "VALUES (%s, %s, %s, %s, %s)")
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                cur.execute(sql, (self.session_id,) + item)
                self.conn.commit()
            except Exception as exc:
                if not self._warned:
                    print(f"[db:{self.backend}] insert failed: {exc}")
                    self._warned = True
                try:
                    self.conn.rollback()
                except Exception:
                    pass
        cur.close()

    def close(self, frames, avg_fps, avg_latency_ms):
        self._queue.put(None)
        self._worker.join(timeout=5)
        try:
            cur = self.conn.cursor()
            cur.execute("UPDATE sessions SET ended_at=%s, frames=%s, avg_fps=%s, avg_latency_ms=%s "
                        "WHERE id=%s",
                        (datetime.now(), int(frames), float(avg_fps), float(avg_latency_ms),
                         self.session_id))
            self.conn.commit()
            cur.close()
        except Exception as exc:
            print(f"[db:{self.backend}] could not save session summary: {exc}")
        finally:
            self.conn.close()


class DatabaseLogger:
    """Fan-out wrapper: logs to zero, one or both databases. Never crashes the app."""

    def __init__(self, dbs):
        self.dbs = dbs

    @classmethod
    def open(cls, spec, classifier):
        dbs = []
        for backend in BACKENDS.get((spec or "none").lower(), []):
            try:
                dbs.append(GestureDB(backend, classifier))
                print(f"[db] Connected to {backend} (database '{config.DB_NAME}')")
            except Exception as exc:
                print(f"[db] Could not connect to {backend}: {exc}")
                print(f"     Check the password in config.py and run: python init_db.py --db {backend}")
        return cls(dbs)

    @property
    def active(self):
        return bool(self.dbs)

    def log_event(self, gesture, action, confidence=None):
        for db in self.dbs:
            db.log_event(gesture, action, confidence)

    def close(self, frames, avg_fps, avg_latency_ms):
        for db in self.dbs:
            db.close(frames, avg_fps, avg_latency_ms)
