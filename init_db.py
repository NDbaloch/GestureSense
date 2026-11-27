"""Create the 'gesturesense' database and its tables, and test the connection.

Usage:  python init_db.py --db postgres
        python init_db.py --db mysql
        python init_db.py --db both
"""
import argparse

import config
from database import BACKENDS, create_database, create_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["mysql", "postgres", "both"], required=True)
    args = ap.parse_args()

    ok = True
    for backend in BACKENDS[args.db]:
        try:
            create_database(backend)
            create_tables(backend)
            print(f"[OK] {backend}: database '{config.DB_NAME}' and tables are ready.")
        except Exception as exc:
            ok = False
            print(f"[FAILED] {backend}: {exc}")
            print("  - Is the server running?  - Are the user/password/port in config.py correct?")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
