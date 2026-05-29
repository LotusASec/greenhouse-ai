"""Initialize the SQLite database by running schema.sql.

Usage: python database/init.py
Reads DB_PATH from environment. Idempotent — safe to run multiple times.
"""

import logging
import os
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("db-init")


def init_db() -> None:
    db_path = os.getenv("DB_PATH")
    if not db_path:
        raise EnvironmentError("DB_PATH environment variable is not set")

    schema_file = Path(__file__).parent / "schema.sql"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to database at %s", db_path)
    conn = sqlite3.connect(db_path)
    try:
        sql = schema_file.read_text()
        conn.executescript(sql)
        conn.commit()
        log.info("Schema applied successfully — database ready")
    except sqlite3.Error as exc:
        log.error("Failed to apply schema: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
