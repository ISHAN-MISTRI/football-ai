"""SQLite storage for tracking results."""

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class TrackingDatabase:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame INTEGER,
                    timestamp REAL,
                    track_id INTEGER,
                    class TEXT,
                    team TEXT,
                    jersey_number TEXT,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    confidence REAL,
                    session_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    source_video TEXT,
                    output_video TEXT,
                    csv_path TEXT,
                    stats_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert_rows(self, rows: list[dict[str, Any]], session_id: str) -> None:
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO tracking
                (frame, timestamp, track_id, class, team, jersey_number,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, session_id)
                VALUES (:frame, :timestamp, :track_id, :class, :team, :jersey_number,
                        :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :confidence, :session_id)""",
                [{**row, "session_id": session_id} for row in rows],
            )

    def save_session(self, session_id: str, source: str, output: str, csv_path: str, stats: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, source_video, output_video, csv_path, stats_json) VALUES (?,?,?,?,?)",
                (session_id, source, output, csv_path, stats),
            )

    def to_dataframe(self, session_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM tracking"
        params: tuple = ()
        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)
