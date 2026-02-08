import json
import os
import sqlite3
from typing import Any, Dict, Optional


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    gcs_uri TEXT NOT NULL,
    transcript TEXT,
    system_prompt TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str) -> None:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()


def get_video_row(db_path: str, video_id: str) -> Dict[str, Any]:
    if not video_id:
        raise ValueError("video_id is required")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT video_id, gcs_uri, transcript, system_prompt, metadata_json "
            "FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"video_id not found in DB: {video_id}")
    return dict(row)


def upsert_video_row(db_path: str, row: Dict[str, Any]) -> None:
    init_db(db_path)
    video_id = row.get("video_id")
    gcs_uri = row.get("gcs_uri")
    if not video_id:
        raise ValueError("row.video_id is required")
    if not gcs_uri:
        raise ValueError("row.gcs_uri is required")

    transcript = row.get("transcript")
    system_prompt = row.get("system_prompt")
    metadata_json = row.get("metadata_json")

    if isinstance(metadata_json, (dict, list)):
        metadata_json = json.dumps(metadata_json, ensure_ascii=False)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO videos (video_id, gcs_uri, transcript, system_prompt, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                gcs_uri = excluded.gcs_uri,
                transcript = excluded.transcript,
                system_prompt = excluded.system_prompt,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (video_id, gcs_uri, transcript, system_prompt, metadata_json),
        )
        conn.commit()
