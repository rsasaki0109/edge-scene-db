"""SQLite database operations for scene-db."""

import sqlite3
from datetime import datetime
from pathlib import Path

from scene_db.models import FileRef, SceneChunk

DEFAULT_DB_DIR = Path.home() / ".scene-db"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "scene.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scene_chunks (
    id              TEXT PRIMARY KEY,
    dataset_name    TEXT NOT NULL,
    sequence_id     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    start_time      TEXT NOT NULL,
    end_time        TEXT NOT NULL,
    start_frame     INTEGER NOT NULL,
    end_frame       INTEGER NOT NULL,
    avg_speed_kmh   REAL DEFAULT 0.0,
    distance_m      REAL DEFAULT 0.0,
    max_accel_ms2   REAL DEFAULT 0.0,
    max_decel_ms2   REAL DEFAULT 0.0,
    avg_yaw_rate_degs REAL DEFAULT 0.0,
    max_yaw_rate_degs REAL DEFAULT 0.0,
    caption         TEXT DEFAULT '',
    UNIQUE(dataset_name, sequence_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS file_refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id    TEXT NOT NULL REFERENCES scene_chunks(id),
    file_type   TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    file_path   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_refs_scene ON file_refs(scene_id);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the database and ensure schema exists."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    return conn


def insert_scene_chunk(conn: sqlite3.Connection, chunk: SceneChunk) -> None:
    """Insert a SceneChunk and its file refs into the database."""
    conn.execute(
        """INSERT OR REPLACE INTO scene_chunks
           (id, dataset_name, sequence_id, chunk_index,
            start_time, end_time, start_frame, end_frame,
            avg_speed_kmh, distance_m,
            max_accel_ms2, max_decel_ms2,
            avg_yaw_rate_degs, max_yaw_rate_degs,
            caption)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk.id,
            chunk.dataset_name,
            chunk.sequence_id,
            chunk.chunk_index,
            chunk.start_time.isoformat(),
            chunk.end_time.isoformat(),
            chunk.start_frame,
            chunk.end_frame,
            chunk.avg_speed_kmh,
            chunk.distance_m,
            chunk.max_accel_ms2,
            chunk.max_decel_ms2,
            chunk.avg_yaw_rate_degs,
            chunk.max_yaw_rate_degs,
            chunk.caption,
        ),
    )
    # Remove old file refs for this scene, then re-insert
    conn.execute("DELETE FROM file_refs WHERE scene_id = ?", (chunk.id,))
    for ref in chunk.file_refs:
        conn.execute(
            """INSERT INTO file_refs (scene_id, file_type, frame_index, file_path)
               VALUES (?, ?, ?, ?)""",
            (ref.scene_id, ref.file_type, ref.frame_index, ref.file_path),
        )


def insert_scene_chunks(conn: sqlite3.Connection, chunks: list[SceneChunk]) -> None:
    """Insert multiple SceneChunks in a single transaction."""
    for chunk in chunks:
        insert_scene_chunk(conn, chunk)
    conn.commit()


def search_scenes(conn: sqlite3.Connection, query: str) -> list[SceneChunk]:
    """Search scenes by caption text (LIKE match)."""
    cursor = conn.execute(
        """SELECT id, dataset_name, sequence_id, chunk_index,
                  start_time, end_time, start_frame, end_frame,
                  avg_speed_kmh, distance_m,
                  max_accel_ms2, max_decel_ms2,
                  avg_yaw_rate_degs, max_yaw_rate_degs,
                  caption
           FROM scene_chunks
           WHERE caption LIKE ?
           ORDER BY dataset_name, sequence_id, chunk_index""",
        (f"%{query}%",),
    )
    return [_row_to_chunk(row) for row in cursor.fetchall()]


def get_scene_by_id(conn: sqlite3.Connection, scene_id: str) -> SceneChunk | None:
    """Fetch a single scene chunk by ID, including file refs."""
    cursor = conn.execute(
        """SELECT id, dataset_name, sequence_id, chunk_index,
                  start_time, end_time, start_frame, end_frame,
                  avg_speed_kmh, distance_m,
                  max_accel_ms2, max_decel_ms2,
                  avg_yaw_rate_degs, max_yaw_rate_degs,
                  caption
           FROM scene_chunks WHERE id = ?""",
        (scene_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    chunk = _row_to_chunk(row)
    chunk.file_refs = _get_file_refs(conn, scene_id)
    return chunk


def list_all_scenes(conn: sqlite3.Connection) -> list[SceneChunk]:
    """List all scene chunks."""
    cursor = conn.execute(
        """SELECT id, dataset_name, sequence_id, chunk_index,
                  start_time, end_time, start_frame, end_frame,
                  avg_speed_kmh, distance_m,
                  max_accel_ms2, max_decel_ms2,
                  avg_yaw_rate_degs, max_yaw_rate_degs,
                  caption
           FROM scene_chunks
           ORDER BY dataset_name, sequence_id, chunk_index"""
    )
    return [_row_to_chunk(row) for row in cursor.fetchall()]


def _row_to_chunk(row: tuple) -> SceneChunk:
    return SceneChunk(
        id=row[0],
        dataset_name=row[1],
        sequence_id=row[2],
        chunk_index=row[3],
        start_time=datetime.fromisoformat(row[4]),
        end_time=datetime.fromisoformat(row[5]),
        start_frame=row[6],
        end_frame=row[7],
        avg_speed_kmh=row[8],
        distance_m=row[9],
        max_accel_ms2=row[10],
        max_decel_ms2=row[11],
        avg_yaw_rate_degs=row[12],
        max_yaw_rate_degs=row[13],
        caption=row[14],
    )


def _get_file_refs(conn: sqlite3.Connection, scene_id: str) -> list[FileRef]:
    cursor = conn.execute(
        """SELECT scene_id, file_type, frame_index, file_path
           FROM file_refs WHERE scene_id = ?
           ORDER BY file_type, frame_index""",
        (scene_id,),
    )
    return [
        FileRef(
            scene_id=row[0],
            file_type=row[1],
            frame_index=row[2],
            file_path=row[3],
        )
        for row in cursor.fetchall()
    ]
