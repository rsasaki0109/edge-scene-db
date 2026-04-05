"""Search functionality for scene-db."""

import sqlite3

from scene_db.db import search_scenes
from scene_db.models import SceneChunk


def search(
    conn: sqlite3.Connection,
    query: str = "",
    min_speed: float | None = None,
    max_speed: float | None = None,
    min_decel: float | None = None,
    min_yaw: float | None = None,
    min_accel: float | None = None,
    sort_by: str | None = None,
) -> list[SceneChunk]:
    """Search scenes matching the query text and/or feature filters."""
    return search_scenes(
        conn, query,
        min_speed=min_speed,
        max_speed=max_speed,
        min_decel=min_decel,
        min_yaw=min_yaw,
        min_accel=min_accel,
        sort_by=sort_by,
    )
