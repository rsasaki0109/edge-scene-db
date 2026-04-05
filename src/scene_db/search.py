"""Search functionality for scene-db."""

import sqlite3

from scene_db.db import search_scenes
from scene_db.models import SceneChunk


def search(conn: sqlite3.Connection, query: str) -> list[SceneChunk]:
    """Search scenes matching the query text in caption."""
    return search_scenes(conn, query)
