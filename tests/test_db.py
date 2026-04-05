"""Tests for scene_db.db."""

import sqlite3
from datetime import datetime

import pytest

from scene_db.db import (
    SCHEMA_SQL,
    insert_scene_chunk,
    insert_scene_chunks,
    search_scenes,
    get_scene_by_id,
    list_all_scenes,
)
from scene_db.models import FileRef, SceneChunk


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema applied."""
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA_SQL)
    yield c
    c.close()


def _make_chunk(
    chunk_id="test_seq_000",
    dataset_name="kitti",
    sequence_id="seq",
    chunk_index=0,
    caption="vehicle moving forward",
    file_refs=None,
):
    t0 = datetime(2011, 9, 26, 13, 0, 0)
    t1 = datetime(2011, 9, 26, 13, 0, 5)
    return SceneChunk(
        id=chunk_id,
        dataset_name=dataset_name,
        sequence_id=sequence_id,
        chunk_index=chunk_index,
        start_time=t0,
        end_time=t1,
        start_frame=0,
        end_frame=49,
        avg_speed_kmh=30.0,
        distance_m=41.7,
        max_accel_ms2=1.5,
        max_decel_ms2=2.0,
        avg_yaw_rate_degs=3.5,
        max_yaw_rate_degs=8.0,
        caption=caption,
        file_refs=file_refs or [],
    )


class TestInsertAndRetrieve:
    def test_insert_single_chunk(self, conn):
        chunk = _make_chunk()
        insert_scene_chunk(conn, chunk)
        conn.commit()
        result = get_scene_by_id(conn, "test_seq_000")
        assert result is not None
        assert result.id == "test_seq_000"
        assert result.avg_speed_kmh == 30.0

    def test_insert_with_file_refs(self, conn):
        refs = [
            FileRef(scene_id="test_seq_000", file_type="image_02", frame_index=0, file_path="/img/0.png"),
            FileRef(scene_id="test_seq_000", file_type="velodyne", frame_index=0, file_path="/vel/0.bin"),
        ]
        chunk = _make_chunk(file_refs=refs)
        insert_scene_chunk(conn, chunk)
        conn.commit()
        result = get_scene_by_id(conn, "test_seq_000")
        assert len(result.file_refs) == 2
        types = {r.file_type for r in result.file_refs}
        assert types == {"image_02", "velodyne"}

    def test_insert_replaces_on_conflict(self, conn):
        chunk = _make_chunk(caption="old caption")
        insert_scene_chunk(conn, chunk)
        conn.commit()
        chunk2 = _make_chunk(caption="new caption")
        insert_scene_chunk(conn, chunk2)
        conn.commit()
        result = get_scene_by_id(conn, "test_seq_000")
        assert result.caption == "new caption"

    def test_insert_scene_chunks_batch(self, conn):
        chunks = [
            _make_chunk(chunk_id="c0", chunk_index=0),
            _make_chunk(chunk_id="c1", chunk_index=1),
            _make_chunk(chunk_id="c2", chunk_index=2),
        ]
        insert_scene_chunks(conn, chunks)
        all_scenes = list_all_scenes(conn)
        assert len(all_scenes) == 3


class TestSearch:
    def test_search_by_caption_substring(self, conn):
        insert_scene_chunks(conn, [
            _make_chunk(chunk_id="a", chunk_index=0, caption="vehicle stationary"),
            _make_chunk(chunk_id="b", chunk_index=1, caption="vehicle moving forward"),
            _make_chunk(chunk_id="c", chunk_index=2, caption="vehicle moving at high speed"),
        ])
        results = search_scenes(conn, "moving")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"b", "c"}

    def test_search_no_match(self, conn):
        insert_scene_chunks(conn, [_make_chunk()])
        results = search_scenes(conn, "nonexistent")
        assert results == []

    def test_search_empty_db(self, conn):
        results = search_scenes(conn, "anything")
        assert results == []


class TestGetSceneById:
    def test_nonexistent_returns_none(self, conn):
        assert get_scene_by_id(conn, "no_such_id") is None


class TestListAllScenes:
    def test_empty_db(self, conn):
        assert list_all_scenes(conn) == []

    def test_ordering(self, conn):
        insert_scene_chunks(conn, [
            _make_chunk(chunk_id="kitti_a_1", dataset_name="kitti", sequence_id="a", chunk_index=1),
            _make_chunk(chunk_id="kitti_a_0", dataset_name="kitti", sequence_id="a", chunk_index=0),
        ])
        scenes = list_all_scenes(conn)
        assert scenes[0].chunk_index == 0
        assert scenes[1].chunk_index == 1
