"""Tests for scene_db.export."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from scene_db.db import SCHEMA_SQL, insert_scene_chunk
from scene_db.export import export_scene
from scene_db.models import FileRef, SceneChunk


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA_SQL)
    yield c
    c.close()


def _make_chunk(scene_id="scene_001", file_refs=None):
    t0 = datetime(2011, 9, 26, 13, 0, 0)
    t1 = datetime(2011, 9, 26, 13, 0, 5)
    return SceneChunk(
        id=scene_id,
        dataset_name="kitti",
        sequence_id="drive_0001",
        chunk_index=0,
        start_time=t0,
        end_time=t1,
        start_frame=0,
        end_frame=2,
        avg_speed_kmh=30.0,
        distance_m=41.7,
        caption="vehicle moving forward, 30 km/h, traveled 41.7 m",
        file_refs=file_refs or [],
    )


class TestExportScene:
    def test_export_nonexistent_scene(self, conn, tmp_path):
        with pytest.raises(ValueError, match="Scene not found"):
            export_scene(conn, "no_such_scene", tmp_path / "out")

    def test_export_with_no_files(self, conn, tmp_path):
        chunk = _make_chunk()
        insert_scene_chunk(conn, chunk)
        conn.commit()
        out = tmp_path / "out"
        n = export_scene(conn, "scene_001", out)
        assert n == 0
        # Metadata file should still be written
        meta = out / "scene_info.txt"
        assert meta.exists()
        content = meta.read_text()
        assert "scene_001" in content
        assert "30.0 km/h" in content

    def test_export_copies_existing_files(self, conn, tmp_path):
        # Create source files
        src_dir = tmp_path / "src"
        (src_dir / "image_02").mkdir(parents=True)
        (src_dir / "velodyne").mkdir(parents=True)
        img = src_dir / "image_02" / "0000000000.png"
        img.write_text("fake image data")
        vel = src_dir / "velodyne" / "0000000000.bin"
        vel.write_bytes(b"\x00\x01\x02")

        refs = [
            FileRef(scene_id="scene_001", file_type="image_02", frame_index=0, file_path=str(img)),
            FileRef(scene_id="scene_001", file_type="velodyne", frame_index=0, file_path=str(vel)),
        ]
        chunk = _make_chunk(file_refs=refs)
        insert_scene_chunk(conn, chunk)
        conn.commit()

        out = tmp_path / "export_out"
        n = export_scene(conn, "scene_001", out)
        assert n == 2
        assert (out / "image_02" / "0000000000.png").exists()
        assert (out / "velodyne" / "0000000000.bin").exists()
        assert (out / "image_02" / "0000000000.png").read_text() == "fake image data"

    def test_export_skips_missing_files(self, conn, tmp_path):
        refs = [
            FileRef(scene_id="scene_001", file_type="image_02", frame_index=0,
                    file_path="/nonexistent/path/img.png"),
        ]
        chunk = _make_chunk(file_refs=refs)
        insert_scene_chunk(conn, chunk)
        conn.commit()

        out = tmp_path / "out"
        n = export_scene(conn, "scene_001", out)
        assert n == 0

    def test_export_creates_output_dir(self, conn, tmp_path):
        chunk = _make_chunk()
        insert_scene_chunk(conn, chunk)
        conn.commit()
        out = tmp_path / "nested" / "deep" / "out"
        export_scene(conn, "scene_001", out)
        assert out.exists()

    def test_metadata_content(self, conn, tmp_path):
        chunk = _make_chunk()
        insert_scene_chunk(conn, chunk)
        conn.commit()
        out = tmp_path / "out"
        export_scene(conn, "scene_001", out)
        content = (out / "scene_info.txt").read_text()
        assert "Scene ID: scene_001" in content
        assert "Dataset: kitti" in content
        assert "Sequence: drive_0001" in content
        assert "Frames: 0-2" in content
        assert "Caption: vehicle moving forward" in content
