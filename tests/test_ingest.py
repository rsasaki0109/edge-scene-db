"""Tests for scene_db.ingest."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scene_db.ingest import (
    parse_timestamps,
    parse_oxts_data,
    discover_files,
    split_into_chunks,
    ingest_sequence,
)
from scene_db.models import OxtsRecord


def _make_record(frame_index=0, timestamp=None, vf=0.0, vl=0.0):
    return OxtsRecord(
        timestamp=timestamp or datetime(2011, 9, 26, 13, 0, 0),
        frame_index=frame_index,
        lat=49.0, lon=8.0, alt=100.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        vf=vf, vl=vl, vu=0.0,
    )


# -- 30-field KITTI oxts data line helper --
def _oxts_line(lat=49.0, lon=8.0, alt=100.0, roll=0.0, pitch=0.0, yaw=0.0,
               vn=0.0, ve=0.0, vf=10.0, vl=0.5, vu=0.0):
    """Build a 30-field KITTI oxts data line."""
    fields = [lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu]
    # Pad remaining 19 fields with zeros
    fields.extend([0.0] * 19)
    return " ".join(str(f) for f in fields)


class TestParseTimestamps:
    def test_basic(self, tmp_path):
        ts_file = tmp_path / "timestamps.txt"
        ts_file.write_text(
            "2011-09-26 13:02:25.594360832\n"
            "2011-09-26 13:02:25.694360832\n"
        )
        result = parse_timestamps(ts_file)
        assert len(result) == 2
        assert result[0].year == 2011
        assert result[0].second == 25
        # Nanosecond truncation: .594360832 -> .594360
        assert result[0].microsecond == 594360

    def test_empty_file(self, tmp_path):
        ts_file = tmp_path / "timestamps.txt"
        ts_file.write_text("")
        assert parse_timestamps(ts_file) == []

    def test_blank_lines_skipped(self, tmp_path):
        ts_file = tmp_path / "timestamps.txt"
        ts_file.write_text(
            "2011-09-26 13:02:25.594360\n"
            "\n"
            "2011-09-26 13:02:25.694360\n"
            "\n"
        )
        result = parse_timestamps(ts_file)
        assert len(result) == 2


class TestParseOxtsData:
    def test_basic(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "0000000000.txt").write_text(_oxts_line(vf=10.0, vl=0.5))
        (data_dir / "0000000001.txt").write_text(_oxts_line(vf=12.0, vl=0.3))
        timestamps = [
            datetime(2011, 9, 26, 13, 0, 0),
            datetime(2011, 9, 26, 13, 0, 0, 100000),
        ]
        records = parse_oxts_data(data_dir, timestamps)
        assert len(records) == 2
        assert records[0].vf == 10.0
        assert records[1].vf == 12.0
        assert records[0].frame_index == 0
        assert records[1].frame_index == 1

    def test_more_files_than_timestamps(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "0000000000.txt").write_text(_oxts_line())
        (data_dir / "0000000001.txt").write_text(_oxts_line())
        timestamps = [datetime(2011, 9, 26, 13, 0, 0)]
        records = parse_oxts_data(data_dir, timestamps)
        assert len(records) == 1

    def test_insufficient_fields_skipped(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "0000000000.txt").write_text("49.0 8.0 100.0")  # only 3 fields
        timestamps = [datetime(2011, 9, 26, 13, 0, 0)]
        records = parse_oxts_data(data_dir, timestamps)
        assert len(records) == 0

    def test_empty_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        timestamps = [datetime(2011, 9, 26, 13, 0, 0)]
        records = parse_oxts_data(data_dir, timestamps)
        assert records == []


class TestDiscoverFiles:
    def test_finds_image_and_velodyne(self, tmp_path):
        seq = tmp_path / "seq"
        (seq / "image_02" / "data").mkdir(parents=True)
        (seq / "image_02" / "data" / "0000000000.png").write_text("")
        (seq / "velodyne_points" / "data").mkdir(parents=True)
        (seq / "velodyne_points" / "data" / "0000000000.bin").write_bytes(b"\x00")
        (seq / "oxts" / "data").mkdir(parents=True)
        (seq / "oxts" / "data" / "0000000000.txt").write_text(_oxts_line())
        result = discover_files(seq)
        assert "image_02" in result
        assert "velodyne" in result
        assert "oxts" in result
        assert len(result["image_02"]) == 1

    def test_empty_sequence_dir(self, tmp_path):
        seq = tmp_path / "empty_seq"
        seq.mkdir()
        result = discover_files(seq)
        assert result == {}

    def test_missing_subdirs(self, tmp_path):
        seq = tmp_path / "partial"
        (seq / "image_00" / "data").mkdir(parents=True)
        (seq / "image_00" / "data" / "0.png").write_text("")
        result = discover_files(seq)
        assert "image_00" in result
        assert "velodyne" not in result


class TestSplitIntoChunks:
    def test_empty_records(self):
        assert split_into_chunks([]) == []

    def test_single_record(self):
        records = [_make_record()]
        chunks = split_into_chunks(records, chunk_duration_sec=5.0)
        assert chunks == [(0, 0)]

    def test_all_within_one_chunk(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=i, timestamp=t0 + timedelta(seconds=i))
            for i in range(4)
        ]
        chunks = split_into_chunks(records, chunk_duration_sec=5.0)
        assert len(chunks) == 1
        assert chunks[0] == (0, 3)

    def test_exact_boundary(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=i, timestamp=t0 + timedelta(seconds=i))
            for i in range(11)
        ]
        # chunk_duration=5: first chunk 0-4, boundary at i=5 (elapsed=5), second chunk 5-9, third at i=10
        chunks = split_into_chunks(records, chunk_duration_sec=5.0)
        assert len(chunks) == 3
        assert chunks[0] == (0, 4)
        assert chunks[1] == (5, 9)
        assert chunks[2] == (10, 10)

    def test_custom_duration(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=i, timestamp=t0 + timedelta(seconds=i))
            for i in range(6)
        ]
        chunks = split_into_chunks(records, chunk_duration_sec=2.0)
        # 0-1, 2-3, 4-5
        assert len(chunks) == 3


class TestIngestSequence:
    def _create_kitti_sequence(self, base_dir, n_frames=6, interval_sec=1.0, vf=10.0):
        """Create minimal KITTI-like sequence directory."""
        seq_dir = base_dir / "2011_09_26_drive_0001_sync"
        oxts_dir = seq_dir / "oxts"
        data_dir = oxts_dir / "data"
        data_dir.mkdir(parents=True)
        img_dir = seq_dir / "image_02" / "data"
        img_dir.mkdir(parents=True)

        t0 = datetime(2011, 9, 26, 13, 0, 0)
        lines = []
        for i in range(n_frames):
            ts = t0 + timedelta(seconds=i * interval_sec)
            lines.append(ts.strftime("%Y-%m-%d %H:%M:%S.%f") + "000")  # pad to nanoseconds
            (data_dir / f"{i:010d}.txt").write_text(_oxts_line(vf=vf))
            (img_dir / f"{i:010d}.png").write_text("")

        (oxts_dir / "timestamps.txt").write_text("\n".join(lines) + "\n")
        return seq_dir

    def test_basic_ingest(self, tmp_path):
        seq_dir = self._create_kitti_sequence(tmp_path, n_frames=6, interval_sec=1.0)
        db_path = tmp_path / "test.db"
        n = ingest_sequence(seq_dir, db_path=db_path, chunk_duration_sec=5.0)
        assert n >= 1

    def test_ingest_creates_db(self, tmp_path):
        seq_dir = self._create_kitti_sequence(tmp_path)
        db_path = tmp_path / "sub" / "test.db"
        ingest_sequence(seq_dir, db_path=db_path)
        assert db_path.exists()

    def test_ingest_missing_timestamps(self, tmp_path):
        seq_dir = tmp_path / "bad_seq"
        seq_dir.mkdir()
        (seq_dir / "oxts").mkdir()
        with pytest.raises(FileNotFoundError, match="timestamps.txt"):
            ingest_sequence(seq_dir)

    def test_ingest_missing_oxts_data(self, tmp_path):
        seq_dir = tmp_path / "bad_seq2"
        oxts_dir = seq_dir / "oxts"
        oxts_dir.mkdir(parents=True)
        (oxts_dir / "timestamps.txt").write_text("2011-09-26 13:00:00.000000\n")
        with pytest.raises(FileNotFoundError, match="oxts data directory"):
            ingest_sequence(seq_dir)

    def test_ingest_multiple_chunks(self, tmp_path):
        seq_dir = self._create_kitti_sequence(tmp_path, n_frames=12, interval_sec=1.0)
        db_path = tmp_path / "test.db"
        n = ingest_sequence(seq_dir, db_path=db_path, chunk_duration_sec=5.0)
        assert n >= 2

    def test_ingest_with_file_refs(self, tmp_path):
        seq_dir = self._create_kitti_sequence(tmp_path, n_frames=3, interval_sec=1.0)
        db_path = tmp_path / "test.db"
        ingest_sequence(seq_dir, db_path=db_path, chunk_duration_sec=10.0)

        from scene_db.db import get_connection, list_all_scenes, get_scene_by_id
        conn = get_connection(db_path)
        scenes = list_all_scenes(conn)
        assert len(scenes) == 1
        scene = get_scene_by_id(conn, scenes[0].id)
        # 3 frames x 2 file types (image_02, oxts)
        assert len(scene.file_refs) > 0
        conn.close()
