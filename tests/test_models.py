"""Tests for scene_db.models."""

from datetime import datetime

from scene_db.models import FileRef, OxtsRecord, SceneChunk


class TestOxtsRecord:
    def test_creation(self):
        ts = datetime(2011, 9, 26, 13, 2, 25)
        rec = OxtsRecord(
            timestamp=ts, frame_index=0,
            lat=49.0, lon=8.0, alt=100.0,
            roll=0.01, pitch=0.02, yaw=0.03,
            vf=10.0, vl=0.5, vu=0.0,
        )
        assert rec.timestamp == ts
        assert rec.frame_index == 0
        assert rec.vf == 10.0

    def test_zero_velocities(self):
        rec = OxtsRecord(
            timestamp=datetime(2011, 1, 1), frame_index=0,
            lat=0, lon=0, alt=0, roll=0, pitch=0, yaw=0,
            vf=0, vl=0, vu=0,
        )
        assert rec.vf == 0.0
        assert rec.vl == 0.0


class TestFileRef:
    def test_creation(self):
        ref = FileRef(
            scene_id="scene_001", file_type="image_02",
            frame_index=5, file_path="/data/img/000005.png",
        )
        assert ref.scene_id == "scene_001"
        assert ref.file_type == "image_02"
        assert ref.frame_index == 5
        assert ref.file_path == "/data/img/000005.png"


class TestSceneChunk:
    def test_creation_with_defaults(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        t1 = datetime(2011, 9, 26, 13, 0, 5)
        chunk = SceneChunk(
            id="kitti_seq_000", dataset_name="kitti",
            sequence_id="seq", chunk_index=0,
            start_time=t0, end_time=t1,
            start_frame=0, end_frame=49,
        )
        assert chunk.avg_speed_kmh == 0.0
        assert chunk.distance_m == 0.0
        assert chunk.max_accel_ms2 == 0.0
        assert chunk.max_decel_ms2 == 0.0
        assert chunk.avg_yaw_rate_degs == 0.0
        assert chunk.max_yaw_rate_degs == 0.0
        assert chunk.caption == ""
        assert chunk.file_refs == []

    def test_creation_with_all_fields(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        t1 = datetime(2011, 9, 26, 13, 0, 5)
        ref = FileRef(scene_id="id1", file_type="velodyne", frame_index=0, file_path="/v/0.bin")
        chunk = SceneChunk(
            id="id1", dataset_name="kitti",
            sequence_id="drive_0001", chunk_index=0,
            start_time=t0, end_time=t1,
            start_frame=0, end_frame=10,
            avg_speed_kmh=36.0, distance_m=50.0,
            max_accel_ms2=2.5, max_decel_ms2=3.0,
            avg_yaw_rate_degs=5.0, max_yaw_rate_degs=15.0,
            caption="vehicle moving forward",
            file_refs=[ref],
        )
        assert chunk.avg_speed_kmh == 36.0
        assert chunk.max_accel_ms2 == 2.5
        assert chunk.max_decel_ms2 == 3.0
        assert chunk.avg_yaw_rate_degs == 5.0
        assert chunk.max_yaw_rate_degs == 15.0
        assert len(chunk.file_refs) == 1
        assert chunk.file_refs[0].file_type == "velodyne"
