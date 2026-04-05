"""Tests for scene_db.ingest_nuscenes."""

import json
from pathlib import Path

import pytest

from scene_db.ingest_nuscenes import _ego_pose_to_speed, ingest_nuscenes


class TestEgoPoseToSpeed:
    def test_single_pose(self):
        poses = [{"translation": [0, 0, 0]}]
        speeds = _ego_pose_to_speed(poses, [1000000])
        assert speeds == [0.0]

    def test_two_poses_stationary(self):
        poses = [
            {"translation": [0, 0, 0]},
            {"translation": [0, 0, 0]},
        ]
        speeds = _ego_pose_to_speed(poses, [1000000, 2000000])
        assert speeds[0] == 0.0
        assert speeds[1] == 0.0

    def test_two_poses_moving(self):
        # Move 3m in x, 4m in y over 1 second -> 5 m/s
        poses = [
            {"translation": [0, 0, 0]},
            {"translation": [3, 4, 0]},
        ]
        speeds = _ego_pose_to_speed(poses, [1000000, 2000000])
        assert speeds[0] == 0.0
        assert speeds[1] == pytest.approx(5.0)

    def test_zero_dt_returns_zero(self):
        poses = [
            {"translation": [0, 0, 0]},
            {"translation": [10, 0, 0]},
        ]
        # Same timestamp
        speeds = _ego_pose_to_speed(poses, [1000000, 1000000])
        assert speeds[1] == 0.0

    def test_multiple_poses(self):
        poses = [
            {"translation": [0, 0, 0]},
            {"translation": [10, 0, 0]},
            {"translation": [30, 0, 0]},
        ]
        # 1 second intervals (1e6 us apart)
        timestamps = [1000000, 2000000, 3000000]
        speeds = _ego_pose_to_speed(poses, timestamps)
        assert len(speeds) == 3
        assert speeds[0] == 0.0
        assert speeds[1] == pytest.approx(10.0)
        assert speeds[2] == pytest.approx(20.0)


def _create_nuscenes_mini(base_dir: Path, n_samples: int = 4, interval_us: int = 500000):
    """Create a minimal nuScenes directory structure for testing."""
    meta_dir = base_dir / "v1.0-mini"
    meta_dir.mkdir(parents=True)
    samples_dir = base_dir / "samples" / "CAM_FRONT"
    samples_dir.mkdir(parents=True)

    # Create sample image files
    for i in range(n_samples):
        (samples_dir / f"frame_{i:04d}.jpg").write_text("fake image")

    # Build ego poses
    ego_poses = []
    for i in range(n_samples):
        ego_poses.append({
            "token": f"ego_{i:04d}",
            "translation": [float(i * 5), 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": 1000000 + i * interval_us,
        })

    # Build sample_data (one per sample, for CAM_FRONT)
    sample_data = []
    for i in range(n_samples):
        sample_data.append({
            "token": f"sd_{i:04d}",
            "sample_token": f"sample_{i:04d}",
            "ego_pose_token": f"ego_{i:04d}",
            "filename": f"samples/CAM_FRONT/frame_{i:04d}.jpg",
            "channel": "CAM_FRONT",
            "timestamp": 1000000 + i * interval_us,
        })

    # Build samples (linked list via next)
    samples = []
    for i in range(n_samples):
        samples.append({
            "token": f"sample_{i:04d}",
            "timestamp": 1000000 + i * interval_us,
            "data": {"CAM_FRONT": f"sd_{i:04d}"},
            "next": f"sample_{i + 1:04d}" if i < n_samples - 1 else "",
        })

    # Build scene
    scenes = [{
        "token": "scene_0001",
        "name": "scene-0001",
        "first_sample_token": "sample_0000",
        "nbr_samples": n_samples,
    }]

    # Write JSON files
    (meta_dir / "scene.json").write_text(json.dumps(scenes))
    (meta_dir / "sample.json").write_text(json.dumps(samples))
    (meta_dir / "sample_data.json").write_text(json.dumps(sample_data))
    (meta_dir / "ego_pose.json").write_text(json.dumps(ego_poses))

    return base_dir


class TestIngestNuscenes:
    def test_basic_ingest(self, tmp_path):
        dataroot = _create_nuscenes_mini(tmp_path / "nuscenes", n_samples=4)
        db_path = tmp_path / "test.db"
        n = ingest_nuscenes(dataroot, db_path=db_path, chunk_duration_sec=10.0)
        assert n >= 1

    def test_ingest_creates_db(self, tmp_path):
        dataroot = _create_nuscenes_mini(tmp_path / "nuscenes")
        db_path = tmp_path / "sub" / "test.db"
        ingest_nuscenes(dataroot, db_path=db_path)
        assert db_path.exists()

    def test_ingest_with_chunks(self, tmp_path):
        # 8 samples, 0.5s apart = 4s total; chunk at 2s -> 2+ chunks
        dataroot = _create_nuscenes_mini(
            tmp_path / "nuscenes", n_samples=8, interval_us=500000
        )
        db_path = tmp_path / "test.db"
        n = ingest_nuscenes(dataroot, db_path=db_path, chunk_duration_sec=2.0)
        assert n >= 2

    def test_ingest_scene_data_correct(self, tmp_path):
        dataroot = _create_nuscenes_mini(tmp_path / "nuscenes", n_samples=4)
        db_path = tmp_path / "test.db"
        ingest_nuscenes(dataroot, db_path=db_path, chunk_duration_sec=100.0)

        from scene_db.db import get_connection, list_all_scenes
        conn = get_connection(db_path)
        scenes = list_all_scenes(conn)
        assert len(scenes) == 1
        assert scenes[0].dataset_name == "nuscenes"
        assert "scene-0001" in scenes[0].sequence_id
        assert scenes[0].file_refs == []  # list_all_scenes doesn't load refs
        conn.close()

    def test_ingest_missing_table(self, tmp_path):
        meta_dir = tmp_path / "bad" / "v1.0-mini"
        meta_dir.mkdir(parents=True)
        # No JSON files
        with pytest.raises(FileNotFoundError):
            ingest_nuscenes(tmp_path / "bad")

    def test_empty_scene(self, tmp_path):
        """Scene with no samples produces zero chunks."""
        meta_dir = tmp_path / "empty" / "v1.0-mini"
        meta_dir.mkdir(parents=True)
        scenes = [{"token": "s1", "name": "empty-scene", "first_sample_token": "nonexistent", "nbr_samples": 0}]
        (meta_dir / "scene.json").write_text(json.dumps(scenes))
        (meta_dir / "sample.json").write_text(json.dumps([]))
        (meta_dir / "sample_data.json").write_text(json.dumps([]))
        (meta_dir / "ego_pose.json").write_text(json.dumps([]))
        n = ingest_nuscenes(tmp_path / "empty", chunk_duration_sec=5.0)
        assert n == 0
