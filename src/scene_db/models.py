"""Data models for scene-db."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OxtsRecord:
    """Single KITTI oxts measurement."""

    timestamp: datetime
    frame_index: int
    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    vf: float  # forward velocity (m/s)
    vl: float  # leftward velocity (m/s)
    vu: float  # upward velocity (m/s)


@dataclass
class FileRef:
    """Reference to a file belonging to a scene."""

    scene_id: str
    file_type: str  # 'image_00', 'image_02', 'velodyne', 'oxts'
    frame_index: int
    file_path: str


@dataclass
class SceneChunk:
    """A time-bounded segment of a driving sequence."""

    id: str  # e.g. "kitti_2011_09_26_drive_0001_sync_003"
    dataset_name: str
    sequence_id: str
    chunk_index: int
    start_time: datetime
    end_time: datetime
    start_frame: int
    end_frame: int
    avg_speed_kmh: float = 0.0
    distance_m: float = 0.0
    max_accel_ms2: float = 0.0  # max longitudinal acceleration (m/s^2)
    max_decel_ms2: float = 0.0  # max longitudinal deceleration (m/s^2, positive = braking)
    avg_yaw_rate_degs: float = 0.0  # average yaw rate (deg/s)
    max_yaw_rate_degs: float = 0.0  # peak yaw rate (deg/s)
    caption: str = ""
    file_refs: list[FileRef] = field(default_factory=list)
