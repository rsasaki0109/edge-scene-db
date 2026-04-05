"""KITTI dataset ingestion: parsing, chunk splitting, and DB insertion."""

from datetime import datetime
from pathlib import Path

from scene_db.db import insert_scene_chunks, get_connection
from scene_db.features import compute_avg_speed_kmh, compute_distance_m, generate_caption
from scene_db.models import FileRef, OxtsRecord, SceneChunk

# KITTI oxts field indices
_OXTS_FIELDS = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu",
    "pos_accuracy", "vel_accuracy",
    "navstat", "numsats", "posmode", "velmode", "orimode",
]


def parse_timestamps(timestamps_file: Path) -> list[datetime]:
    """Parse KITTI timestamps.txt file."""
    timestamps = []
    for line in timestamps_file.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # KITTI format: "2011-09-26 13:02:25.594360832"
        # Python can't parse nanoseconds, truncate to microseconds
        if "." in line:
            base, frac = line.rsplit(".", 1)
            frac = frac[:6]  # truncate to microseconds
            line = f"{base}.{frac}"
        timestamps.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f"))
    return timestamps


def parse_oxts_data(oxts_data_dir: Path, timestamps: list[datetime]) -> list[OxtsRecord]:
    """Parse all oxts data files in a directory."""
    records = []
    data_files = sorted(oxts_data_dir.glob("*.txt"))
    for i, data_file in enumerate(data_files):
        if i >= len(timestamps):
            break
        values = data_file.read_text().strip().split()
        if len(values) < 11:
            continue
        records.append(
            OxtsRecord(
                timestamp=timestamps[i],
                frame_index=i,
                lat=float(values[0]),
                lon=float(values[1]),
                alt=float(values[2]),
                roll=float(values[3]),
                pitch=float(values[4]),
                yaw=float(values[5]),
                vf=float(values[8]),
                vl=float(values[9]),
                vu=float(values[10]),
            )
        )
    return records


def discover_files(sequence_dir: Path) -> dict[str, list[Path]]:
    """Discover available data files in a KITTI sequence directory."""
    file_map: dict[str, list[Path]] = {}
    for subdir in ["image_00", "image_01", "image_02", "image_03"]:
        data_dir = sequence_dir / subdir / "data"
        if data_dir.exists():
            file_map[subdir] = sorted(data_dir.glob("*"))
    velodyne_dir = sequence_dir / "velodyne_points" / "data"
    if velodyne_dir.exists():
        file_map["velodyne"] = sorted(velodyne_dir.glob("*.bin"))
    oxts_dir = sequence_dir / "oxts" / "data"
    if oxts_dir.exists():
        file_map["oxts"] = sorted(oxts_dir.glob("*.txt"))
    return file_map


def split_into_chunks(
    records: list[OxtsRecord], chunk_duration_sec: float = 5.0
) -> list[tuple[int, int]]:
    """Split records into time-based chunks. Returns list of (start_idx, end_idx) inclusive."""
    if not records:
        return []
    chunks = []
    chunk_start = 0
    for i in range(1, len(records)):
        elapsed = (records[i].timestamp - records[chunk_start].timestamp).total_seconds()
        if elapsed >= chunk_duration_sec:
            chunks.append((chunk_start, i - 1))
            chunk_start = i
    # Final chunk (include remaining frames)
    if chunk_start < len(records):
        chunks.append((chunk_start, len(records) - 1))
    return chunks


def _extract_sequence_id(sequence_dir: Path) -> str:
    """Extract sequence ID from directory name."""
    return sequence_dir.name


def ingest_sequence(
    sequence_dir: Path,
    dataset_name: str = "kitti",
    chunk_duration_sec: float = 5.0,
    db_path: Path | None = None,
) -> int:
    """Ingest a single KITTI sequence into the database. Returns number of chunks created."""
    # Find oxts data
    oxts_dir = sequence_dir / "oxts"
    timestamps_file = oxts_dir / "timestamps.txt"
    oxts_data_dir = oxts_dir / "data"

    if not timestamps_file.exists():
        raise FileNotFoundError(f"timestamps.txt not found: {timestamps_file}")
    if not oxts_data_dir.exists():
        raise FileNotFoundError(f"oxts data directory not found: {oxts_data_dir}")

    timestamps = parse_timestamps(timestamps_file)
    records = parse_oxts_data(oxts_data_dir, timestamps)

    if not records:
        return 0

    file_map = discover_files(sequence_dir)
    sequence_id = _extract_sequence_id(sequence_dir)
    chunk_ranges = split_into_chunks(records, chunk_duration_sec)

    scene_chunks = []
    for chunk_idx, (start_idx, end_idx) in enumerate(chunk_ranges):
        chunk_records = records[start_idx : end_idx + 1]
        avg_speed = compute_avg_speed_kmh(chunk_records)
        distance = compute_distance_m(chunk_records)
        caption = generate_caption(avg_speed, distance)

        chunk_id = f"{dataset_name}_{sequence_id}_{chunk_idx:03d}"

        # Collect file references for this chunk's frame range
        file_refs = []
        for frame_idx in range(start_idx, end_idx + 1):
            for file_type, file_list in file_map.items():
                if frame_idx < len(file_list):
                    file_refs.append(
                        FileRef(
                            scene_id=chunk_id,
                            file_type=file_type,
                            frame_index=frame_idx,
                            file_path=str(file_list[frame_idx]),
                        )
                    )

        scene_chunks.append(
            SceneChunk(
                id=chunk_id,
                dataset_name=dataset_name,
                sequence_id=sequence_id,
                chunk_index=chunk_idx,
                start_time=chunk_records[0].timestamp,
                end_time=chunk_records[-1].timestamp,
                start_frame=start_idx,
                end_frame=end_idx,
                avg_speed_kmh=avg_speed,
                distance_m=distance,
                caption=caption,
                file_refs=file_refs,
            )
        )

    conn = get_connection(db_path)
    try:
        insert_scene_chunks(conn, scene_chunks)
    finally:
        conn.close()

    return len(scene_chunks)
