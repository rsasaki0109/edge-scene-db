"""Export scene data to a directory."""

import shutil
import sqlite3
from pathlib import Path

from scene_db.db import get_scene_by_id


def export_scene(conn: sqlite3.Connection, scene_id: str, output_dir: Path) -> int:
    """Export all files for a scene to the output directory.

    Returns the number of files copied.
    """
    chunk = get_scene_by_id(conn, scene_id)
    if chunk is None:
        raise ValueError(f"Scene not found: {scene_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for ref in chunk.file_refs:
        src = Path(ref.file_path)
        if not src.exists():
            continue
        # Preserve structure: <output_dir>/<file_type>/<filename>
        type_dir = output_dir / ref.file_type
        type_dir.mkdir(exist_ok=True)
        dst = type_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    # Write scene metadata
    meta_file = output_dir / "scene_info.txt"
    meta_file.write_text(
        f"Scene ID: {chunk.id}\n"
        f"Dataset: {chunk.dataset_name}\n"
        f"Sequence: {chunk.sequence_id}\n"
        f"Frames: {chunk.start_frame}-{chunk.end_frame}\n"
        f"Time: {chunk.start_time.isoformat()} - {chunk.end_time.isoformat()}\n"
        f"Speed: {chunk.avg_speed_kmh:.1f} km/h\n"
        f"Distance: {chunk.distance_m:.1f} m\n"
        f"Max accel: {chunk.max_accel_ms2:.2f} m/s^2\n"
        f"Max decel: {chunk.max_decel_ms2:.2f} m/s^2\n"
        f"Avg yaw rate: {chunk.avg_yaw_rate_degs:.1f} deg/s\n"
        f"Max yaw rate: {chunk.max_yaw_rate_degs:.1f} deg/s\n"
        f"Caption: {chunk.caption}\n"
    )

    return copied
