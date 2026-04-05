"""Feature extraction and caption generation."""

import math

from scene_db.models import OxtsRecord


def compute_avg_speed_kmh(records: list[OxtsRecord]) -> float:
    """Compute average speed in km/h from forward and lateral velocities."""
    if not records:
        return 0.0
    total = sum(math.sqrt(r.vf**2 + r.vl**2) for r in records)
    return (total / len(records)) * 3.6  # m/s -> km/h


def compute_distance_m(records: list[OxtsRecord]) -> float:
    """Compute total distance traveled using trapezoidal integration."""
    if len(records) < 2:
        return 0.0
    distance = 0.0
    for i in range(1, len(records)):
        dt = (records[i].timestamp - records[i - 1].timestamp).total_seconds()
        if dt <= 0:
            continue
        speed_prev = math.sqrt(records[i - 1].vf ** 2 + records[i - 1].vl ** 2)
        speed_curr = math.sqrt(records[i].vf ** 2 + records[i].vl ** 2)
        distance += 0.5 * (speed_prev + speed_curr) * dt
    return distance


def generate_caption(avg_speed_kmh: float, distance_m: float) -> str:
    """Generate a rule-based caption from features."""
    if avg_speed_kmh < 1.0:
        motion = "vehicle stationary"
    elif avg_speed_kmh < 15.0:
        motion = "vehicle moving slowly"
    elif avg_speed_kmh < 50.0:
        motion = "vehicle moving forward"
    else:
        motion = "vehicle moving at high speed"

    speed_str = f"{avg_speed_kmh:.0f} km/h"
    dist_str = f"{distance_m:.1f} m"
    return f"{motion}, {speed_str}, traveled {dist_str}"
