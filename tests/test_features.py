"""Tests for scene_db.features."""

import math
from datetime import datetime, timedelta

import pytest

from scene_db.features import compute_avg_speed_kmh, compute_distance_m, generate_caption
from scene_db.models import OxtsRecord


def _make_record(frame_index=0, timestamp=None, vf=0.0, vl=0.0):
    return OxtsRecord(
        timestamp=timestamp or datetime(2011, 9, 26, 13, 0, 0),
        frame_index=frame_index,
        lat=49.0, lon=8.0, alt=100.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        vf=vf, vl=vl, vu=0.0,
    )


class TestComputeAvgSpeedKmh:
    def test_empty_records(self):
        assert compute_avg_speed_kmh([]) == 0.0

    def test_single_record_stationary(self):
        assert compute_avg_speed_kmh([_make_record(vf=0.0)]) == 0.0

    def test_single_record_moving(self):
        # 10 m/s forward -> 36 km/h
        result = compute_avg_speed_kmh([_make_record(vf=10.0)])
        assert result == pytest.approx(36.0)

    def test_forward_only(self):
        records = [_make_record(vf=10.0), _make_record(vf=10.0)]
        assert compute_avg_speed_kmh(records) == pytest.approx(36.0)

    def test_lateral_velocity_contributes(self):
        # vf=3, vl=4 -> speed=5 m/s -> 18 km/h
        records = [_make_record(vf=3.0, vl=4.0)]
        assert compute_avg_speed_kmh(records) == pytest.approx(18.0)

    def test_varying_speeds(self):
        records = [_make_record(vf=0.0), _make_record(vf=10.0)]
        # avg speed = (0 + 10) / 2 = 5 m/s = 18 km/h
        assert compute_avg_speed_kmh(records) == pytest.approx(18.0)


class TestComputeDistanceM:
    def test_empty_records(self):
        assert compute_distance_m([]) == 0.0

    def test_single_record(self):
        assert compute_distance_m([_make_record(vf=10.0)]) == 0.0

    def test_constant_speed(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0),
        ]
        # 10 m/s for 1 second = 10 m
        assert compute_distance_m(records) == pytest.approx(10.0)

    def test_acceleration(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=0.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=2), vf=10.0),
        ]
        # trapezoidal: 0.5 * (0 + 10) * 2 = 10 m
        assert compute_distance_m(records) == pytest.approx(10.0)

    def test_zero_dt_skipped(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0, vf=10.0),  # same timestamp
        ]
        assert compute_distance_m(records) == 0.0

    def test_multiple_segments(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0),
            _make_record(frame_index=2, timestamp=t0 + timedelta(seconds=2), vf=10.0),
        ]
        # 10 m/s for 2 seconds = 20 m
        assert compute_distance_m(records) == pytest.approx(20.0)

    def test_lateral_velocity_contributes(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=3.0, vl=4.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=3.0, vl=4.0),
        ]
        # speed = 5 m/s, dt = 1 -> 5 m
        assert compute_distance_m(records) == pytest.approx(5.0)


class TestGenerateCaption:
    def test_stationary(self):
        caption = generate_caption(0.0, 0.0)
        assert "stationary" in caption
        assert "0 km/h" in caption

    def test_slow(self):
        caption = generate_caption(10.0, 5.0)
        assert "slowly" in caption

    def test_forward(self):
        caption = generate_caption(30.0, 100.0)
        assert "moving forward" in caption

    def test_high_speed(self):
        caption = generate_caption(80.0, 500.0)
        assert "high speed" in caption

    def test_boundary_1kmh(self):
        # Exactly 1.0 -> "moving slowly" (not stationary)
        caption = generate_caption(1.0, 1.0)
        assert "slowly" in caption

    def test_boundary_15kmh(self):
        caption = generate_caption(15.0, 20.0)
        assert "moving forward" in caption

    def test_boundary_50kmh(self):
        caption = generate_caption(50.0, 200.0)
        assert "high speed" in caption

    def test_distance_formatting(self):
        caption = generate_caption(30.0, 123.456)
        assert "123.5 m" in caption
