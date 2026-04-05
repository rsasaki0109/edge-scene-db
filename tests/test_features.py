"""Tests for scene_db.features."""

import math
from datetime import datetime, timedelta

import pytest

from scene_db.features import (
    SceneFeatures,
    compute_acceleration,
    compute_avg_speed_kmh,
    compute_distance_m,
    compute_yaw_rate,
    extract_features,
    generate_caption,
)
from scene_db.models import OxtsRecord


def _make_record(frame_index=0, timestamp=None, vf=0.0, vl=0.0, yaw=0.0):
    return OxtsRecord(
        timestamp=timestamp or datetime(2011, 9, 26, 13, 0, 0),
        frame_index=frame_index,
        lat=49.0, lon=8.0, alt=100.0,
        roll=0.0, pitch=0.0, yaw=yaw,
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


class TestComputeAcceleration:
    def test_empty_records(self):
        assert compute_acceleration([]) == (0.0, 0.0)

    def test_single_record(self):
        assert compute_acceleration([_make_record(vf=10.0)]) == (0.0, 0.0)

    def test_constant_speed(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0),
        ]
        accel, decel = compute_acceleration(records)
        assert accel == pytest.approx(0.0)
        assert decel == pytest.approx(0.0)

    def test_accelerating(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=0.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=5.0),
        ]
        accel, decel = compute_acceleration(records)
        assert accel == pytest.approx(5.0)
        assert decel == pytest.approx(0.0)

    def test_decelerating(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=5.0),
        ]
        accel, decel = compute_acceleration(records)
        assert accel == pytest.approx(0.0)
        assert decel == pytest.approx(5.0)

    def test_mixed(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=5.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0),
            _make_record(frame_index=2, timestamp=t0 + timedelta(seconds=2), vf=3.0),
        ]
        accel, decel = compute_acceleration(records)
        assert accel == pytest.approx(5.0)
        assert decel == pytest.approx(7.0)

    def test_zero_dt_skipped(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0),
            _make_record(frame_index=1, timestamp=t0, vf=20.0),  # same timestamp
        ]
        accel, decel = compute_acceleration(records)
        assert accel == 0.0
        assert decel == 0.0


class TestComputeYawRate:
    def test_empty_records(self):
        assert compute_yaw_rate([]) == (0.0, 0.0)

    def test_single_record(self):
        assert compute_yaw_rate([_make_record(yaw=0.5)]) == (0.0, 0.0)

    def test_no_yaw_change(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=10.0, yaw=0.5),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0, yaw=0.5),
        ]
        avg, mx = compute_yaw_rate(records)
        assert avg == pytest.approx(0.0)
        assert mx == pytest.approx(0.0)

    def test_constant_yaw_rate(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        # 0.1 rad/s yaw rate for 2 intervals
        records = [
            _make_record(frame_index=0, timestamp=t0, yaw=0.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), yaw=0.1),
            _make_record(frame_index=2, timestamp=t0 + timedelta(seconds=2), yaw=0.2),
        ]
        avg, mx = compute_yaw_rate(records)
        expected_degs = math.degrees(0.1)
        assert avg == pytest.approx(expected_degs)
        assert mx == pytest.approx(expected_degs)

    def test_yaw_wrapping(self):
        """Test that yaw wrapping around pi/-pi is handled."""
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, yaw=3.1),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), yaw=-3.1),
        ]
        avg, mx = compute_yaw_rate(records)
        # Small angle difference crossing pi boundary
        expected = abs(math.degrees(math.atan2(math.sin(-3.1 - 3.1), math.cos(-3.1 - 3.1))))
        assert avg == pytest.approx(expected, abs=0.1)

    def test_zero_dt_skipped(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, yaw=0.0),
            _make_record(frame_index=1, timestamp=t0, yaw=1.0),
        ]
        avg, mx = compute_yaw_rate(records)
        assert avg == 0.0
        assert mx == 0.0


class TestExtractFeatures:
    def test_empty_records(self):
        feat = extract_features([])
        assert feat.avg_speed_kmh == 0.0
        assert feat.distance_m == 0.0
        assert feat.max_accel_ms2 == 0.0
        assert feat.max_decel_ms2 == 0.0
        assert feat.avg_yaw_rate_degs == 0.0
        assert feat.max_yaw_rate_degs == 0.0

    def test_returns_scene_features(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=0.0, yaw=0.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=10.0, yaw=0.1),
            _make_record(frame_index=2, timestamp=t0 + timedelta(seconds=2), vf=5.0, yaw=0.3),
        ]
        feat = extract_features(records)
        assert isinstance(feat, SceneFeatures)
        assert feat.avg_speed_kmh > 0
        assert feat.distance_m > 0
        assert feat.max_accel_ms2 > 0
        assert feat.max_decel_ms2 > 0
        assert feat.avg_yaw_rate_degs > 0
        assert feat.max_yaw_rate_degs > 0

    def test_consistent_with_individual_functions(self):
        t0 = datetime(2011, 9, 26, 13, 0, 0)
        records = [
            _make_record(frame_index=0, timestamp=t0, vf=5.0),
            _make_record(frame_index=1, timestamp=t0 + timedelta(seconds=1), vf=15.0),
        ]
        feat = extract_features(records)
        assert feat.avg_speed_kmh == pytest.approx(compute_avg_speed_kmh(records))
        assert feat.distance_m == pytest.approx(compute_distance_m(records))
        accel, decel = compute_acceleration(records)
        assert feat.max_accel_ms2 == pytest.approx(accel)
        assert feat.max_decel_ms2 == pytest.approx(decel)
        avg_yr, max_yr = compute_yaw_rate(records)
        assert feat.avg_yaw_rate_degs == pytest.approx(avg_yr)
        assert feat.max_yaw_rate_degs == pytest.approx(max_yr)


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

    def test_sharp_turn(self):
        caption = generate_caption(30.0, 100.0, max_yaw_rate_degs=35.0)
        assert "sharp turn" in caption

    def test_turning(self):
        caption = generate_caption(30.0, 100.0, avg_yaw_rate_degs=12.0)
        assert "turning" in caption

    def test_gentle_curve(self):
        caption = generate_caption(30.0, 100.0, avg_yaw_rate_degs=5.0)
        assert "gentle curve" in caption

    def test_hard_braking(self):
        caption = generate_caption(30.0, 100.0, max_decel_ms2=5.0)
        assert "hard braking" in caption

    def test_braking(self):
        caption = generate_caption(30.0, 100.0, max_decel_ms2=3.0)
        assert "braking" in caption
        assert "hard" not in caption

    def test_no_turning_or_braking_by_default(self):
        caption = generate_caption(30.0, 100.0)
        assert "turn" not in caption
        assert "braking" not in caption
        assert "curve" not in caption

    def test_combined_turning_and_braking(self):
        caption = generate_caption(30.0, 100.0, max_decel_ms2=5.0, max_yaw_rate_degs=35.0)
        assert "sharp turn" in caption
        assert "hard braking" in caption
