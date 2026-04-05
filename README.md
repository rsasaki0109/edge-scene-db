# edge-scene-db

**Find the edge cases hiding in your driving logs.**

edge-scene-db ingests autonomous driving and robotics log data, chunks it by time, extracts motion features, and automatically detects localization/perception edge cases.

## Supported Datasets

| Format | Sensors | Command |
|---|---|---|
| **KITTI** | Camera, LiDAR, GPS/IMU | `scene-db ingest /path/to/drive_sync/` |
| **nuScenes** | 6 cameras, LiDAR, RADAR | `scene-db ingest /path/ --dataset-name nuscenes` |
| **rosbag** (GLIM, FAST-LIO, LIO-SAM, etc.) | LiDAR + IMU | `scene-db ingest file.bag` |
| **PPC Dataset** | GNSS + IMU | `scene-db ingest /path/PPC-Dataset/` |

Auto-detection: `.bag` files → rosbag, `oxts/` → KITTI, `v1.0-mini/` → nuScenes, `reference.csv` → PPC.

## Installation

```bash
pip install -e .                    # Core
pip install -e ".[rosbag]"          # + rosbag support (GLIM, FAST-LIO, etc.)
pip install -e ".[embedding]"       # + semantic search
pip install -e ".[vlm]"             # + VLM captioning (OpenAI)
pip install -e ".[all]"             # Everything
```

## Quick Start

```bash
# 1. Ingest data (auto-detects format)
scene-db ingest /path/to/kitti_drive/
scene-db ingest recording.bag
scene-db ingest /path/PPC-Dataset/

# 2. See what you have
scene-db stats

# 3. Detect edge cases automatically
scene-db edge-cases
scene-db edge-cases -c localization --severity critical

# 4. Search with filters
scene-db search "turning"
scene-db search --min-yaw 20 --sort yaw
scene-db search --min-decel 2.0 --sort decel
scene-db search --max-speed 5

# 5. Export a scene
scene-db export --id <scene_id> -o ./output/
```

## Edge Case Detection

`scene-db edge-cases` automatically flags scenes that stress localization or perception systems:

| Category | Rule | Threshold | Severity |
|---|---|---|---|
| **Localization** | High yaw rate | > 20 deg/s | critical |
| **Localization** | High speed (GPS latency) | > 60 km/h | warning/critical |
| **Localization** | Near-zero speed (GPS noise) | < 3 km/h | warning |
| **Localization** | Start from stop | accel + stationary | warning |
| **Both** | Yaw + decel combined | yaw > 10 + decel > 1.0 | critical |
| **Perception** | Hard braking (pitch shift) | > 3.0 m/s² | critical |
| **Perception** | Decel to stop (tracking handoff) | decel > 1 + slow | warning |

```bash
# Filter by category and severity
scene-db edge-cases -c localization --severity critical -n 20
scene-db edge-cases -c perception -n 10
```

## Feature Extraction

Each 5-second scene chunk gets:

| Feature | Description | Edge case relevance |
|---|---|---|
| `avg_speed_kmh` | Average speed | GPS latency, LiDAR distortion |
| `distance_m` | Distance traveled | Dead-reckoning drift |
| `max_accel_ms2` | Peak acceleration | Sensor dynamics |
| `max_decel_ms2` | Peak deceleration | Pitch shift, FOV change |
| `avg_yaw_rate_degs` | Average heading change rate | IMU bias, wheel slip |
| `max_yaw_rate_degs` | Peak heading change rate | EKF heading stress |

Captions auto-generated with keywords: `stationary`, `moving slowly`, `moving forward`, `high speed`, `turning`, `sharp turn`, `gentle curve`, `braking`, `hard braking`.

## CLI Reference

| Command | Description |
|---|---|
| `scene-db ingest <path>` | Ingest dataset (auto-detects format) |
| `scene-db edge-cases` | Detect localization/perception edge cases |
| `scene-db search <query>` | Search by caption text + feature filters |
| `scene-db stats` | Show database statistics |
| `scene-db index [--embed]` | Show index / build embeddings |
| `scene-db export --id <id>` | Export scene files |

### Key Options

```
scene-db ingest:
  --dataset-name TEXT       auto, kitti, nuscenes, rosbag, ppc
  --chunk-duration FLOAT    Chunk duration in seconds [default: 5.0]
  --vlm                     Use VLM captioning (requires OPENAI_API_KEY)
  --imu-topic TEXT          IMU topic for rosbag
  --odom-topic TEXT         Odometry topic for rosbag

scene-db search:
  --min-speed / --max-speed Filter by speed (km/h)
  --min-decel               Filter by deceleration (m/s²)
  --min-yaw                 Filter by yaw rate (deg/s)
  --sort                    Sort by: speed, decel, yaw, accel
  -s, --semantic            Semantic search (requires embeddings)

scene-db edge-cases:
  -c, --category            localization, perception, both
  --severity                critical, warning, info
  -n                        Max results [default: 20]
```

## Tested Datasets

| Source | Type | Sensors | Scenes | Link |
|---|---|---|---|---|
| KITTI (25 seq) | Vehicle | LiDAR+Camera+GPS/IMU | 147 | [kitti](https://www.cvlibs.net/datasets/kitti/raw_data.php) |
| nuScenes mini | Vehicle | 6cam+LiDAR+RADAR | 40 | [nuscenes](https://www.nuscenes.org/download) |
| GLIM (Ouster) | Handheld | OS1-128 + IMU | 23 | [glim](https://github.com/koide3/glim) |
| Cartographer 3D | Backpack | 2x VLP-16 + IMU | 243 | [cartographer](https://google-cartographer-ros.readthedocs.io/) |
| PPC Dataset | Vehicle | GNSS + IMU | 2354 | [ppc](https://github.com/taroz/PPC-Dataset) |
| **Total** | | | **2807** | **97,447 frames** |

## Architecture

```
Raw Data (KITTI / nuScenes / rosbag / PPC)
  → Ingest & Chunk (5-sec time windows)
    → Feature Extraction (speed, distance, yaw rate, acceleration)
      → Captioning (rule-based or VLM)
        → SQLite Storage
          → Edge Case Detection (localization / perception rules)
          → Search (text, filters, or semantic embedding)
            → Export
```

## Development

```bash
git clone https://github.com/rsasaki0109/edge-scene-db.git
cd edge-scene-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,rosbag]"
pytest  # 120 tests
```

## License

MIT
