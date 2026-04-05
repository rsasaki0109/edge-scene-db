# edge-scene-db

**Find the edge cases hiding in your driving logs.**

edge-scene-db ingests autonomous driving and robotics log data, chunks it by time, extracts motion features, and automatically detects localization/perception edge cases. Tested on 2912 scenes across 9 datasets.

## Supported Formats

| Format | Sensors | Command |
|---|---|---|
| **KITTI** | Camera, LiDAR, GPS/IMU | `scene-db ingest /path/to/drive_sync/` |
| **nuScenes** | 6 cameras, LiDAR, RADAR | `scene-db ingest /path/ --dataset-name nuscenes` |
| **rosbag (ROS1)** | LiDAR + IMU | `scene-db ingest file.bag` |
| **rosbag2 (ROS2)** | LiDAR + IMU | `scene-db ingest rosbag2_dir/` |
| **PPC Dataset** | GNSS + IMU | `scene-db ingest /path/PPC-Dataset/` |

Auto-detection: `.bag` → ROS1, directory with `metadata.yaml` → ROS2, `oxts/` → KITTI, `v1.0-mini/` → nuScenes, `reference.csv` → PPC.

## Installation

```bash
pip install -e .                    # Core (KITTI, nuScenes, PPC)
pip install -e ".[rosbag]"          # + rosbag/rosbag2 (GLIM, FAST-LIO, etc.)
pip install -e ".[viz]"             # + visualization (matplotlib)
pip install -e ".[embedding]"       # + semantic search
pip install -e ".[vlm]"             # + VLM captioning (OpenAI)
pip install -e ".[all]"             # Everything
```

## Quick Start

```bash
# 1. Ingest data (auto-detects format)
scene-db ingest /path/to/kitti_drive/
scene-db ingest recording.bag             # ROS1
scene-db ingest rosbag2_recording/        # ROS2
scene-db ingest /path/PPC-Dataset/

# 2. See what you have
scene-db stats
scene-db sequences                        # distance, loops, revisits

# 3. Detect edge cases automatically
scene-db edge-cases
scene-db edge-cases -c localization --severity critical

# 4. Search with filters
scene-db search "turning"
scene-db search --min-yaw 20 --sort yaw
scene-db search --min-decel 2.0 --sort decel
scene-db search --max-speed 5

# 5. Visualize
scene-db visualize -o ./plots

# 6. Export a scene
scene-db export --id <scene_id> -o ./output/
```

## Edge Case Detection

`scene-db edge-cases` automatically flags scenes that stress localization or perception:

| Category | Rule | Threshold | Severity |
|---|---|---|---|
| **Localization** | High yaw rate | > 20 deg/s | critical |
| **Localization** | High speed (GPS latency) | > 60 km/h | warning/critical |
| **Localization** | Near-zero speed (GPS noise) | < 3 km/h | warning |
| **Localization** | Start from stop (GNSS reacq.) | accel + stationary | warning |
| **Localization** | LiDAR degeneration | low dynamics | warning |
| **Localization** | IMU drift | speed > 200 km/h | critical |
| **Both** | Yaw + decel combined | yaw > 10 + decel > 1.0 | critical |
| **Perception** | Hard braking (pitch shift) | > 3.0 m/s² | critical |
| **Perception** | Decel to stop (tracking handoff) | decel > 1 + slow | warning |

## Visualization

```bash
scene-db visualize -o ./plots
```

Generates:
- **feature_histograms.png** - Speed, deceleration, yaw rate distributions + category bar chart
- **trajectory_*.png** - 2D XY trajectory with time gradient, start/end markers, loop closure detection
- **edge_case_summary.png** - Severity and category breakdown

## CLI Reference

| Command | Description |
|---|---|
| `scene-db ingest <path>` | Ingest dataset (auto-detects format) |
| `scene-db edge-cases` | Detect localization/perception edge cases |
| `scene-db search <query>` | Search by caption text + feature filters |
| `scene-db stats` | Show database statistics |
| `scene-db sequences` | Sequence analysis (distance, loops, revisits) |
| `scene-db visualize` | Generate plots (histograms, trajectories, edge cases) |
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

scene-db visualize:
  -o, --output-dir          Output directory [default: ./plots]
  --loop-threshold          Loop detection threshold in meters [default: 10.0]
```

## Tested Datasets

| Source | Type | Sensors | Scenes | Key Edge Cases | Link |
|---|---|---|---|---|---|
| KITTI (25 seq) | Vehicle | LiDAR+Camera+GPS/IMU | 147 | yaw 30 deg/s, 77 km/h | [kitti](https://www.cvlibs.net/datasets/kitti/raw_data.php) |
| nuScenes mini | Vehicle | 6cam+LiDAR+RADAR | 40 | - | [nuscenes](https://www.nuscenes.org/download) |
| GLIM (Ouster) | Handheld | OS1-128 + IMU | 23 | IMU drift 588 km/h | [glim](https://github.com/koide3/glim) |
| Cartographer 3D | Backpack | 2x VLP-16 + IMU | 243 | 20min drift test | [cartographer](https://google-cartographer-ros.readthedocs.io/) |
| PPC Dataset | Vehicle | GNSS + IMU | 2354 | Loop closure, urban canyon | [ppc](https://github.com/taroz/PPC-Dataset) |
| AIST Park | Vehicle | Ouster OS1 + IMU | 29 | decel 11.2 m/s² | [zenodo](https://zenodo.org/records/6836915) |
| Flatwall | Handheld | Livox + IMU | 7 | LiDAR degeneration | [zenodo](https://zenodo.org/records/7641866) |
| AlienGo | Quadruped | Livox + Camera + IMU | 69 | decel 29693, yaw 45118 | [zenodo](https://zenodo.org/records/6787389) |
| **Total** | | | **2912** | | **168,073 frames** |

## LiDAR SLAM Validation Guide

| Test Case | Recommended Data | Why |
|---|---|---|
| **Sanity check** | GLIM os1_128 (491 MB) | Small Ouster bag, easy to run |
| **Aggressive dynamics** | AIST Park (2.1 GB) | decel 11.2 m/s², hard braking |
| **LiDAR degeneration** | Flatwall (306 MB) | Wall-only, scan matching fails |
| **Long-term drift** | Cartographer 3D (9.3 GB) | 20 minutes, IMU drift visible |
| **Loop closure** | PPC Tokyo run1/run2 | 10 km loop, 1386 revisits, RTK truth |
| **Urban canyon** | PPC Nagoya/Tokyo | GNSS multipath, signal blockage |
| **High yaw rate** | KITTI drive_0014 / PPC | 30+ deg/s intersection turns |
| **Quadruped walking** | AlienGo (774 MB) | 29693 m/s² IMU, LVIO stress test |

### Sequence Analysis

```bash
scene-db sequences
```

```
ppc/tokyo_run1     9.9 km  40 min  ✓ 2m   1386 revisits
ppc/tokyo_run2     6.9 km  30 min  ✓ 1m   1663 revisits
ppc/nagoya_run2    4.6 km  32 min  1698m   1156 revisits
```

## Architecture

```
Raw Data (KITTI / nuScenes / rosbag / rosbag2 / PPC)
  -> Ingest & Chunk (5-sec time windows)
    -> Feature Extraction (speed, distance, yaw rate, acceleration)
      -> Captioning (rule-based or VLM)
        -> SQLite Storage
          -> Edge Case Detection (loc / per rules)
          -> Sequence Analysis (distance, loops, revisits)
          -> Search (text, filters, or semantic)
          -> Visualize (histograms, trajectories, edge cases)
            -> Export
```

## Development

```bash
git clone https://github.com/rsasaki0109/edge-scene-db.git
cd edge-scene-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,rosbag,viz]"
pytest  # 120 tests
```

## License

MIT
