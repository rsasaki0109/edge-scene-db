# scene-db

**Search and extract scenes from autonomous driving and robotics log data.**

scene-db splits driving logs into time-based scene chunks, extracts features, and lets you search and export specific scenes via a simple CLI.

## Features

- KITTI raw dataset ingestion
- Time-based scene chunking (configurable duration)
- Automatic feature extraction (speed, distance)
- Rule-based scene captioning
- Text search across scene captions
- Scene export to directory

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Ingest a KITTI sequence

```bash
scene-db ingest /path/to/2011_09_26_drive_0001_sync
```

This parses oxts (GPS/IMU) data, splits the sequence into 5-second chunks, computes speed/distance features, and stores everything in a local SQLite database (`~/.scene-db/scene.db`).

### 2. Search for scenes

```bash
scene-db search "moving forward"
```

```
Found 2 scene(s):

  [kitti_2011_09_26_drive_0001_sync_000]
    vehicle moving forward, 26 km/h, traveled 32.6 m
    frames 0-9, 2011-09-26T13:02:25 - 2011-09-26T13:02:29.500000

  [kitti_2011_09_26_drive_0001_sync_001]
    vehicle moving forward, 44 km/h, traveled 55.1 m
    frames 10-19, 2011-09-26T13:02:30 - 2011-09-26T13:02:34.500000
```

### 3. Export a scene

```bash
scene-db export --id kitti_2011_09_26_drive_0001_sync_000 -o ./my_scene
```

Copies all associated files (images, point clouds, oxts data) into the output directory with a `scene_info.txt` metadata file.

### 4. Check index status

```bash
scene-db index
```

## CLI Reference

| Command | Description |
|---|---|
| `scene-db ingest <path>` | Ingest a KITTI sequence directory |
| `scene-db index` | Show index status |
| `scene-db search <query>` | Search scenes by caption text |
| `scene-db export --id <id>` | Export scene files to a directory |

### Options

```
scene-db ingest:
  --dataset-name TEXT    Dataset name [default: kitti]
  --chunk-duration FLOAT Chunk duration in seconds [default: 5.0]
  --db PATH              Database path [default: ~/.scene-db/scene.db]

scene-db export:
  --id TEXT              Scene chunk ID (required)
  -o, --output PATH      Output directory [default: ./export]
  --db PATH              Database path
```

## KITTI Data Format

scene-db expects KITTI raw data in the standard directory structure:

```
<sequence_dir>/
├── oxts/
│   ├── timestamps.txt
│   └── data/
│       ├── 0000000000.txt
│       ├── 0000000001.txt
│       └── ...
├── image_00/data/
├── image_01/data/
├── image_02/data/
├── image_03/data/
└── velodyne_points/data/
```

Download KITTI raw data from: https://www.cvlibs.net/datasets/kitti/raw_data.php

## Architecture

```
SceneChunk
├── id: "kitti_{sequence}_{chunk_index}"
├── features: avg_speed_kmh, distance_m
├── caption: rule-based text description
└── file_refs: [image, pointcloud, oxts files]
```

- **Storage**: SQLite (`~/.scene-db/scene.db`)
- **Chunking**: Fixed-length time windows (default 5 seconds)
- **Search**: SQL LIKE on caption text

## Development

```bash
git clone https://github.com/your-username/scene-db.git
cd scene-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## License

MIT
