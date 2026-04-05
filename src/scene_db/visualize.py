"""Visualization module for scene-db: histograms, trajectories, edge-case summaries."""

from __future__ import annotations

from pathlib import Path

from scene_db.edge_detect import EdgeCase
from scene_db.models import SceneChunk


def plot_feature_histograms(scenes: list[SceneChunk], output_path: str | Path) -> Path:
    """Generate a 2x2 grid of feature histograms and save as PNG.

    Panels:
      - Speed distribution (km/h)
      - Max deceleration distribution (m/s^2)
      - Max yaw rate distribution (deg/s)
      - Scene category bar chart

    Returns the resolved output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)

    speeds = [s.avg_speed_kmh for s in scenes]
    decels = [s.max_decel_ms2 for s in scenes]
    yaw_rates = [s.max_yaw_rate_degs for s in scenes]

    # Extract categories from captions
    keywords = [
        "stationary",
        "moving slowly",
        "moving forward",
        "high speed",
        "braking",
        "hard braking",
        "turning",
        "sharp turn",
        "gentle curve",
    ]
    categories: dict[str, int] = {}
    for s in scenes:
        for kw in keywords:
            if kw in s.caption:
                categories[kw] = categories.get(kw, 0) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Scene Feature Overview ({len(scenes)} scenes)", fontsize=14)

    # Speed histogram
    ax = axes[0, 0]
    ax.hist(speeds, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Average Speed (km/h)")
    ax.set_ylabel("Count")
    ax.set_title("Speed Distribution")

    # Deceleration histogram
    ax = axes[0, 1]
    ax.hist(decels, bins=20, color="#DD8452", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Max Deceleration (m/s\u00b2)")
    ax.set_ylabel("Count")
    ax.set_title("Max Deceleration Distribution")

    # Yaw rate histogram
    ax = axes[1, 0]
    ax.hist(yaw_rates, bins=20, color="#55A868", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Max Yaw Rate (deg/s)")
    ax.set_ylabel("Count")
    ax.set_title("Max Yaw Rate Distribution")

    # Category bar chart
    ax = axes[1, 1]
    if categories:
        sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
        labels = [c[0] for c in sorted_cats]
        counts = [c[1] for c in sorted_cats]
        ax.barh(labels, counts, color="#8172B3", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Count")
        ax.set_title("Scene Categories")
        ax.invert_yaxis()
    else:
        ax.text(
            0.5,
            0.5,
            "No categories detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Scene Categories")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def plot_trajectory(
    positions: list[tuple[float, float]],
    output_path: str | Path,
    title: str = "Trajectory",
    loop_threshold_m: float = 10.0,
) -> Path:
    """Plot a 2D XY trajectory with time-gradient colouring.

    Args:
        positions: list of (x, y) in metres (local frame).
        output_path: destination PNG path.
        title: plot title.
        loop_threshold_m: distance threshold to consider start~=end a loop.

    Returns the resolved output path.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    output_path = Path(output_path)

    if len(positions) < 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(
            0.5,
            0.5,
            "Insufficient position data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Build line segments coloured by normalised time index
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap="viridis", linewidth=2)
    lc.set_array(t)
    ax.add_collection(lc)

    # Start / end markers
    ax.plot(xs[0], ys[0], "o", color="green", markersize=10, label="Start", zorder=5)
    ax.plot(xs[-1], ys[-1], "s", color="red", markersize=10, label="End", zorder=5)

    # Loop closure indicator
    loop_dist = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    if loop_dist < loop_threshold_m:
        ax.annotate(
            f"Loop closed ({loop_dist:.1f} m)",
            xy=(xs[-1], ys[-1]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"),
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    cbar = fig.colorbar(lc, ax=ax, label="Time (normalised)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["start", "25%", "50%", "75%", "end"])

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def plot_edge_case_summary(edge_cases: list[EdgeCase], output_path: str | Path) -> Path:
    """Plot edge-case summary: severity and category breakdowns.

    Args:
        edge_cases: list of EdgeCase objects from edge_detect.
        output_path: destination PNG path.

    Returns the resolved output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)

    if not edge_cases:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No edge cases detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Edge Case Summary")
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path

    # Severity counts
    severity_order = ["critical", "warning", "info"]
    severity_colors = {"critical": "#D64541", "warning": "#F5AB35", "info": "#4ECDC4"}
    severity_counts = {s: 0 for s in severity_order}
    for c in edge_cases:
        severity_counts[c.severity] = severity_counts.get(c.severity, 0) + 1

    # Category counts
    category_order = ["localization", "perception", "both"]
    category_colors = {"localization": "#4C72B0", "perception": "#55A868", "both": "#8172B3"}
    category_counts = {c: 0 for c in category_order}
    for c in edge_cases:
        category_counts[c.category] = category_counts.get(c.category, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Edge Case Summary ({len(edge_cases)} total)", fontsize=14)

    # Severity horizontal bar
    ax = axes[0]
    sev_labels = [s for s in severity_order if severity_counts[s] > 0]
    sev_vals = [severity_counts[s] for s in sev_labels]
    sev_cols = [severity_colors[s] for s in sev_labels]
    ax.barh(sev_labels, sev_vals, color=sev_cols, edgecolor="white", height=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Severity Breakdown")
    ax.invert_yaxis()
    for i, v in enumerate(sev_vals):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=11)

    # Category horizontal bar
    ax = axes[1]
    cat_labels = [c for c in category_order if category_counts[c] > 0]
    cat_vals = [category_counts[c] for c in cat_labels]
    cat_cols = [category_colors[c] for c in cat_labels]
    ax.barh(cat_labels, cat_vals, color=cat_cols, edgecolor="white", height=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Category Breakdown")
    ax.invert_yaxis()
    for i, v in enumerate(cat_vals):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path
