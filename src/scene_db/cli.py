"""CLI entry point for scene-db."""

from pathlib import Path
from typing import Optional

import typer

from scene_db.db import get_connection, list_all_scenes
from scene_db.export import export_scene
from scene_db.ingest import ingest_sequence
from scene_db.search import search

app = typer.Typer(help="Search and extract scenes from autonomous driving log data.")


def _db_path_option() -> Path | None:
    """Common DB path option (returns None to use default)."""
    return None


@app.command()
def ingest(
    dataset_path: Path = typer.Argument(..., help="Path to KITTI sequence directory"),
    dataset_name: str = typer.Option("kitti", help="Dataset name"),
    chunk_duration: float = typer.Option(5.0, help="Chunk duration in seconds"),
    db: Optional[Path] = typer.Option(None, help="Database path (default: ~/.scene-db/scene.db)"),
) -> None:
    """Ingest a KITTI dataset sequence into the scene database."""
    if not dataset_path.exists():
        typer.echo(f"Error: path does not exist: {dataset_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Ingesting {dataset_path} ...")
    try:
        n = ingest_sequence(dataset_path, dataset_name, chunk_duration, db)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Done. Created {n} scene chunks.")


@app.command()
def index(
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Rebuild the search index (re-generate captions for all scenes)."""
    conn = get_connection(db)
    scenes = list_all_scenes(conn)
    typer.echo(f"Index contains {len(scenes)} scenes.")
    conn.close()


@app.command(name="search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query text"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Search scenes by text query."""
    conn = get_connection(db)
    results = search(conn, query)
    conn.close()

    if not results:
        typer.echo("No scenes found.")
        return

    typer.echo(f"Found {len(results)} scene(s):\n")
    for s in results:
        typer.echo(f"  [{s.id}]")
        typer.echo(f"    {s.caption}")
        typer.echo(f"    frames {s.start_frame}-{s.end_frame}, "
                   f"{s.start_time.isoformat()} - {s.end_time.isoformat()}")
        typer.echo()


@app.command()
def export(
    id: str = typer.Option(..., "--id", help="Scene chunk ID to export"),
    output: Path = typer.Option("./export", "-o", "--output", help="Output directory"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Export a scene's files to a directory."""
    conn = get_connection(db)
    try:
        n = export_scene(conn, id, output)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        conn.close()
    typer.echo(f"Exported {n} files to {output}")
