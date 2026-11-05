"""Storage accounting never follows links outside the jobs directory."""

import os
import shutil
from pathlib import Path

from clipper.state import JobStore


def free_bytes(path: Path) -> int:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def source_bytes(store: JobStore, video_id: str) -> int:
    path = store.job_dir(video_id) / "source.mp4"
    return path.stat().st_size if path.is_file() and not path.is_symlink() else 0


def job_sizes(store: JobStore) -> list[dict]:
    result = []
    for video_id in store.list_jobs():
        directory = store.job_dir(video_id)
        if directory.is_symlink():
            continue
        total = 0
        for root, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for name in files:
                path = Path(root) / name
                try:
                    if not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    pass
        result.append(
            {"video_id": video_id, "bytes": total, "source_bytes": source_bytes(store, video_id)}
        )
    return result
