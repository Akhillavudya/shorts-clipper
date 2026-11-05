"""Process-lifetime ownership and bounded raw log reads."""

from pathlib import Path


class RunnerClaim:
    """An OS lock released even after a crash; the file is never unlinked."""

    def __init__(self, jobs_root: Path):
        self.path = jobs_root / "_queue" / "runner.lock"
        self.handle = None

    def acquire(self):
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "Another server owns this queue. Stop it before starting this one."
            ) from exc
        self.handle = handle

    def release(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def log_tail(path: Path, lines: int) -> str:
    # Read at most 1 MiB, even for a worker that has logged for days.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 1024 * 1024))
        data = stream.read()
    if size > len(data):
        data = data.partition(b"\n")[2]
    return b"\n".join(data.splitlines()[-lines:]).decode("utf-8", errors="replace")
