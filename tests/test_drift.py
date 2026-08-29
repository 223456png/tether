"""Phase 5 tests: FileSnapshot symbol extraction + DriftDetector levels."""

import asyncio
import os
from pathlib import Path

from tether.filesystem import DriftDetector, DriftLevel
from tether.memory import FileSnapshot, MemoryStore
from tether.runtime.runtime import TetherRuntime

PY_CODE = """\
import os


def alpha():
    return 1


def beta(x):
    return x + 1


class Calculator:
    def add(self, a, b):
        return a + b
"""


def _write(root: Path, rel: str, content: str) -> Path:
    """Write a file under root and return its full path."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("utf-8"))
    return p


def _snap(root: Path, rel: str, task_id: str = "task-drift") -> FileSnapshot:
    """Build a snapshot for root/rel with a workspace-relative path."""
    snap = FileSnapshot.from_file(task_id, root / rel)
    snap.path = rel
    return snap


def test_detect_match(tmp_path: Path) -> None:
    """Unchanged file -> MATCH, detected=False."""
    _write(tmp_path, "src/app.py", PY_CODE)
    detector = DriftDetector(tmp_path)
    result = detector.detect(_snap(tmp_path, "src/app.py"))

    assert result.level == DriftLevel.MATCH
    assert result.detected is False


def test_detect_metadata_change(tmp_path: Path) -> None:
    """mtime-only change (touch), content identical -> METADATA."""
    p = _write(tmp_path, "src/app.py", PY_CODE)
    detector = DriftDetector(tmp_path)
    snap = _snap(tmp_path, "src/app.py")

    # Push mtime forward without touching content.
    os.utime(p, (p.stat().st_atime + 100, p.stat().st_mtime + 100))

    result = detector.detect(snap)
    assert result.level == DriftLevel.METADATA
    assert result.detected is True
    assert result.details["mtime_changed"] is True
    assert result.details["size_changed"] is False


def test_detect_content_change_same_symbols(tmp_path: Path) -> None:
    """Comment/whitespace edits keep symbols stable -> CONTENT."""
    _write(tmp_path, "src/app.py", PY_CODE)
    detector = DriftDetector(tmp_path)
    snap = _snap(tmp_path, "src/app.py")

    # Same functions, changed comment + whitespace only.
    _write(
        tmp_path, "src/app.py",
        "# entirely new comment\n\n\ndef alpha():\n    return 1\n\n\n"
        "def beta(x):\n    return x + 1\n\n\n"
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n",
    )

    result = detector.detect(snap)
    assert result.level == DriftLevel.CONTENT
    assert result.detected is True
    assert result.details["md5_changed"] is True
    assert result.details["symbols_same"] is True


def test_detect_structure_change(tmp_path: Path) -> None:
    """Removed beta, added gamma -> STRUCTURE with added/removed."""
    _write(tmp_path, "src/app.py", PY_CODE)
    detector = DriftDetector(tmp_path)
    snap = _snap(tmp_path, "src/app.py")

    _write(
        tmp_path, "src/app.py",
        "def alpha():\n    return 1\n\n\ndef gamma():\n    return 2\n",
    )

    result = detector.detect(snap)
    assert result.level == DriftLevel.STRUCTURE
    assert result.detected is True
    assert set(result.details["removed"]) == {"beta", "add", "Calculator"}
    assert result.details["added"] == ["gamma"]


def test_detect_missing_file(tmp_path: Path) -> None:
    """Snapshot exists but the file was deleted -> MISSING."""
    _write(tmp_path, "src/app.py", PY_CODE)
    detector = DriftDetector(tmp_path)
    snap = _snap(tmp_path, "src/app.py")

    (tmp_path / "src/app.py").unlink()

    result = detector.detect(snap)
    assert result.level == DriftLevel.MISSING
    assert result.detected is True


def test_memory_store_invalidate(tmp_path: Path) -> None:
    """invalidate_file_snapshot removes the stored snapshot."""
    store = MemoryStore(tmp_path)
    p = _write(tmp_path, "src/app.py", PY_CODE)
    snap = _snap(tmp_path, "src/app.py")
    store.save_file_snapshot(snap)

    assert store.load_file_snapshot("task-drift", "src/app.py") is not None

    store.invalidate_file_snapshot("task-drift", "src/app.py")
    assert store.load_file_snapshot("task-drift", "src/app.py") is None


def test_runtime_read_with_drift(tmp_path: Path) -> None:
    """Read -> cache -> edit file -> read again detects drift and refreshes."""
    runtime = TetherRuntime("Drift runtime", tmp_path)
    _write(tmp_path, "src/app.py", PY_CODE)

    content1, snap1 = asyncio.run(
        runtime._read_file_with_drift_check("src/app.py")
    )
    assert "def alpha" in content1
    assert snap1.md5 == FileSnapshot.from_file(
        runtime.state.task_id, tmp_path / "src/app.py"
    ).md5

    # Structural edit: remove beta, add gamma.
    _write(
        tmp_path, "src/app.py",
        "def alpha():\n    return 1\n\n\ndef gamma():\n    return 2\n",
    )

    content2, snap2 = asyncio.run(
        runtime._read_file_with_drift_check("src/app.py")
    )
    assert "def gamma" in content2
    assert snap2.md5 != snap1.md5
    assert snap2.entry_id != snap1.entry_id

    # Store now holds only the refreshed snapshot.
    stored = runtime.memory_store.load_file_snapshot(
        runtime.state.task_id, "src/app.py"
    )
    assert stored is not None
    assert stored.md5 == snap2.md5


def test_runtime_read_without_drift(tmp_path: Path) -> None:
    """Unchanged file -> cached snapshot reused, no refresh."""
    runtime = TetherRuntime("No-drift runtime", tmp_path)
    _write(tmp_path, "src/app.py", PY_CODE)

    _, snap1 = asyncio.run(runtime._read_file_with_drift_check("src/app.py"))
    _, snap2 = asyncio.run(runtime._read_file_with_drift_check("src/app.py"))

    # Same snapshot object reused: no invalidation, no rebuild.
    assert snap1.entry_id == snap2.entry_id
    snaps = runtime.memory_store.list_file_snapshots(runtime.state.task_id)
    assert len(snaps) == 1
