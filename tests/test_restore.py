"""Tests for content-backed snapshots and file restoration on recovery."""

from pathlib import Path

from tether.checkpoint.manager import CheckpointManager
from tether.checkpoint.recovery import RecoveryManager
from tether.filesystem.drift import DriftDetector
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.store import MemoryStore
from tether.runtime.state import TaskState, TaskStatus


def test_snapshot_stores_content_when_asked(tmp_path: Path) -> None:
    """include_content=True keeps the file text; default keeps None."""
    target = tmp_path / "a.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    with_content = FileSnapshot.from_file("t1", target, include_content=True)
    without_content = FileSnapshot.from_file("t1", target)

    assert with_content.content == "def f():\n    return 1\n"
    assert without_content.content is None


def test_snapshot_skips_oversized_content(tmp_path: Path) -> None:
    """Files above the 64KB cap keep metadata only (bounded storage)."""
    target = tmp_path / "big.py"
    target.write_text("x = 1\n" * 20000, encoding="utf-8")  # 120KB

    snap = FileSnapshot.from_file("t2", target, include_content=True)

    assert snap.content is None
    assert snap.size > 64 * 1024


def test_deleted_file_restored_from_content_snapshot(tmp_path: Path) -> None:
    """file_deleted recovers when a content-backed snapshot exists."""
    ws = tmp_path
    store = MemoryStore(ws)
    checkpoints = CheckpointManager(ws)

    target = ws / "src/b.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    snap = FileSnapshot.from_file("bench", target, include_content=True)
    snap.path = "src/b.py"
    store.save_file_snapshot(snap)

    state = _failed_state()
    checkpoints.save_full(
        state, store, step_log={1: ["src/a.py"], 2: ["src/b.py"]}
    )
    target.unlink()  # external deletion

    manager = RecoveryManager(checkpoints, store, DriftDetector(ws))
    result = manager.recover(state.task_id)

    assert result.success is True
    assert result.restored_files == ["src/b.py"]
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_deleted_file_still_fails_without_content(tmp_path: Path) -> None:
    """Metadata-only snapshots cannot restore: limitation preserved."""
    ws = tmp_path
    store = MemoryStore(ws)
    checkpoints = CheckpointManager(ws)

    target = ws / "src/b.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    snap = FileSnapshot.from_file("bench", target)  # no content
    snap.path = "src/b.py"
    store.save_file_snapshot(snap)

    state = _failed_state()
    checkpoints.save_full(state, store, step_log={2: ["src/b.py"]})
    target.unlink()

    manager = RecoveryManager(checkpoints, store, DriftDetector(ws))
    result = manager.recover(state.task_id)

    assert result.success is False
    assert "missing from workspace" in (result.reason or "")


def _failed_state() -> TaskState:
    state = TaskState(task_id="bench", goal="bench goal")
    state.step_index = 3
    state.status = TaskStatus.FAILED
    return state


async def test_runtime_snapshots_written_files_with_content(tmp_path: Path) -> None:
    """After a successful write_file, a content-backed snapshot exists."""
    from tests.test_agent_loop import ScriptedProvider, _final_response, _tool_response
    from tether.runtime.runtime import TetherRuntime

    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "out.txt", "content": "authored"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Write out.txt", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    snap = runtime.memory_store.load_file_snapshot(
        runtime.state.task_id, "out.txt"
    )
    assert snap is not None
    assert snap.content == "authored"
