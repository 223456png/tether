"""Tests for checkpoint persistence: compaction and load round-trips."""

from pathlib import Path

from tether.checkpoint import CheckpointManager
from tether.memory.store import MemoryStore
from tether.runtime.state import TaskState


def _state(step: int) -> TaskState:
    state = TaskState(goal="compaction test")
    state.step_index = step
    return state


def _task_lines(tmp_path: Path, task_id: str) -> list[str]:
    path = tmp_path / "checkpoints" / f"{task_id}.jsonl"
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_no_compaction_under_threshold(tmp_path: Path) -> None:
    """Few checkpoints: the append-only history is untouched."""
    manager = CheckpointManager(tmp_path, keep_last_checkpoints=20)
    store = MemoryStore(tmp_path)
    state = _state(0)
    for step in range(1, 6):
        state.step_index = step
        manager.save_full(state, store, {})

    assert len(_task_lines(tmp_path, state.task_id)) == 5
    loaded = manager.load_full(state.task_id)
    assert loaded is not None
    assert loaded.task_state.step_index == 5


def test_compaction_keeps_last_n(tmp_path: Path) -> None:
    """Beyond keep_last_checkpoints, older full snapshots are dropped."""
    manager = CheckpointManager(tmp_path, keep_last_checkpoints=3)
    store = MemoryStore(tmp_path)
    state = _state(0)
    for step in range(1, 11):
        state.step_index = step
        manager.save_full(state, store, {})

    lines = _task_lines(tmp_path, state.task_id)
    assert len(lines) == 3
    # The latest snapshot survives the compaction.
    loaded = manager.load_full(state.task_id)
    assert loaded is not None
    assert loaded.task_state.step_index == 10


def test_compaction_disabled_with_large_bound(tmp_path: Path) -> None:
    """keep_last_checkpoints=1000 behaves like the old append-only flow."""
    manager = CheckpointManager(tmp_path, keep_last_checkpoints=1000)
    store = MemoryStore(tmp_path)
    state = _state(0)
    for step in range(1, 26):
        state.step_index = step
        manager.save_full(state, store, {})

    assert len(_task_lines(tmp_path, state.task_id)) == 25


# 2026-09-30 审计修复回归：schema 漂移的 checkpoint 行跳过而非崩掉恢复。
def test_load_skips_schema_invalid_lines(tmp_path: Path) -> None:
    """A hand-edited / schema-drifted line is skipped; older state loads."""
    import json as _json

    manager = CheckpointManager(tmp_path, keep_last_checkpoints=20)
    state = _state(1)
    manager.save(state)
    newer = _state(2)
    newer.task_id = state.task_id
    manager.save(newer)

    path = tmp_path / "checkpoints" / f"{state.task_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps({"goal_is_missing_here": True}) + "\n")

    loaded = manager.load(state.task_id)
    assert loaded is not None
    assert loaded.task_id == state.task_id
    assert loaded.step_index == 2
