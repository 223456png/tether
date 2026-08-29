"""Phase 2 tests: three-layer memory data structures + MemoryStore."""

import hashlib
from pathlib import Path

import pytest

from tether.memory import (
    EpisodicNotes,
    FileSnapshot,
    MemoryStore,
    TaskSummary,
)
from tether.memory.file_snapshot import compute_md5


def test_task_summary() -> None:
    """TaskSummary round-trips through to_dict/from_dict with fields intact."""
    summary = TaskSummary(
        task_id="task-1",
        goal="Fix login bug",
        current_plan=["locate bug", "patch auth", "run tests"],
        completed=["locate bug"],
        next_action="patch auth",
        constraints=["do not touch DB schema"],
    )
    data = summary.to_dict()
    restored = TaskSummary.from_dict(data)

    assert restored.entry_id == summary.entry_id
    assert restored.task_id == summary.task_id
    assert restored.goal == summary.goal
    assert restored.current_plan == summary.current_plan
    assert restored.completed == summary.completed
    assert restored.next_action == summary.next_action
    assert restored.constraints == summary.constraints
    assert restored.created_at == summary.created_at


def test_file_snapshot_from_file(tmp_path: Path) -> None:
    """FileSnapshot.from_file computes md5/size/mtime/language/symbols."""
    src = tmp_path / "example.py"
    content = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Calculator:\n"
        "    def run(self):\n"
        "        return add(1, 2)\n"
    )
    src.write_bytes(content.encode("utf-8"))

    snap = FileSnapshot.from_file("task-1", src)

    assert snap.md5 == hashlib.md5(content.encode()).hexdigest()
    assert snap.md5 == compute_md5(src)
    assert snap.size == len(content.encode())
    assert snap.mtime == pytest.approx(src.stat().st_mtime)
    assert snap.language == "python"
    assert snap.path == src.as_posix()
    assert "add" in snap.symbols
    assert "Calculator" in snap.symbols


def test_memory_store_save_load(tmp_path: Path) -> None:
    """MemoryStore saves and loads a TaskSummary with fields intact."""
    store = MemoryStore(tmp_path)
    summary = TaskSummary(
        task_id="task-2",
        goal="Refactor module",
        current_plan=["step 1", "step 2"],
        completed=["step 1"],
        next_action="step 2",
        constraints=["keep public API"],
    )
    store.save_task_summary(summary)

    loaded = store.load_task_summary("task-2")
    assert loaded is not None
    assert loaded.to_dict() == summary.to_dict()

    # Missing task returns None
    assert store.load_task_summary("no-such-task") is None


def test_memory_store_file_snapshots(tmp_path: Path) -> None:
    """MemoryStore lists all saved FileSnapshots and deletes by path."""
    store = MemoryStore(tmp_path)
    for name in ("a.py", "b.py", "c.js"):
        f = tmp_path / name
        f.write_text(f"# {name}\n", encoding="utf-8")
        snap = FileSnapshot.from_file("task-3", f)
        store.save_file_snapshot(snap)

    snapshots = store.list_file_snapshots("task-3")
    assert len(snapshots) == 3
    assert {s.path for s in snapshots} == {
        (tmp_path / "a.py").as_posix(),
        (tmp_path / "b.py").as_posix(),
        (tmp_path / "c.js").as_posix(),
    }

    # load by path
    loaded = store.load_file_snapshot("task-3", (tmp_path / "b.py").as_posix())
    assert loaded is not None
    assert loaded.language == "python"

    # delete by path, then verify it's gone
    store.delete_file_snapshot("task-3", (tmp_path / "b.py").as_posix())
    assert store.load_file_snapshot("task-3", (tmp_path / "b.py").as_posix()) is None
    assert len(store.list_file_snapshots("task-3")) == 2


def test_memory_store_episodic(tmp_path: Path) -> None:
    """MemoryStore saves, loads, and deletes EpisodicNotes."""
    store = MemoryStore(tmp_path)
    notes = [
        EpisodicNotes(
            task_id="task-4",
            type="lesson",
            content="Always run tests before committing",
            confidence=0.9,
            source_task="task-0",
        ),
        EpisodicNotes(
            task_id="task-4",
            type="mistake",
            content="Forgot to close the file handle",
            confidence=0.7,
            source_task="task-1",
        ),
    ]
    for note in notes:
        store.save_episodic_note(note)

    loaded = store.load_episodic_notes("task-4")
    assert len(loaded) == 2
    by_content = {n.content for n in loaded}
    assert by_content == {n.content for n in notes}
    # Storage order follows entry_id filenames; compare order-independently.
    loaded_by_id = {n.entry_id: n for n in loaded}
    for note in notes:
        assert loaded_by_id[note.entry_id].to_dict() == note.to_dict()

    # delete one note, the other survives
    store.delete_episodic_note("task-4", notes[0].entry_id)
    remaining = store.load_episodic_notes("task-4")
    assert len(remaining) == 1
    assert remaining[0].entry_id == notes[1].entry_id
