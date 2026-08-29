"""Phase 6 tests: smart checkpoint recovery + drift-aware replay planning."""

import asyncio
from pathlib import Path

import pytest

from tether.checkpoint import CheckpointManager, RecoveryManager, RecoveryScenario
from tether.filesystem import DriftDetector
from tether.memory import FileSnapshot, MemoryStore
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskState

TASK_ID = "recovery-task"

FILES = {
    "src/a.py": "def alpha():\n    return 1\n",
    "src/b.py": "def beta():\n    return 2\n",
    "src/c.py": "def gamma():\n    return 3\n",
}

# Which files each step used (1-3).
STEP_LOG = {1: ["src/a.py"], 2: ["src/b.py"], 3: ["src/c.py"]}


def _setup_workspace(tmp_path: Path, files=None) -> None:
    """Write the fixture files under tmp_path."""
    for rel, content in (files or FILES).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode())


def _make_manager(
    tmp_path: Path,
    error_message: str = "",
    task_id: str = TASK_ID,
) -> RecoveryManager:
    """Build a RecoveryManager with a 3-step checkpoint saved."""
    _setup_workspace(tmp_path)
    store = MemoryStore(tmp_path)
    cm = CheckpointManager(tmp_path)

    for rel in FILES:
        snap = FileSnapshot.from_file(task_id, tmp_path / rel)
        snap.path = rel
        store.save_file_snapshot(snap)

    state = TaskState(task_id=task_id, goal="Recovery goal")
    state.step_index = 3
    state.error_message = error_message or None
    cm.save_full(state, store, STEP_LOG)

    detector = DriftDetector(tmp_path)
    return RecoveryManager(cm, store, detector)


def test_recovery_no_drift(tmp_path: Path) -> None:
    """Unchanged workspace: nothing to replay, everything skippable."""
    rm = _make_manager(tmp_path)
    result = rm.analyze(TASK_ID)

    assert result.success is True
    assert result.steps_to_replay == []
    assert result.steps_to_skip == [1, 2, 3]
    assert result.drift_detected is False
    assert result.scenario == RecoveryScenario.PROCESS_CRASH


def test_recovery_content_drift(tmp_path: Path) -> None:
    """Comment-only edit of the file read at step 1 -> replay step 1."""
    rm = _make_manager(tmp_path)
    (tmp_path / "src/a.py").write_bytes(
        b"# new comment\ndef alpha():\n    return 1\n"
    )

    result = rm.analyze(TASK_ID)
    assert result.success is True
    assert result.drift_detected is True
    assert result.drift_details["levels"]["src/a.py"] == "CONTENT"
    assert 1 in result.steps_to_replay
    assert result.scenario == RecoveryScenario.FILE_MODIFIED


def test_recovery_structure_drift(tmp_path: Path) -> None:
    """Symbol-level change of the step-2 file -> replay from step 2 on."""
    rm = _make_manager(tmp_path)
    (tmp_path / "src/b.py").write_bytes(b"def beta2():\n    return 20\n")

    result = rm.analyze(TASK_ID)
    assert result.success is True
    assert result.drift_details["levels"]["src/b.py"] == "STRUCTURE"
    assert result.steps_to_replay == [2, 3]  # from first affected step
    assert result.steps_to_skip == [1]


def test_recovery_file_deleted(tmp_path: Path) -> None:
    """Deleted file -> recovery fails with an explicit reason."""
    rm = _make_manager(tmp_path)
    (tmp_path / "src/a.py").unlink()

    result = rm.analyze(TASK_ID)
    assert result.success is False
    assert result.drift_detected is True
    assert "src/a.py" in result.reason
    assert result.scenario == RecoveryScenario.FILE_DELETED

    # Runtime resume surfaces a clear error.
    runtime = TetherRuntime("Deleted file", tmp_path)
    with pytest.raises(RuntimeError, match="Cannot recover"):
        asyncio.run(runtime.resume(TASK_ID))


def test_recovery_context_overflow(tmp_path: Path) -> None:
    """Context overflow -> forced compression (budget lowered) on recover."""
    rm = _make_manager(tmp_path, error_message="Context overflow: token limit exceeded")
    result = rm.recover(TASK_ID)

    assert result.success is True
    assert result.scenario == RecoveryScenario.CONTEXT_OVERFLOW
    assert result.compression_applied is True
    assert result.task_state.context_budget == 8000


def test_recovery_timeout(tmp_path: Path) -> None:
    """Timeout -> the interrupted step is marked for replay."""
    rm = _make_manager(
        tmp_path, error_message="Tool timeout: 'run_test' did not finish in 5s"
    )
    result = rm.analyze(TASK_ID)

    assert result.success is True
    assert result.scenario == RecoveryScenario.TIMEOUT
    assert result.steps_to_replay == [3]


def test_recovery_api_rate_limit(tmp_path: Path) -> None:
    """Rate limit -> exponential backoff doubles across recoveries."""
    rm = _make_manager(tmp_path, error_message="API rate limit: HTTP 429")

    first = rm.recover(TASK_ID)
    assert first.success is True
    assert first.scenario == RecoveryScenario.API_RATE_LIMIT
    assert first.backoff_seconds is not None

    second = rm.recover(TASK_ID)
    assert second.backoff_seconds == first.backoff_seconds * 2


def test_recovery_oom(tmp_path: Path) -> None:
    """OOM -> recovery succeeds with a clean state."""
    rm = _make_manager(tmp_path, error_message="MemoryError: out of memory")
    result = rm.recover(TASK_ID)

    assert result.success is True
    assert result.scenario == RecoveryScenario.OOM
    assert result.task_state is not None
    assert result.task_state.error_message is None


def test_recovery_user_interrupt(tmp_path: Path) -> None:
    """Ctrl+C interrupt -> recovery succeeds and execution continues."""
    rm = _make_manager(tmp_path, error_message="KeyboardInterrupt: user interrupt")
    result = rm.recover(TASK_ID)

    assert result.success is True
    assert result.scenario == RecoveryScenario.USER_INTERRUPT
    assert result.task_state.error_message is None

    # Full runtime flow: resume a clean interrupted task and finish it.
    runtime = TetherRuntime("Interrupted task", tmp_path, tool_timeout=5)
    asyncio.run(runtime.resume(TASK_ID))
    assert runtime.state.status.value == "completed"
    assert runtime.state.step_index > 3


def test_recovery_dependency_failure(tmp_path: Path) -> None:
    """Dependency install failure -> step marked for retry."""
    rm = _make_manager(
        tmp_path, error_message="pip install failed: dependency not found"
    )
    result = rm.analyze(TASK_ID)

    assert result.success is True
    assert result.scenario == RecoveryScenario.DEPENDENCY_FAILURE
    assert result.steps_to_replay == [3]
