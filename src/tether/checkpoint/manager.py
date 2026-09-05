"""JSONL checkpoint persistence for Tether tasks."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tether.runtime.state import TaskState

# Directories never included in the workspace file scan.
_IGNORED_DIRS = {"checkpoints", "memory", "logs", ".git", "__pycache__",
                 ".pytest_cache", ".libs", "node_modules", ".venv"}


@dataclass
class CheckpointSnapshot:
    """Full checkpoint: task state plus a lightweight workspace fingerprint.

    Only ``path`` + ``md5`` are kept per file (not the whole FileSnapshot)
    so checkpoint lines stay small while still supporting drift detection.
    """

    task_state: TaskState
    file_snapshots: list[dict[str, str]] = field(default_factory=list)
    workspace_files: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_log: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (JSONL line payload)."""
        return {
            "checkpoint_version": 2,
            "task_state": self.task_state.model_dump(mode="json"),
            "file_snapshot_summaries": self.file_snapshots,
            "workspace_files": self.workspace_files,
            "step_log": self.step_log,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointSnapshot":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls(
            task_state=TaskState.model_validate(data["task_state"]),
            file_snapshots=list(data.get("file_snapshot_summaries", [])),
            workspace_files=list(data.get("workspace_files", [])),
            step_log={str(k): list(v) for k, v in data.get("step_log", {}).items()},
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class CheckpointManager:
    """Append-only JSONL checkpoint storage.

    Each task gets one ``{task_id}.jsonl`` file under ``{workspace_dir}/checkpoints``.
    Every save appends one JSON line, so the file is a full history of states.
    Lines come in two flavors: plain TaskState (Phase 1, "v1") and full
    CheckpointSnapshot lines tagged ``checkpoint_version: 2``.
    """

    def __init__(self, workspace_dir: Path) -> None:
        """Create the checkpoints directory under ``workspace_dir`` if needed."""
        self.workspace_dir = Path(workspace_dir)
        self.checkpoint_dir = self.workspace_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, task_id: str) -> Path:
        """Return the JSONL file path for ``task_id``."""
        return self.checkpoint_dir / f"{task_id}.jsonl"

    def _read_lines(self, task_id: str) -> list[dict]:
        """Read all non-empty JSON lines for ``task_id`` (empty list if absent)."""
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return []
        records = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint line skipped in {}", path)
        return records

    def save(self, state: TaskState) -> None:
        """Append a serialized snapshot of ``state`` as one JSONL line."""
        path = self._checkpoint_path(state.task_id)
        line = state.model_dump(mode="json")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info(
            "Checkpoint saved | task_id={} step={} status={}",
            state.task_id, state.step_index, state.status.value,
        )

    def load(self, task_id: str) -> TaskState | None:
        """Load the latest plain TaskState line for ``task_id``.

        Full (v2) checkpoint lines are skipped so this keeps returning the
        Phase-1 style state. Returns ``None`` if no usable line exists.
        """
        records = self._read_lines(task_id)
        for rec in reversed(records):
            if "checkpoint_version" not in rec:
                return TaskState.model_validate(rec)
        if records:
            # Only v2 lines exist: extract the embedded state.
            return TaskState.model_validate(records[-1]["task_state"])
        logger.warning("Checkpoint file empty or missing for task {}", task_id)
        return None

    def save_full(
        self,
        state: TaskState,
        memory_store: "object",
        step_log: dict[int, list[str]] | None = None,
    ) -> None:
        """Append a full checkpoint (v2) including a workspace fingerprint.

        1. Pull all FileSnapshots for the task from ``memory_store``.
        2. Keep only path + md5 per file.
        3. Scan the workspace for the current file list.
        4. Append one JSONL line as a CheckpointSnapshot.
        """
        summaries = [
            {"path": snap.path, "md5": snap.md5}
            for snap in memory_store.list_file_snapshots(state.task_id)
        ]
        workspace_files = self.scan_workspace()
        snapshot = CheckpointSnapshot(
            task_state=state,
            file_snapshots=summaries,
            workspace_files=workspace_files,
            step_log={str(k): v for k, v in (step_log or {}).items()},
        )
        path = self._checkpoint_path(state.task_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        logger.info(
            "Full checkpoint saved | task_id={} step={} status={} files={}",
            state.task_id, state.step_index, state.status.value, len(summaries),
        )

    def load_full(self, task_id: str) -> CheckpointSnapshot | None:
        """Load the latest full (v2) checkpoint as a CheckpointSnapshot.

        Falls back to wrapping the latest plain line (empty fingerprint)
        when only Phase-1 checkpoints exist. Returns ``None`` if nothing
        can be loaded.
        """
        records = self._read_lines(task_id)
        for rec in reversed(records):
            if rec.get("checkpoint_version") == 2:
                return CheckpointSnapshot.from_dict(rec)
        for rec in reversed(records):
            if "checkpoint_version" not in rec:
                state = TaskState.model_validate(rec)
                logger.debug("Only v1 checkpoint found for {}; wrapping", task_id)
                return CheckpointSnapshot(task_state=state)
        logger.warning("No checkpoint found for task {}", task_id)
        return None

    def scan_workspace(self) -> list[str]:
        """List all workspace files (relative posix paths), ignoring internals."""
        files: list[str] = []
        if not self.workspace_dir.exists():
            return files
        for p in sorted(self.workspace_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.workspace_dir)
            if rel.parts and rel.parts[0] in _IGNORED_DIRS:
                continue
            files.append(rel.as_posix())
        return files

    def list_checkpoints(self, task_id: str) -> list[Path]:
        """Return checkpoint file paths for ``task_id`` (extensible for future shards)."""
        path = self._checkpoint_path(task_id)
        return [path] if path.exists() else []
