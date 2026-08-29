"""MemoryStore: unified file-based storage for the three memory layers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.task_summary import TaskSummary

if TYPE_CHECKING:  # pragma: no cover - circular-import guard
    from tether.filesystem.drift import DriftDetector


class MemoryStore:
    """File-backed store for TaskSummary / FileSnapshot / EpisodicNotes.

    Layout under ``{workspace_dir}/memory/{task_id}/``:
      - ``task_summary.json``                  (overwritten in place)
      - ``file_snapshots/{entry_id}.json``     (one file per snapshot)
      - ``episodic_notes/{entry_id}.json``     (one file per note)
    """

    def __init__(self, workspace_dir: Path) -> None:
        """Create the memory root directory."""
        self.workspace_dir = Path(workspace_dir)
        self.memory_dir = self.workspace_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _task_dir(self, task_id: str) -> Path:
        """Return (and create if needed) the per-task memory directory."""
        task_dir = self.memory_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        """Atomically overwrite ``path`` with pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        """Read a JSON file, returning None if missing or corrupted."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read {}: {}", path, exc)
            return None

    # ------------------------------------------------------------------
    # TaskSummary
    # ------------------------------------------------------------------
    def save_task_summary(self, summary: TaskSummary) -> None:
        """Persist the task summary (overwrite on each save)."""
        path = self._task_dir(summary.task_id) / "task_summary.json"
        self._write_json(path, summary.to_dict())
        logger.info("TaskSummary saved | task_id={}", summary.task_id)

    def load_task_summary(self, task_id: str) -> Optional[TaskSummary]:
        """Load the task summary for ``task_id``, or None if absent."""
        path = self.memory_dir / task_id / "task_summary.json"
        data = self._read_json(path)
        if data is None:
            logger.warning("TaskSummary not found | task_id={}", task_id)
            return None
        return TaskSummary.from_dict(data)

    # ------------------------------------------------------------------
    # FileSnapshot
    # ------------------------------------------------------------------
    def _snapshot_dir(self, task_id: str) -> Path:
        """Return (and create if needed) the file_snapshots directory."""
        d = self._task_dir(task_id) / "file_snapshots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_file_snapshot(self, snapshot: FileSnapshot) -> None:
        """Persist a file snapshot keyed by its entry_id."""
        path = self._snapshot_dir(snapshot.task_id) / f"{snapshot.entry_id}.json"
        self._write_json(path, snapshot.to_dict())
        logger.info("FileSnapshot saved | task_id={} path={}", snapshot.task_id, snapshot.path)

    def load_file_snapshot(self, task_id: str, file_path: str) -> Optional[FileSnapshot]:
        """Load the snapshot whose ``path`` matches ``file_path`` (posix-normalized)."""
        target = Path(file_path).as_posix()
        for snap in self.list_file_snapshots(task_id):
            if snap.path == target:
                return snap
        logger.warning("FileSnapshot not found | task_id={} path={}", task_id, file_path)
        return None

    def list_file_snapshots(self, task_id: str) -> List[FileSnapshot]:
        """Return all snapshots for ``task_id`` (empty list if none)."""
        d = self.memory_dir / task_id / "file_snapshots"
        if not d.exists():
            return []
        snapshots: List[FileSnapshot] = []
        for f in sorted(d.glob("*.json")):
            data = self._read_json(f)
            if data is not None:
                snapshots.append(FileSnapshot.from_dict(data))
        return snapshots

    def delete_file_snapshot(self, task_id: str, file_path: str) -> None:
        """Delete the snapshot matching ``file_path`` (used on drift invalidation)."""
        target = Path(file_path).as_posix()
        for snap in self.list_file_snapshots(task_id):
            if snap.path == target:
                path = self._snapshot_dir(task_id) / f"{snap.entry_id}.json"
                path.unlink(missing_ok=True)
                logger.info("FileSnapshot deleted | task_id={} path={}", task_id, file_path)
                return
        logger.warning("FileSnapshot delete missed | task_id={} path={}", task_id, file_path)

    def invalidate_file_snapshot(self, task_id: str, file_path: str) -> None:
        """Invalidate (delete) the snapshot for ``file_path`` after drift.

        A semantic alias over ``delete_file_snapshot`` used by the drift
        detection flow.
        """
        self.delete_file_snapshot(task_id, file_path)
        logger.info("Invalidated file snapshot: {}", file_path)

    def load_snapshot_with_drift(
        self,
        task_id: str,
        file_path: str,
        detector: "DriftDetector",
    ) -> Tuple[Optional[FileSnapshot], "object"]:
        """Load a snapshot and run drift detection on it.

        Returns ``(snapshot, result)``. When no snapshot exists the result
        is MISSING; otherwise the caller decides whether to invalidate
        based on ``result.level``.
        """
        # Imported lazily: tether.filesystem.drift imports this package.
        from tether.filesystem.drift import DriftLevel, DriftResult

        snapshot = self.load_file_snapshot(task_id, file_path)
        if snapshot is None:
            return None, DriftResult(level=DriftLevel.MISSING, detected=True)
        return snapshot, detector.detect(snapshot)

    # ------------------------------------------------------------------
    # EpisodicNotes
    # ------------------------------------------------------------------
    def _episodic_dir(self, task_id: str) -> Path:
        """Return (and create if needed) the episodic_notes directory."""
        d = self._task_dir(task_id) / "episodic_notes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_episodic_note(self, note: EpisodicNotes) -> None:
        """Persist an episodic note keyed by its entry_id."""
        path = self._episodic_dir(note.task_id) / f"{note.entry_id}.json"
        self._write_json(path, note.to_dict())
        logger.info("EpisodicNote saved | task_id={} type={}", note.task_id, note.type)

    def load_episodic_notes(self, task_id: str) -> List[EpisodicNotes]:
        """Return all episodic notes for ``task_id`` (empty list if none)."""
        d = self.memory_dir / task_id / "episodic_notes"
        if not d.exists():
            return []
        notes: List[EpisodicNotes] = []
        for f in sorted(d.glob("*.json")):
            data = self._read_json(f)
            if data is not None:
                notes.append(EpisodicNotes.from_dict(data))
        return notes

    def delete_episodic_note(self, task_id: str, entry_id: str) -> None:
        """Delete an episodic note by entry_id (no-op if absent)."""
        path = self._episodic_dir(task_id) / f"{entry_id}.json"
        if path.exists():
            path.unlink()
            logger.info("EpisodicNote deleted | task_id={} entry_id={}", task_id, entry_id)
        else:
            logger.warning("EpisodicNote delete missed | task_id={} entry_id={}", task_id, entry_id)
