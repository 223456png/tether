"""RecoveryManager: smart checkpoint recovery with workspace drift analysis.

Recovery is not "mechanically going back in time" — it decides, in the
*current* workspace state, which steps can be kept and which must be redone.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger

from tether.checkpoint.manager import CheckpointManager, CheckpointSnapshot
from tether.filesystem.drift import DriftDetector, DriftLevel
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import compute_md5
from tether.memory.store import MemoryStore
from tether.runtime.state import TaskState

# Budget used when a context overflow forced the recovery.
_RECOVERY_CONTEXT_BUDGET = 8000
_MAX_BACKOFF_SECONDS = 60.0


class RecoveryScenario(Enum):
    """The 10 recovery scenarios Tether understands."""

    PROCESS_CRASH = "process_crash"        # process died unexpectedly
    TIMEOUT = "timeout"                    # tool execution timed out
    API_RATE_LIMIT = "api_rate_limit"      # LLM API returned 429
    CONTEXT_OVERFLOW = "context_overflow"  # context window exceeded
    USER_INTERRUPT = "user_interrupt"      # manual Ctrl+C style stop
    FILE_MODIFIED = "file_modified"        # workspace file changed externally
    FILE_DELETED = "file_deleted"          # workspace file removed
    TOOL_FAILURE = "tool_failure"          # tool failed but is retryable
    OOM = "oom"                            # out of memory
    DEPENDENCY_FAILURE = "dependency_failure"  # pip install etc. failed


# error_message keyword -> scenario (checked in order).
_ERROR_SCENARIO_RULES: List[tuple] = [
    (("timeout", "timed out"), RecoveryScenario.TIMEOUT),
    (("rate limit", "429", "ratelimit"), RecoveryScenario.API_RATE_LIMIT),
    (("context overflow", "context limit", "token limit"), RecoveryScenario.CONTEXT_OVERFLOW),
    (("interrupt", "keyboardinterrupt", "ctrl+c"), RecoveryScenario.USER_INTERRUPT),
    (("out of memory", "oom", "memoryerror"), RecoveryScenario.OOM),
    (("dependency", "pip install", "module not found"), RecoveryScenario.DEPENDENCY_FAILURE),
    (("tool",), RecoveryScenario.TOOL_FAILURE),
]


@dataclass
class RecoveryResult:
    """Outcome of a recovery analysis / execution."""

    success: bool
    task_state: Optional[TaskState] = None
    steps_to_replay: List[int] = field(default_factory=list)
    steps_to_skip: List[int] = field(default_factory=list)
    drift_detected: bool = False
    drift_details: Dict[str, object] = field(default_factory=dict)
    reason: Optional[str] = None
    scenario: Optional[RecoveryScenario] = None
    backoff_seconds: Optional[float] = None
    compression_applied: bool = False


class RecoveryManager:
    """Analyzes checkpoints + workspace drift and executes smart recovery."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        memory_store: MemoryStore,
        drift_detector: DriftDetector,
    ) -> None:
        """Store collaborators and per-task retry counters."""
        self.checkpoint_manager = checkpoint_manager
        self.memory_store = memory_store
        self.drift_detector = drift_detector
        self._retry_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Scenario inference
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_error_scenario(error_message: Optional[str]) -> Optional[RecoveryScenario]:
        """Map an error message to a scenario via keyword rules."""
        if not error_message:
            return None
        lowered = error_message.lower()
        for keywords, scenario in _ERROR_SCENARIO_RULES:
            if any(kw in lowered for kw in keywords):
                return scenario
        return None

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------
    def _detect_file_drift(
        self, snapshot: CheckpointSnapshot
    ) -> Dict[str, DriftLevel]:
        """Compare checkpoint summaries against the live workspace.

        MD5 is compared directly (cheap, size/mtime independent). When the
        full FileSnapshot is still in the MemoryStore, the DriftDetector
        cascade refines CONTENT vs STRUCTURE.
        """
        root = self.drift_detector.workspace_root
        levels: Dict[str, DriftLevel] = {}
        for summary in snapshot.file_snapshots:
            path = summary["path"]
            file_path = root / path
            if not file_path.exists():
                levels[path] = DriftLevel.MISSING
                continue
            if compute_md5(file_path) == summary["md5"]:
                levels[path] = DriftLevel.MATCH
                continue
            full = self.memory_store.load_file_snapshot(
                snapshot.task_state.task_id, path
            )
            if full is not None:
                levels[path] = self.drift_detector.detect(full).level
            else:
                # No full snapshot to refine with: assume content drift.
                levels[path] = DriftLevel.CONTENT
        return levels

    def _affected_steps(
        self, snapshot: CheckpointSnapshot, drifted_files: List[str]
    ) -> Dict[str, List[int]]:
        """Map drifted files to the steps that used them (via the step log)."""
        mapping: Dict[str, List[int]] = {}
        for path in drifted_files:
            steps = [
                int(step)
                for step, files in snapshot.step_log.items()
                if path in files
            ]
            mapping[path] = sorted(steps)
        return mapping

    # ------------------------------------------------------------------
    # analyze / recover
    # ------------------------------------------------------------------
    def analyze(self, task_id: str) -> RecoveryResult:
        """Analyze recovery feasibility without executing anything.

        Loads the latest full checkpoint, detects per-file drift, and plans
        which steps must be replayed vs skipped. Missing files make the
        recovery fail with an explicit reason.
        """
        snapshot = self.checkpoint_manager.load_full(task_id)
        if snapshot is None:
            return RecoveryResult(
                success=False,
                reason=f"No checkpoint found for task {task_id}",
            )
        state = snapshot.task_state

        levels = self._detect_file_drift(snapshot)
        missing = [p for p, lv in levels.items() if lv == DriftLevel.MISSING]
        # Drifted files keep their DriftLevel for severity decisions;
        # names are produced only at the details boundary.
        drifted = {
            p: lv
            for p, lv in levels.items()
            if lv in (DriftLevel.CONTENT, DriftLevel.STRUCTURE)
        }

        scenario = self._infer_error_scenario(state.error_message)
        if scenario is None:
            if missing:
                scenario = RecoveryScenario.FILE_DELETED
            elif drifted:
                scenario = RecoveryScenario.FILE_MODIFIED
            else:
                scenario = RecoveryScenario.PROCESS_CRASH

        if missing:
            reason = (
                f"Recovery impossible: files missing from workspace: {missing}"
            )
            logger.error("Recovery analysis failed | task={} {}", task_id, reason)
            return RecoveryResult(
                success=False,
                task_state=state,
                drift_detected=True,
                drift_details={
                    "missing_files": missing,
                    "levels": {p: lv.name for p, lv in levels.items()},
                },
                reason=reason,
                scenario=RecoveryScenario.FILE_DELETED,
            )

        affected = self._affected_steps(snapshot, list(drifted.keys()))
        all_affected_steps = sorted({s for steps in affected.values() for s in steps})
        current = state.step_index

        # Plan replay range according to drift severity / scenario.
        replay: List[int] = []
        if drifted:
            structure_files = [
                p for p, lv in drifted.items() if lv == DriftLevel.STRUCTURE
            ]
            if structure_files:
                # Structural change: replay from the first affected step on.
                first = min(
                    (s for f in structure_files for s in affected.get(f, [])),
                    default=current,
                )
                replay = list(range(max(1, first), current + 1))
            else:
                # Content-only change: replay the last couple of affected steps.
                replay = all_affected_steps[-2:]
        elif scenario == RecoveryScenario.TIMEOUT:
            # The step that was waiting on the tool never finished.
            replay = [current]
        elif scenario in (
            RecoveryScenario.TOOL_FAILURE,
            RecoveryScenario.DEPENDENCY_FAILURE,
            RecoveryScenario.OOM,
            RecoveryScenario.API_RATE_LIMIT,
        ):
            replay = [current]

        replay = sorted(set(replay) & {s for s in range(1, current + 1)})
        skip = [s for s in range(1, current + 1) if s not in replay]

        drift_details: Dict[str, object] = {
            "levels": {p: lv.name for p, lv in levels.items()},
            "affected_files": list(drifted.keys()),
            "file_steps": affected,
        }

        result = RecoveryResult(
            success=True,
            task_state=state,
            steps_to_replay=replay,
            steps_to_skip=skip,
            drift_detected=bool(drifted),
            drift_details=drift_details,
            scenario=scenario,
        )
        logger.info(
            "Recovery analysis | task={} scenario={} drift={} replay={} skip={}",
            task_id, scenario.name, bool(drifted), replay, skip,
        )
        return result

    def compute_backoff(self, task_id: str) -> float:
        """Exponential backoff for rate-limited tasks: min(60, 2**attempt)."""
        attempt = self._retry_counts.get(task_id, 0)
        self._retry_counts[task_id] = attempt + 1
        return float(min(_MAX_BACKOFF_SECONDS, 2 ** attempt))

    def recover(self, task_id: str) -> RecoveryResult:
        """Execute recovery: analyze, invalidate drifted snapshots, fix state.

        - Drifted files have their FileSnapshots invalidated so the next
          read rebuilds them from disk.
        - Context overflow lowers ``context_budget`` (forced compression).
        - Rate limits produce an exponential ``backoff_seconds``.
        """
        result = self.analyze(task_id)
        if not result.success:
            return result
        assert result.task_state is not None
        state = result.task_state

        if result.drift_detected:
            for file_path in result.drift_details.get("affected_files", []):
                self.memory_store.invalidate_file_snapshot(task_id, file_path)
            logger.warning(
                "Drift handled | task={} invalidated files={}",
                task_id, result.drift_details.get("affected_files"),
            )

        if result.scenario == RecoveryScenario.CONTEXT_OVERFLOW:
            state.context_budget = _RECOVERY_CONTEXT_BUDGET
            result.compression_applied = True
            logger.info(
                "Context overflow | task={} budget lowered to {}",
                task_id, _RECOVERY_CONTEXT_BUDGET,
            )

        if result.scenario == RecoveryScenario.API_RATE_LIMIT:
            result.backoff_seconds = self.compute_backoff(task_id)
            logger.info(
                "Rate limited | task={} backoff={}s",
                task_id, result.backoff_seconds,
            )

        self._record_lesson_note(task_id, result)

        # The runtime transitions to RECOVERING -> RUNNING on resume.
        state.error_message = None
        state.touch()
        logger.info(
            "Recovery executed | task={} replay={} skip={}",
            task_id, result.steps_to_replay, result.steps_to_skip,
        )
        return result

    def _record_lesson_note(self, task_id: str, result: RecoveryResult) -> None:
        """Persist a 'lesson' episodic note about this recovery (deduped)."""
        if result.scenario is None:
            return
        content = (
            f"Recovered from {result.scenario.name}: replay={result.steps_to_replay}, "
            f"skip={result.steps_to_skip}"
        )
        for note in self.memory_store.load_episodic_notes(task_id):
            if note.content == content:
                return  # same lesson already recorded
        self.memory_store.save_episodic_note(
            EpisodicNotes(
                task_id=task_id,
                type="lesson",
                content=content,
                confidence=0.9,
                source_task=task_id,
            )
        )
