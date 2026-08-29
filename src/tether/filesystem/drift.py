"""DriftDetector: three-level file drift detection.

Levels: stat (size/mtime) -> MD5 -> symbol structure. The agent must never
make decisions from stale file state, so any detected drift invalidates
the cached snapshot (handled by the caller / MemoryStore).
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

from tether.filesystem.diff import compare_symbols
from tether.memory.file_snapshot import (
    FileSnapshot,
    compute_md5,
    extract_signatures,
    extract_symbols,
)

# Files larger than this skip the (relatively costly) symbol extraction.
_AST_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class DriftLevel(IntEnum):
    """Severity of file drift between snapshot and disk."""

    MATCH = 0      # identical (size + mtime)
    METADATA = 1   # stat changed but MD5 identical
    CONTENT = 2    # MD5 changed but symbol structure identical
    STRUCTURE = 3  # symbols added/removed
    MISSING = 4    # file no longer exists


@dataclass
class DriftResult:
    """Outcome of one drift detection run."""

    level: DriftLevel
    detected: bool
    details: Dict[str, object] = field(default_factory=dict)


class DriftDetector:
    """Detects drift between cached FileSnapshots and the live filesystem.

    Results are cached per (path, stat) pair so repeated checks of an
    unchanged file within the same process are cheap.
    """

    def __init__(self, workspace_root: Path, enable_ast: bool = True) -> None:
        """Store workspace root; ``enable_ast`` toggles symbol-level checks."""
        self.workspace_root = Path(workspace_root)
        self.enable_ast = enable_ast
        self._cache: Dict[str, Tuple[int, int, DriftResult]] = {}

    def _cached_detect(self, snapshot: FileSnapshot) -> Optional[DriftResult]:
        """Return a cached result if the file's stat has not changed."""
        file_path = self.workspace_root / snapshot.path
        try:
            stat = file_path.stat()
        except OSError:
            return None
        key = snapshot.path
        cached = self._cache.get(key)
        if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
            return cached[2]
        return None

    def detect(self, snapshot: FileSnapshot) -> DriftResult:
        """Run the three-level detection cascade for one snapshot.

        Level 1 checks size/mtime, Level 2 checks MD5, Level 3 compares
        extracted symbols. Returns the most specific DriftResult.
        """
        cached = self._cached_detect(snapshot)
        if cached is not None:
            return cached

        file_path = self.workspace_root / snapshot.path

        if not file_path.exists():
            result = DriftResult(level=DriftLevel.MISSING, detected=True)
            self._store(snapshot.path, file_path, result)
            return result

        # Level 1: stat check
        stat = file_path.stat()
        size_match = stat.st_size == snapshot.size
        mtime_match = stat.st_mtime == snapshot.mtime
        if size_match and mtime_match:
            result = DriftResult(level=DriftLevel.MATCH, detected=False)
            self._store(snapshot.path, file_path, result)
            return result

        # Level 2: MD5 check
        current_md5 = compute_md5(file_path)
        if current_md5 == snapshot.md5:
            result = DriftResult(
                level=DriftLevel.METADATA,
                detected=True,
                details={
                    "size_changed": not size_match,
                    "mtime_changed": not mtime_match,
                },
            )
            self._store(snapshot.path, file_path, result)
            return result

        # Level 3: symbol + signature comparison (optional, size-capped)
        if self.enable_ast and stat.st_size <= _AST_MAX_FILE_SIZE:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            current_symbols = extract_symbols(text, snapshot.language)
            current_signatures = extract_signatures(text, snapshot.language)
            # Signature comparison only applies when the snapshot actually
            # stored signatures (legacy snapshots default to []).
            signatures_changed = bool(snapshot.signatures) and (
                sorted(snapshot.signatures) != sorted(current_signatures)
            )
            diff = compare_symbols(snapshot.symbols, current_symbols)
            if diff["changed"] or signatures_changed:
                details = {
                    "added": diff["added"],
                    "removed": diff["removed"],
                    "common": diff["common"],
                    "signatures_changed": signatures_changed,
                }
                result = DriftResult(
                    level=DriftLevel.STRUCTURE,
                    detected=True,
                    details=details,
                )
            else:
                result = DriftResult(
                    level=DriftLevel.CONTENT,
                    detected=True,
                    details={"md5_changed": True, "symbols_same": True},
                )
            self._store(snapshot.path, file_path, result)
            logger.debug(
                "Drift detected | {} level={} details={}",
                snapshot.path, result.level.name, result.details,
            )
            return result

        # AST disabled: MD5 change alone counts as CONTENT drift.
        result = DriftResult(
            level=DriftLevel.CONTENT,
            detected=True,
            details={"md5_changed": True},
        )
        self._store(snapshot.path, file_path, result)
        logger.debug(
            "Drift detected | {} level=CONTENT (md5 changed)", snapshot.path,
        )
        return result

    def _store(self, path_key: str, file_path: Path, result: DriftResult) -> None:
        """Cache a result keyed by the file's current stat."""
        try:
            stat = file_path.stat()
            self._cache[path_key] = (stat.st_size, stat.st_mtime_ns, result)
        except OSError:
            self._cache.pop(path_key, None)
