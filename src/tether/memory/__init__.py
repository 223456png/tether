"""Memory package: three-layer memory for Tether.

Layers:
  - TaskSummary   : task skeleton (never pruned)
  - FileSnapshot  : per-file cache for drift detection
  - EpisodicNotes : long-term experience

Also configures a loguru file sink so ``logs/`` is auto-created on import.
"""

from loguru import logger

from tether.memory.base import MemoryEntry
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.store import MemoryStore
from tether.memory.task_summary import TaskSummary

__all__ = [
    "MemoryEntry",
    "TaskSummary",
    "FileSnapshot",
    "EpisodicNotes",
    "MemoryStore",
]

# Configure the rotating file sink once per process (module import is
# cached, so this runs once even with repeated `import tether.memory`).
_FILE_SINK_CONFIGURED = False
if not _FILE_SINK_CONFIGURED:
    logger.add(
        "logs/tether_{time}.log",
        rotation="10 MB",
        retention="3 days",
        level="DEBUG",
    )
    _FILE_SINK_CONFIGURED = True
