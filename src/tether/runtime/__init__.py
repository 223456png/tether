"""Runtime package: state machine and main loop.

``TetherRuntime`` is exported lazily to avoid a circular import with
``tether.checkpoint`` (the runtime imports the checkpoint manager, and
checkpoint models import ``TaskState`` from this package).
"""

from typing import Any

from tether.runtime.state import TaskState, TaskStatus

__all__ = ["TaskState", "TaskStatus", "TetherRuntime"]


def __getattr__(name: str) -> Any:
    """Lazily import TetherRuntime on first attribute access."""
    if name == "TetherRuntime":
        from tether.runtime.runtime import TetherRuntime

        return TetherRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
