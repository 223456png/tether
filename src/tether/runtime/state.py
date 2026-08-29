"""Task state definitions for the Tether runtime."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle states of a Tether task."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


def _utc_now() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TaskState(BaseModel):
    """Pydantic model representing the full state of a task."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    step_index: int = 0
    context_budget: int = 16000
    error_message: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def touch(self) -> None:
        """Refresh ``updated_at`` to the current UTC time."""
        self.updated_at = _utc_now()
