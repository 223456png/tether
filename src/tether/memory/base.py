"""Abstract base class for all memory entries."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def new_entry_id() -> str:
    """Generate a new UUID for a memory entry."""
    return uuid.uuid4().hex


class MemoryEntry(ABC):
    """Common fields and serialization contract for all memory entries.

    Concrete classes combine this with ``pydantic.BaseModel`` via
    ``class TaskSummary(MemoryEntry, BaseModel)`` style inheritance or by
    re-declaring the fields. This base defines the shared identity fields
    and the (de)serialization contract.
    """

    entry_id: str
    task_id: str
    created_at: datetime

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize this entry to a plain dict (JSON-compatible)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize an entry from a plain dict."""
