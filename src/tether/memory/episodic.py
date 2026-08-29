"""EpisodicNotes: long-term experience layer (lessons, patterns, mistakes)."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from tether.memory.base import MemoryEntry, new_entry_id

NoteType = Literal["lesson", "pattern", "mistake", "preference"]


class EpisodicNotes(MemoryEntry, BaseModel):
    """A single experiential note that makes the agent smarter over time."""

    entry_id: str = Field(default_factory=new_entry_id)
    task_id: str
    type: NoteType
    content: str
    confidence: float = 1.0
    source_task: str = ""
    usage_count: int = 0
    last_used: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicNotes":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls.model_validate(data)
