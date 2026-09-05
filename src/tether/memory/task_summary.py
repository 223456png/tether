"""TaskSummary: the task skeleton layer (never pruned)."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from tether.memory.base import MemoryEntry, new_entry_id


class TaskSummary(MemoryEntry, BaseModel):
    """High-level task skeleton: goal, plan, progress, constraints.

    This is the layer BudgetAllocator will never prune.
    """

    entry_id: str = Field(default_factory=new_entry_id)
    task_id: str
    goal: str
    current_plan: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    next_action: str = ""
    constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSummary":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls.model_validate(data)
