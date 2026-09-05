"""EventRecorder: append-only JSONL observability stream.

One line per runtime event under ``{workspace}/logs/events.jsonl``:
task lifecycle, LLM turns (tokens, compression level), tool executions
(cached vs real), recovery decisions. Cheap enough to always be on;
grep-able afterwards and directly consumable by benchmark tooling.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class EventRecorder:
    """Writes structured events to a JSONL file and keeps them in memory."""

    def __init__(self, workspace_dir: Path, task_id: str = "") -> None:
        """Prepare the logs directory; ``task_id`` stamps every event."""
        self.log_path = Path(workspace_dir) / "logs" / "events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_id = task_id
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> None:
        """Append one event line; ``fields`` carry the event payload."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "task_id": self.task_id,
            **fields,
        }
        self.events.append(entry)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:  # observability must never break the loop
            logger.warning("EventRecorder write failed: {}", exc)

    def query(self, event: str | None = None) -> list[dict[str, Any]]:
        """Return recorded events, optionally filtered by type."""
        if event is None:
            return list(self.events)
        return [e for e in self.events if e["event"] == event]
