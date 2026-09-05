"""Compression policy: levels and budget configuration."""

from dataclasses import dataclass
from enum import IntEnum


class CompressionLevel(IntEnum):
    """Progressive context compression levels (cumulative)."""

    NONE = 0               # keep everything
    COMPRESS_TOOL = 1      # truncate recent tool results to summaries
    COMPRESS_EPISODIC = 2  # filter episodic notes by confidence
    COMPRESS_FILE = 3      # keep only current files in full
    MINIMAL = 4            # extreme mode: skeleton + minimal everything else


@dataclass
class BudgetConfig:
    """Token budget distribution across context sections.

    TaskSummary and the System prompt are never pruned (core principle);
    their ratios reserve space, and `reserve_ratio` is a safety buffer.
    """

    total_budget: int = 16000
    system_ratio: float = 0.125
    task_summary_ratio: float = 0.125
    file_snapshot_ratio: float = 0.25
    episodic_ratio: float = 0.125
    tool_result_ratio: float = 0.25
    reserve_ratio: float = 0.125

    def __post_init__(self) -> None:
        """Validate that section ratios sum to ~1.0."""
        total = (
            self.system_ratio
            + self.task_summary_ratio
            + self.file_snapshot_ratio
            + self.episodic_ratio
            + self.tool_result_ratio
            + self.reserve_ratio
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"BudgetConfig ratios must sum to 1.0, got {total}")
