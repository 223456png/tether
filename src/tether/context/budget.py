"""BudgetAllocator: level-based (0-4) context pruning under a token budget."""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from tether.context.policy import BudgetConfig, CompressionLevel
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.task_summary import TaskSummary

# Per-level compression knobs.
_TOOL_TRUNCATE_LEN = {CompressionLevel.NONE: None, 1: 200, 2: 200, 3: 200, 4: 100}
_TOOL_KEEP_COUNT = {CompressionLevel.NONE: None, 1: None, 2: None, 3: None, 4: 2}
_EPISODIC_MIN_CONFIDENCE = {CompressionLevel.NONE: 0.0, 1: 0.0, 2: 0.7, 3: 0.7, 4: 0.8}
_EPISODIC_MAX_NOTES = {CompressionLevel.NONE: None, 1: None, 2: 5, 3: 5, 4: 3}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 3 (mixed CJK/English, conservative)."""
    return len(text) // 3


class BudgetAllocator:
    """Chooses the mildest compression level that fits the token budget."""

    def __init__(self, config: BudgetConfig) -> None:
        """Store config; start at level 0 with empty stats."""
        self.config = config
        self._current_level = CompressionLevel.NONE
        self._last_compression_stats: Dict[str, object] = {}

    @property
    def current_level(self) -> CompressionLevel:
        """Compression level used by the most recent allocate() call."""
        return self._current_level

    @property
    def last_compression_stats(self) -> Dict[str, object]:
        """Stats dict from the most recent allocate() call."""
        return self._last_compression_stats

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------
    @staticmethod
    def _render_task_summary(summary: Optional[TaskSummary]) -> str:
        """Render TaskSummary in full (never pruned)."""
        if summary is None:
            return "(no task summary)"
        lines = [f"Goal: {summary.goal}"]
        if summary.current_plan:
            lines.append("Plan:")
            lines += [f"  - {step}" for step in summary.current_plan]
        if summary.completed:
            lines.append("Completed:")
            lines += [f"  - {step}" for step in summary.completed]
        if summary.next_action:
            lines.append(f"Next action: {summary.next_action}")
        if summary.constraints:
            lines.append("Constraints:")
            lines += [f"  - {c}" for c in summary.constraints]
        return "\n".join(lines)

    @staticmethod
    def _render_files(
        snapshots: List[FileSnapshot],
        current_files: List[str],
        level: CompressionLevel,
    ) -> str:
        """Render file snapshots according to the compression level."""
        current = {str(f).replace("\\", "/") for f in current_files}
        lines: List[str] = []
        for snap in snapshots:
            is_current = snap.path in current
            if not is_current:
                if level >= CompressionLevel.MINIMAL:
                    continue  # Level 4: drop non-current files entirely
                if level >= CompressionLevel.COMPRESS_FILE:
                    # Level 3: degrade to path + md5 fingerprint.
                    lines.append(f"- {snap.path} (md5={snap.md5})")
                    continue
            symbols = ", ".join(snap.symbols) if snap.symbols else "-"
            lines.append(
                f"- {snap.path} ({snap.language}, {snap.size}B) "
                f"symbols: [{symbols}] — {snap.summary}"
            )
        return "\n".join(lines) if lines else "(no file context)"

    @staticmethod
    def _select_episodic(
        notes: List[EpisodicNotes],
        level: CompressionLevel,
    ) -> List[EpisodicNotes]:
        """Pick the notes that survive at ``level`` (confidence + usage rank)."""
        min_conf = _EPISODIC_MIN_CONFIDENCE[level]
        max_notes = _EPISODIC_MAX_NOTES[level]
        selected = [n for n in notes if n.confidence >= min_conf]
        selected.sort(key=lambda n: n.usage_count, reverse=True)
        if max_notes is not None:
            selected = selected[:max_notes]
        return selected

    @classmethod
    def _render_episodic(
        cls,
        notes: List[EpisodicNotes],
        level: CompressionLevel,
    ) -> str:
        """Render episodic notes, filtered by confidence/count per level."""
        selected = cls._select_episodic(notes, level)
        lines = [
            f"- [{n.type}] {n.content} (confidence={n.confidence}, used={n.usage_count})"
            for n in selected
        ]
        return "\n".join(lines) if lines else "(no episodic notes)"

    @staticmethod
    def _render_tools(
        tool_results: List[Tuple[str, str]],
        level: CompressionLevel,
    ) -> str:
        """Render recent tool results, truncated per level."""
        keep = _TOOL_KEEP_COUNT[level]
        truncate = _TOOL_TRUNCATE_LEN[level]
        selected = tool_results if keep is None else tool_results[-keep:]
        lines = []
        for action, result in selected:
            if truncate is not None and len(result) > truncate:
                result = result[:truncate] + "…"
            lines.append(f"- {action} -> {result}")
        return "\n".join(lines) if lines else "(no tool results)"

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def _render_all(
        self,
        level: CompressionLevel,
        system_prompt: str,
        task_summary: Optional[TaskSummary],
        file_snapshots: List[FileSnapshot],
        episodic_notes: List[EpisodicNotes],
        tool_results: List[Tuple[str, str]],
        current_files: List[str],
    ) -> Dict[str, str]:
        """Render every context section at the given compression level."""
        return {
            "system": system_prompt,
            "task_summary": self._render_task_summary(task_summary),
            "file_context": self._render_files(file_snapshots, current_files, level),
            "episodic_context": self._render_episodic(episodic_notes, level),
            "tool_context": self._render_tools(tool_results, level),
        }

    def render_uncompressed(
        self,
        system_prompt: str,
        task_summary: Optional[TaskSummary],
        file_snapshots: List[FileSnapshot],
        episodic_notes: List[EpisodicNotes],
        tool_results: List[Tuple[str, str]],
        current_files: List[str],
    ) -> Dict[str, str]:
        """Render all sections at Level 0 (used as the validation baseline)."""
        return self._render_all(
            CompressionLevel.NONE, system_prompt, task_summary,
            file_snapshots, episodic_notes, tool_results, current_files,
        )

    def render_at_level(
        self,
        level: int,
        system_prompt: str,
        task_summary: Optional[TaskSummary],
        file_snapshots: List[FileSnapshot],
        episodic_notes: List[EpisodicNotes],
        tool_results: List[Tuple[str, str]],
        current_files: List[str],
    ) -> Dict[str, str]:
        """Render all sections at an explicit compression level.

        Used by the benchmark to sweep levels independently of a token
        budget; accepts an int (0-4) or a CompressionLevel.
        """
        return self._render_all(
            CompressionLevel(level), system_prompt, task_summary,
            file_snapshots, episodic_notes, tool_results, current_files,
        )

    def allocate(
        self,
        system_prompt: str,
        task_summary: Optional[TaskSummary],
        file_snapshots: List[FileSnapshot],
        episodic_notes: List[EpisodicNotes],
        tool_results: List[Tuple[str, str]],
        current_files: List[str],
    ) -> dict:
        """Pick the mildest compression level fitting the budget.

        Returns a dict with the five rendered sections, the chosen level,
        and compression stats. TaskSummary/System are never pruned.
        """
        original_sections = self._render_all(
            CompressionLevel.NONE, system_prompt, task_summary,
            file_snapshots, episodic_notes, tool_results, current_files,
        )
        original_tokens = sum(estimate_tokens(s) for s in original_sections.values())

        chosen_level = CompressionLevel.NONE
        sections = original_sections
        total_tokens = original_tokens
        for level in CompressionLevel:
            candidate = self._render_all(
                level, system_prompt, task_summary,
                file_snapshots, episodic_notes, tool_results, current_files,
            )
            tokens = sum(estimate_tokens(s) for s in candidate.values())
            if tokens <= self.config.total_budget:
                chosen_level, sections, total_tokens = level, candidate, tokens
                break
        else:
            # Even MINIMAL exceeds the budget: force it and warn.
            chosen_level = CompressionLevel.MINIMAL
            sections = self._render_all(
                chosen_level, system_prompt, task_summary,
                file_snapshots, episodic_notes, tool_results, current_files,
            )
            total_tokens = sum(estimate_tokens(s) for s in sections.values())
            logger.warning(
                "Budget exceeded even at MINIMAL level: {} > {} tokens",
                total_tokens, self.config.total_budget,
            )

        reduction = (
            (original_tokens - total_tokens) / original_tokens if original_tokens else 0.0
        )
        selected_note_ids = [
            n.entry_id for n in self._select_episodic(episodic_notes, chosen_level)
        ]
        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": total_tokens,
            "reduction_ratio": round(reduction, 4),
            # Lets the caller mark the notes that actually entered the
            # context (usage_count / last_used bookkeeping).
            "selected_episodic_ids": selected_note_ids,
        }
        self._current_level = chosen_level
        self._last_compression_stats = stats
        logger.debug(
            "BudgetAllocator | level={} tokens {} -> {} (reduction {:.1%})",
            chosen_level.name, original_tokens, total_tokens, reduction,
        )

        return {
            "system": sections["system"],
            "task_summary": sections["task_summary"],
            "file_context": sections["file_context"],
            "episodic_context": sections["episodic_context"],
            "tool_context": sections["tool_context"],
            "level": chosen_level,
            "stats": stats,
        }
