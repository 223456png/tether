"""BudgetAllocator: level-based (0-4) context pruning under a token budget.

Two allocation passes:

1. Pick the mildest *global* compression level that fits the total budget.
2. Enforce *per-section caps* (the BudgetConfig ratios): a section still
   exceeding its share is trimmed further, so one greedy section (e.g. a
   huge tool dump) can never starve the others (e.g. file context).
"""

import re

from loguru import logger

from tether.context.policy import BudgetConfig, CompressionLevel
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.task_summary import TaskSummary

# Per-level compression knobs (truncation lengths come from BudgetConfig).
_TOOL_KEEP_COUNT = {CompressionLevel.NONE: None, 1: None, 2: None, 3: None, 4: 2}
_EPISODIC_MIN_CONFIDENCE = {CompressionLevel.NONE: 0.0, 1: 0.0, 2: 0.7, 3: 0.7, 4: 0.8}
_EPISODIC_MAX_NOTES = {CompressionLevel.NONE: None, 1: None, 2: 5, 3: 5, 4: 3}

# Episodic cap fitting ladder: note counts tried until the section fits.
_EPISODIC_FIT_LADDER = (8, 4, 2, 1)

# CJK unified ideographs + extensions, kana, fullwidth forms — characters
# that modern LLM tokenizers encode at ~1 token per character. Shared by
# estimate_tokens (budget gating) and the validator (CJK-aware scoring).
_CJK_CHAR_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """Segmented token estimate, conservative in both directions.

    CJK characters count ~1 token each (modern LLM tokenizers); other
    characters ~1 token per 4. The old ``len(text) // 3`` rule
    underestimated CJK-heavy text by 2-3x (budget gateways passed while
    real token usage overran the cap) and overestimated English by 33%.
    """
    if not text:
        return 0
    cjk = len(_CJK_CHAR_RE.findall(text))
    other = len(text) - cjk
    return max(1, cjk + other // 4)


class BudgetAllocator:
    """Chooses the mildest compression level that fits the token budget."""

    def __init__(self, config: BudgetConfig) -> None:
        """Store config; start at level 0 with empty stats."""
        self.config = config
        self._current_level = CompressionLevel.NONE
        self._last_compression_stats: dict[str, object] = {}

    @property
    def current_level(self) -> CompressionLevel:
        """Compression level used by the most recent allocate() call."""
        return self._current_level

    @property
    def last_compression_stats(self) -> dict[str, object]:
        """Stats dict from the most recent allocate() call."""
        return self._last_compression_stats

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------
    def _truncate_for(self, level: CompressionLevel) -> int | None:
        """Visible-char budget for one tool result at ``level`` (None = keep)."""
        if level == CompressionLevel.NONE:
            return None
        if level >= CompressionLevel.MINIMAL:
            return self.config.tool_minimal_chars
        return self.config.tool_truncate_chars

    @staticmethod
    def _truncate_head_tail(text: str, limit: int) -> str:
        """Keep the head AND tail of an over-long text within ``limit`` chars.

        Logs carry information at both ends — the triggering input at the
        top, the verdict/traceback at the bottom — so a head-only cut
        hides exactly the conclusion. The visible budget is split 2:1
        between head and tail, with an explicit omission marker. The
        marker reserves worst-case space, so the output is guaranteed
        ``<= limit`` chars.
        """
        if len(text) <= limit:
            return text
        prefix, suffix = "...[truncated ", " chars]"
        # Worst-case marker: the omitted count can have as many digits as
        # the full text length.
        reserve = len(prefix) + len(suffix) + len(str(len(text)))
        keep = max(limit - reserve, 1)
        head = max(keep * 2 // 3, 1)
        tail = keep - head
        omitted = len(text) - head - tail
        marker = f"{prefix}{omitted}{suffix}"
        if tail > 0:
            return f"{text[:head]}{marker}{text[-tail:]}"
        return f"{text[:head]}{marker}"

    @classmethod
    def _apply_truncate(cls, result: str, truncate: int | None) -> str:
        """Apply the level's truncation policy to one tool result."""
        if truncate is None or len(result) <= truncate:
            return result
        return cls._truncate_head_tail(result, truncate)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------
    @staticmethod
    def _render_task_summary(summary: TaskSummary | None) -> str:
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
    def _full_file_line(snap: FileSnapshot) -> str:
        """One file rendered in full (symbols + summary)."""
        symbols = ", ".join(snap.symbols) if snap.symbols else "-"
        return (
            f"- {snap.path} ({snap.language}, {snap.size}B) "
            f"symbols: [{symbols}] — {snap.summary}"
        )

    @staticmethod
    def _render_files(
        snapshots: list[FileSnapshot],
        current_files: list[str],
        level: CompressionLevel,
    ) -> str:
        """Render file snapshots according to the compression level."""
        current = {str(f).replace("\\", "/") for f in current_files}
        lines: list[str] = []
        for snap in snapshots:
            is_current = snap.path in current
            if not is_current:
                if level >= CompressionLevel.MINIMAL:
                    continue  # Level 4: drop non-current files entirely
                if level >= CompressionLevel.COMPRESS_FILE:
                    # Level 3: degrade to path + md5 fingerprint.
                    lines.append(f"- {snap.path} (md5={snap.md5})")
                    continue
            lines.append(BudgetAllocator._full_file_line(snap))
        return "\n".join(lines) if lines else "(no file context)"

    @staticmethod
    def _select_episodic(
        notes: list[EpisodicNotes],
        level: CompressionLevel,
    ) -> list[EpisodicNotes]:
        """Pick the notes that survive at ``level`` (confidence + usage rank)."""
        min_conf = _EPISODIC_MIN_CONFIDENCE[level]
        max_notes = _EPISODIC_MAX_NOTES[level]
        selected = [n for n in notes if n.confidence >= min_conf]
        selected.sort(key=lambda n: n.usage_count, reverse=True)
        if max_notes is not None:
            selected = selected[:max_notes]
        return selected

    @staticmethod
    def _render_note_lines(selected: list[EpisodicNotes]) -> str:
        """Render the selected notes (shared by level render and cap fit)."""
        lines = [
            f"- [{n.type}] {n.content} (confidence={n.confidence}, used={n.usage_count})"
            for n in selected
        ]
        return "\n".join(lines) if lines else "(no episodic notes)"

    @classmethod
    def _render_episodic(
        cls,
        notes: list[EpisodicNotes],
        level: CompressionLevel,
    ) -> str:
        """Render episodic notes, filtered by confidence/count per level."""
        return cls._render_note_lines(cls._select_episodic(notes, level))

    def _render_tools(
        self,
        tool_results: list[tuple[str, str]],
        level: CompressionLevel,
    ) -> str:
        """Render recent tool results, truncated per level."""
        keep = _TOOL_KEEP_COUNT[level]
        truncate = self._truncate_for(level)
        selected = tool_results if keep is None else tool_results[-keep:]
        lines = [
            f"- {action} -> {self._apply_truncate(result, truncate)}"
            for action, result in selected
        ]
        return "\n".join(lines) if lines else "(no tool results)"

    # ------------------------------------------------------------------
    # Per-section cap fitting (second allocation pass)
    # ------------------------------------------------------------------
    def _section_caps(self) -> dict[str, int]:
        """Per-section token caps derived from the BudgetConfig ratios."""
        return {
            "file_context": int(
                self.config.total_budget * self.config.file_snapshot_ratio
            ),
            "episodic_context": int(
                self.config.total_budget * self.config.episodic_ratio
            ),
            "tool_context": int(
                self.config.total_budget * self.config.tool_result_ratio
            ),
        }

    def _fit_tool_section(
        self,
        tool_results: list[tuple[str, str]],
        level: CompressionLevel,
        cap: int,
    ) -> str:
        """Keep the *newest* results whose rendered lines fit the cap."""
        truncate = self._truncate_for(level)
        keep = _TOOL_KEEP_COUNT[level]
        selected = tool_results if keep is None else tool_results[-keep:]
        kept: list[str] = []
        used = 0
        for action, result in reversed(selected):  # newest first
            line = f"- {action} -> {self._apply_truncate(result, truncate)}"
            cost = estimate_tokens(line)
            if kept and used + cost > cap:
                break
            kept.append(line)
            used += cost
        kept.reverse()
        return "\n".join(kept) if kept else "(no tool results)"

    def _fit_file_section(
        self,
        file_snapshots: list[FileSnapshot],
        current_files: list[str],
        cap: int,
    ) -> str:
        """Degrade files progressively until the section fits the cap."""
        current = {str(f).replace("\\", "/") for f in current_files}
        current_full = [
            self._full_file_line(s) for s in file_snapshots if s.path in current
        ]
        noncurrent_fp = [
            f"- {s.path} (md5={s.md5})" for s in file_snapshots if s.path not in current
        ]
        all_fp = [f"- {s.path} (md5={s.md5})" for s in file_snapshots]
        fallback = "(no file context)"
        candidates = [
            current_full + noncurrent_fp,  # L3-style degradation
            current_full,                  # current files only
            all_fp,                        # fingerprints for everything
        ]
        for lines in candidates:
            text = "\n".join(lines) if lines else fallback
            if estimate_tokens(text) <= cap:
                return text
        # Last resort: keep as many leading fingerprint lines as fit.
        kept: list[str] = []
        used = 0
        for line in all_fp:
            cost = estimate_tokens(line)
            if kept and used + cost > cap:
                break
            kept.append(line)
            used += cost
        return "\n".join(kept) if kept else fallback

    def _fit_episodic_section(
        self,
        episodic_notes: list[EpisodicNotes],
        cap: int,
    ) -> tuple[str, list[EpisodicNotes]]:
        """Shrink the note count down a ladder until the section fits."""
        ranked = sorted(episodic_notes, key=lambda n: n.usage_count, reverse=True)
        for max_notes in (len(ranked), *_EPISODIC_FIT_LADDER):
            selected = ranked[:max_notes]
            text = self._render_note_lines(selected)
            if estimate_tokens(text) <= cap:
                return text, selected
        return "(no episodic notes)", []

    def _enforce_section_caps(
        self,
        chosen_level: CompressionLevel,
        sections: dict[str, str],
        file_snapshots: list[FileSnapshot],
        episodic_notes: list[EpisodicNotes],
        tool_results: list[tuple[str, str]],
        current_files: list[str],
    ) -> tuple[dict[str, str], list[str], list[str]]:
        """Trim any section over its ratio cap; returns (sections, trimmed, note_ids)."""
        caps = self._section_caps()
        out = dict(sections)
        trimmed: list[str] = []
        note_ids = [n.entry_id for n in self._select_episodic(episodic_notes, chosen_level)]

        if estimate_tokens(out["tool_context"]) > caps["tool_context"]:
            out["tool_context"] = self._fit_tool_section(
                tool_results, chosen_level, caps["tool_context"]
            )
            trimmed.append("tool_context")
        if estimate_tokens(out["file_context"]) > caps["file_context"]:
            out["file_context"] = self._fit_file_section(
                file_snapshots, current_files, caps["file_context"]
            )
            trimmed.append("file_context")
        if estimate_tokens(out["episodic_context"]) > caps["episodic_context"]:
            text, kept_notes = self._fit_episodic_section(
                episodic_notes, caps["episodic_context"]
            )
            out["episodic_context"] = text
            note_ids = [n.entry_id for n in kept_notes]
            trimmed.append("episodic_context")
        return out, trimmed, note_ids

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def _render_all(
        self,
        level: CompressionLevel,
        system_prompt: str,
        task_summary: TaskSummary | None,
        file_snapshots: list[FileSnapshot],
        episodic_notes: list[EpisodicNotes],
        tool_results: list[tuple[str, str]],
        current_files: list[str],
    ) -> dict[str, str]:
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
        task_summary: TaskSummary | None,
        file_snapshots: list[FileSnapshot],
        episodic_notes: list[EpisodicNotes],
        tool_results: list[tuple[str, str]],
        current_files: list[str],
    ) -> dict[str, str]:
        """Render all sections at Level 0 (used as the validation baseline)."""
        return self._render_all(
            CompressionLevel.NONE, system_prompt, task_summary,
            file_snapshots, episodic_notes, tool_results, current_files,
        )

    def render_at_level(
        self,
        level: int,
        system_prompt: str,
        task_summary: TaskSummary | None,
        file_snapshots: list[FileSnapshot],
        episodic_notes: list[EpisodicNotes],
        tool_results: list[tuple[str, str]],
        current_files: list[str],
    ) -> dict[str, str]:
        """Render all sections at an explicit compression level.

        Used by the benchmark to sweep levels independently of a token
        budget; accepts an int (0-4) or a CompressionLevel. Deliberately
        bypasses the per-section caps so the sweep measures pure levels.
        """
        return self._render_all(
            CompressionLevel(level), system_prompt, task_summary,
            file_snapshots, episodic_notes, tool_results, current_files,
        )

    def allocate(
        self,
        system_prompt: str,
        task_summary: TaskSummary | None,
        file_snapshots: list[FileSnapshot],
        episodic_notes: list[EpisodicNotes],
        tool_results: list[tuple[str, str]],
        current_files: list[str],
    ) -> dict:
        """Pick the mildest compression level fitting the budget.

        Returns a dict with the five rendered sections, the chosen level,
        and compression stats. TaskSummary/System are never pruned. With
        ``enforce_section_caps`` (default) a second pass trims any section
        over its ratio share; ``stats["section_trimmed"]`` lists what was
        additionally cut.
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

        if self.config.enforce_section_caps:
            sections, trimmed_sections, selected_note_ids = self._enforce_section_caps(
                chosen_level, sections, file_snapshots, episodic_notes,
                tool_results, current_files,
            )
            total_tokens = sum(estimate_tokens(s) for s in sections.values())
        else:
            trimmed_sections = []
            selected_note_ids = [
                n.entry_id for n in self._select_episodic(episodic_notes, chosen_level)
            ]

        reduction = (
            (original_tokens - total_tokens) / original_tokens if original_tokens else 0.0
        )
        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": total_tokens,
            "reduction_ratio": round(reduction, 4),
            # Sections additionally trimmed by the per-section cap pass.
            "section_trimmed": trimmed_sections,
            # Lets the caller mark the notes that actually entered the
            # context (usage_count / last_used bookkeeping).
            "selected_episodic_ids": selected_note_ids,
        }
        self._current_level = chosen_level
        self._last_compression_stats = stats
        logger.debug(
            "BudgetAllocator | level={} tokens {} -> {} (reduction {:.1%}) trimmed={}",
            chosen_level.name, original_tokens, total_tokens, reduction,
            trimmed_sections,
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
