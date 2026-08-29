"""ContextAssembler: loads memory layers, allocates budget, validates, assembles."""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from tether.context.budget import BudgetAllocator
from tether.context.validator import RougeValidator
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.store import MemoryStore
from tether.memory.task_summary import TaskSummary
from tether.runtime.state import TaskState

_SECTION_ORDER = ("system", "task_summary", "file_context", "episodic_context", "tool_context")


class ContextAssembler:
    """Builds the final LLM context string from memory + budget allocation.

    After allocation, the compressed context is validated against the
    uncompressed baseline with ROUGE-L; on failure the original context is
    used instead (better to spend tokens than lose key information).
    """

    def __init__(
        self,
        allocator: BudgetAllocator,
        validator: Optional[RougeValidator] = None,
        enable_validation: bool = True,
    ) -> None:
        """Store allocator and validator; validation is on by default."""
        self.allocator = allocator
        self.validator = validator or RougeValidator()
        self.enable_validation = enable_validation
        self.rollback_count = 0

    @staticmethod
    def _join(sections: Dict[str, str]) -> str:
        """Join rendered sections into the final context string."""
        headers = {
            "system": "[SYSTEM]",
            "task_summary": "[TASK SUMMARY]",
            "file_context": "[FILE CONTEXT]",
            "episodic_context": "[EPISODIC NOTES]",
            "tool_context": "[RECENT TOOLS]",
        }
        return "\n\n".join(
            f"{headers[key]}\n{sections[key]}" for key in _SECTION_ORDER
        )

    def _load_memory(
        self, task_state: TaskState, memory_store: MemoryStore
    ) -> Tuple[TaskSummary, List[FileSnapshot], List[EpisodicNotes]]:
        """Load the three memory layers, synthesizing a summary if absent."""
        summary = memory_store.load_task_summary(task_state.task_id)
        if summary is None:
            summary = TaskSummary(
                task_id=task_state.task_id,
                goal=task_state.goal,
                next_action=f"continue at step {task_state.step_index}",
            )
            logger.debug("No stored TaskSummary; synthesized from TaskState")
        snapshots = memory_store.list_file_snapshots(task_state.task_id)
        notes = memory_store.load_episodic_notes(task_state.task_id)
        return summary, snapshots, notes

    @staticmethod
    def _extract_keywords(summary: TaskSummary) -> List[str]:
        """Extract must-survive keywords from the task summary."""
        return [s for s in (summary.goal, summary.next_action) if s]

    def assemble(
        self,
        system_prompt: str,
        task_state: TaskState,
        memory_store: MemoryStore,
        tool_results: List[Tuple[str, str]],
        current_files: List[str],
    ) -> str:
        """Load memory, allocate budget, validate compression, assemble.

        If validation is enabled and fails, rolls back to the uncompressed
        context and increments ``rollback_count``.
        """
        summary, snapshots, notes = self._load_memory(task_state, memory_store)

        allocated = self.allocator.allocate(
            system_prompt=system_prompt,
            task_summary=summary,
            file_snapshots=snapshots,
            episodic_notes=notes,
            tool_results=tool_results,
            current_files=current_files,
        )
        compressed_context = self._join(allocated)

        if not self.enable_validation:
            logger.debug("Validation disabled; using compressed context")
            return compressed_context

        original_sections = self.allocator.render_uncompressed(
            system_prompt=system_prompt,
            task_summary=summary,
            file_snapshots=snapshots,
            episodic_notes=notes,
            tool_results=tool_results,
            current_files=current_files,
        )
        original_context = self._join(original_sections)

        result = self.validator.validate(
            original_context, compressed_context,
            required_keywords=self._extract_keywords(summary),
        )
        if result.passed:
            final_context = compressed_context
        else:
            self.rollback_count += 1
            logger.warning(
                "Validation failed ({}), rolled back to original context "
                "(task_id={}, rollback #{}), tokens {} -> {}",
                result.reason, task_state.task_id, self.rollback_count,
                result.compressed_token_count, result.original_token_count,
            )
            final_context = original_context

        logger.debug(
            "Context assembled | task_id={} level={} validated={} chars={}",
            task_state.task_id,
            allocated["level"].name,
            result.passed,
            len(final_context),
        )
        return final_context
