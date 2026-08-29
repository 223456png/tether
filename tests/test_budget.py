"""Phase 3 tests: BudgetAllocator levels 0-4 + ContextAssembler."""

from pathlib import Path

from tether.context import (
    BudgetAllocator,
    BudgetConfig,
    CompressionLevel,
    ContextAssembler,
)
from tether.memory import EpisodicNotes, FileSnapshot, MemoryStore, TaskSummary
from tether.runtime.state import TaskState


def _make_summary(goal: str = "Fix login bug") -> TaskSummary:
    return TaskSummary(
        task_id="task-1",
        goal=goal,
        current_plan=["locate bug", "patch auth"],
        completed=["locate bug"],
        next_action="patch auth",
        constraints=["do not touch DB schema"],
    )


def _make_snapshot(path: str, summary: str, symbols=None, task_id: str = "task-1") -> FileSnapshot:
    return FileSnapshot(
        task_id=task_id,
        path=path,
        md5="0" * 32,
        size=1024,
        mtime=1700000000.0,
        language="python",
        summary=summary,
        symbols=symbols or ["main"],
    )


def _make_note(content: str, confidence: float, usage: int = 0, task_id: str = "task-1") -> EpisodicNotes:
    return EpisodicNotes(
        task_id=task_id,
        type="lesson",
        content=content,
        confidence=confidence,
        source_task="task-0",
        usage_count=usage,
    )


def test_level_0_no_compression() -> None:
    """Small data fits the budget: level 0, zero reduction."""
    allocator = BudgetAllocator(BudgetConfig(total_budget=16000))
    result = allocator.allocate(
        system_prompt="You are a coding agent.",
        task_summary=_make_summary(),
        file_snapshots=[_make_snapshot("src/main.py", "entry point")],
        episodic_notes=[_make_note("run tests often", 0.9)],
        tool_results=[("read_file", "file content here")],
        current_files=["src/main.py"],
    )

    assert result["level"] == CompressionLevel.NONE
    assert result["stats"]["reduction_ratio"] == 0
    assert result["stats"]["original_tokens"] == result["stats"]["compressed_tokens"]
    assert "run tests often" in result["episodic_context"]
    assert "file content here" in result["tool_context"]


def test_level_1_compresses_tools() -> None:
    """Large tool output exceeds budget at L0 but fits when truncated at L1."""
    allocator = BudgetAllocator(BudgetConfig(total_budget=2000))
    tools = [(f"run_test_{i}", "x" * 1000) for i in range(10)]  # ~3333 tokens at L0
    result = allocator.allocate(
        system_prompt="You are a coding agent.",
        task_summary=_make_summary(),
        file_snapshots=[],
        episodic_notes=[],
        tool_results=tools,
        current_files=[],
    )

    assert result["level"] == CompressionLevel.COMPRESS_TOOL
    # Original exceeded the budget; compressed fits.
    assert result["stats"]["original_tokens"] > 2000
    assert result["stats"]["compressed_tokens"] <= 2000
    assert result["stats"]["reduction_ratio"] > 0
    # Each tool result truncated to a 200-char summary.
    for line in result["tool_context"].splitlines():
        assert len(line.split("-> ", 1)[1]) <= 201  # 200 chars + ellipsis


def test_level_2_filters_episodic() -> None:
    """Low-confidence episodic notes are dropped at L2."""
    allocator = BudgetAllocator(BudgetConfig(total_budget=1500))
    high = [_make_note("lesson " + "h" * 700, 0.9) for _ in range(6)]
    low = [_make_note("low " + "l" * 700, 0.4) for _ in range(2)]
    result = allocator.allocate(
        system_prompt="You are a coding agent.",
        task_summary=_make_summary(),
        file_snapshots=[],
        episodic_notes=high + low,
        tool_results=[],
        current_files=[],
    )

    assert result["level"] == CompressionLevel.COMPRESS_EPISODIC
    assert "low " not in result["episodic_context"]
    assert "lesson " in result["episodic_context"]
    # Top-5 cap by usage_count.
    assert len(result["episodic_context"].splitlines()) == 5


def test_level_3_filters_files() -> None:
    """Non-current files degrade to path+md5 at L3; current files stay full."""
    allocator = BudgetAllocator(BudgetConfig(total_budget=1000))
    current = [_make_snapshot("src/a.py", "current-a" + "a" * 800),
               _make_snapshot("src/b.py", "current-b" + "b" * 800)]
    other = [_make_snapshot("src/c.py", "other-c" + "c" * 800),
             _make_snapshot("src/d.py", "other-d" + "d" * 800),
             _make_snapshot("src/e.py", "other-e" + "e" * 800)]
    result = allocator.allocate(
        system_prompt="You are a coding agent.",
        task_summary=_make_summary(),
        file_snapshots=current + other,
        episodic_notes=[],
        tool_results=[],
        current_files=["src/a.py", "src/b.py"],
    )

    assert result["level"] == CompressionLevel.COMPRESS_FILE
    assert "current-a" in result["file_context"]
    assert "current-b" in result["file_context"]
    # Non-current files: path + md5 fingerprint only.
    assert "src/c.py" in result["file_context"]
    assert "other-c" not in result["file_context"]
    assert "other-d" not in result["file_context"]


def test_level_4_minimal() -> None:
    """Huge data forces MINIMAL: <=2 tools of <=100 chars, summary intact."""
    allocator = BudgetAllocator(BudgetConfig(total_budget=500))
    tools = [(f"tool_{i}", "y" * 600) for i in range(5)]
    notes = [_make_note("note " + "n" * 200, 0.9) for _ in range(4)]
    snaps = [_make_snapshot("src/a.py", "current" + "a" * 300),
             _make_snapshot("src/b.py", "other" + "b" * 300)]
    result = allocator.allocate(
        system_prompt="You are a coding agent.",
        task_summary=_make_summary(goal="Minimal test goal"),
        file_snapshots=snaps,
        episodic_notes=notes,
        tool_results=tools,
        current_files=["src/a.py"],
    )

    assert result["level"] == CompressionLevel.MINIMAL
    tool_lines = result["tool_context"].splitlines()
    assert len(tool_lines) == 2  # only the last 2 tool results
    for line in tool_lines:
        assert len(line.split("-> ", 1)[1]) <= 101  # 100 chars + ellipsis
    assert "tool_0" not in result["tool_context"]
    assert "tool_4" in result["tool_context"]
    # TaskSummary is never pruned.
    assert "Minimal test goal" in result["task_summary"]
    # Non-current files are dropped entirely at MINIMAL.
    assert "src/b.py" not in result["file_context"]
    assert "src/a.py" in result["file_context"]


def test_assembler_integration(tmp_path: Path) -> None:
    """ContextAssembler loads from MemoryStore and builds the full context."""
    store = MemoryStore(tmp_path)
    task_state = TaskState(goal="Integration goal", task_id="task-int")
    store.save_task_summary(
        TaskSummary(
            task_id="task-int",
            goal="Integration goal",
            current_plan=["step 1", "step 2"],
            completed=["step 1"],
            next_action="step 2",
            constraints=["keep it simple"],
        )
    )
    store.save_file_snapshot(_make_snapshot("src/main.py", "the entry module", task_id="task-int"))
    store.save_file_snapshot(_make_snapshot("src/util.py", "helpers", task_id="task-int"))
    store.save_episodic_note(_make_note("always write tests", 0.95, task_id="task-int"))

    assembler = ContextAssembler(BudgetAllocator(BudgetConfig()))
    context = assembler.assemble(
        system_prompt="You are a coding agent.",
        task_state=task_state,
        memory_store=store,
        tool_results=[("read_file('src/main.py')", "def main(): ...")],
        current_files=["src/main.py"],
    )

    for section in ("[SYSTEM]", "[TASK SUMMARY]", "[FILE CONTEXT]",
                    "[EPISODIC NOTES]", "[RECENT TOOLS]"):
        assert section in context
    assert "Integration goal" in context
    assert "src/main.py" in context
    assert "always write tests" in context
    assert "def main(): ..." in context
    # Section order: SYSTEM first, RECENT TOOLS last.
    assert context.index("[SYSTEM]") < context.index("[TASK SUMMARY]")
    assert context.index("[EPISODIC NOTES]") < context.index("[RECENT TOOLS]")
