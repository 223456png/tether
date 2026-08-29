"""Phase 4 tests: ROUGE-L validator + assembler rollback integration."""

from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

from tether.context import (
    BudgetAllocator,
    BudgetConfig,
    CompressionLevel,
    ContextAssembler,
    RougeValidator,
)
from tether.memory import MemoryStore, TaskSummary
from tether.runtime.state import TaskState

ORIGINAL = (
    "The auth module handles user login via JWT tokens. "
    "The login endpoint validates credentials against the database. "
    "Token refresh happens every 30 minutes. "
    "Password hashing uses bcrypt with 12 rounds. "
    "The session store is Redis with a 24 hour TTL."
)


def test_rouge_validator_passes() -> None:
    """A near-identical summary keeps high ROUGE-L and passes."""
    compressed = (
        "The auth module handles user login via JWT tokens. "
        "Token refresh happens every 30 minutes. "
        "Password hashing uses bcrypt."
    )
    validator = RougeValidator(threshold=0.5)
    result = validator.validate(ORIGINAL, compressed)

    assert result.passed is True
    assert result.was_rolled_back is False
    assert result.rouge_l_score >= 0.5
    assert result.reason is None
    assert result.original_token_count >= result.compressed_token_count


def test_rouge_validator_fails() -> None:
    """A scrambled/unrelated compression loses semantics and rolls back."""
    compressed = (
        "zzz qqq xyz banana apple cherry delta echo foxtrot golf hotel "
        "india juliet kilo lima mike november oscar papa quebec romeo "
        "sierra tango uniform victor whiskey xray yankee zulu alpha bravo"
    )
    validator = RougeValidator(threshold=0.7)
    result = validator.validate(ORIGINAL, compressed)

    assert result.passed is False
    assert result.was_rolled_back is True
    assert result.rouge_l_score < 0.7
    assert result.reason is not None
    assert "ROUGE-L" in result.reason


def test_keyword_check() -> None:
    """High ROUGE-L still fails when a required keyword is missing."""
    validator = RougeValidator(threshold=0.7)
    # Identical text -> ROUGE-L = 1.0, but keyword absent.
    result = validator.validate(ORIGINAL, ORIGINAL, required_keywords=["SESSION_SECRET"])

    assert result.passed is False
    assert result.was_rolled_back is True
    assert result.rouge_l_score >= 0.7  # similarity is fine...
    assert "missing keywords" in result.reason  # ...but keyword check fails

    # Same call with the keyword present passes.
    ok = validator.validate(ORIGINAL, ORIGINAL + " SESSION_SECRET", required_keywords=["SESSION_SECRET"])
    assert ok.passed is True


def test_validation_disabled(tmp_path: Path) -> None:
    """With enable_validation=False the validator is never called."""
    validator = MagicMock(spec=RougeValidator)
    assembler = ContextAssembler(
        BudgetAllocator(BudgetConfig()), validator=validator, enable_validation=False,
    )
    store = MemoryStore(tmp_path)
    task_state = TaskState(goal="Disabled validation goal")
    context = assembler.assemble(
        system_prompt="You are a coding agent.",
        task_state=task_state,
        memory_store=store,
        tool_results=[("read_file", "def main(): ...")],
        current_files=[],
    )
    validator.validate.assert_not_called()
    assert "[SYSTEM]" in context
    assert "Disabled validation goal" in context


def test_assembler_integration_rollback(tmp_path: Path) -> None:
    """Heavy tool results -> compression -> ROUGE-L fails -> rollback."""
    store = MemoryStore(tmp_path)
    task_state = TaskState(goal="Rollback goal")
    # Diverse per-tool content so truncation genuinely loses information
    # (repeated characters would keep ROUGE-L artificially high).
    tools: List[Tuple[str, str]] = [
        (f"tool_{i}", " ".join(f"detail_{i}_{j}" for j in range(120)))
        for i in range(10)
    ]

    assembler = ContextAssembler(
        BudgetAllocator(BudgetConfig(total_budget=2000)),
        RougeValidator(threshold=0.7),
        enable_validation=True,
    )
    context = assembler.assemble(
        system_prompt="You are a coding agent.",
        task_state=task_state,
        memory_store=store,
        tool_results=tools,
        current_files=[],
    )

    # Compression dropped most tool detail -> rollback to original.
    assert assembler.rollback_count == 1
    assert "tool_0" in context
    assert "detail_0_119" in context  # full original tool text preserved

    # A second assemble with tiny data does not roll back.
    context2 = assembler.assemble(
        system_prompt="You are a coding agent.",
        task_state=task_state,
        memory_store=store,
        tool_results=[("read_file", "short output")],
        current_files=[],
    )
    assert assembler.rollback_count == 1
    assert "short output" in context2


def test_cache_hit() -> None:
    """Identical text pairs hit the cache on the second validation."""
    validator = RougeValidator(threshold=0.7)
    first = validator.validate(ORIGINAL, ORIGINAL)
    assert validator.cache_misses == 1
    assert validator.cache_hits == 0

    second = validator.validate(ORIGINAL, ORIGINAL)
    assert validator.cache_hits == 1
    assert validator.cache_misses == 1
    assert second.rouge_l_score == first.rouge_l_score  # cached value reused
