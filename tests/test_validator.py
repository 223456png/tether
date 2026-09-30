"""Phase 4 tests: ROUGE-L validator + assembler rollback integration."""

from pathlib import Path
from unittest.mock import MagicMock

from tether.context import (
    BudgetAllocator,
    BudgetConfig,
    ContextAssembler,
    RougeValidator,
)
from tether.memory import MemoryStore
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
    tools: list[tuple[str, str]] = [
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


# 2026-09-30 CJK 审计修复回归：rouge_score 的 tokenizer 把一切非 ASCII
# 字符当分隔符——纯中文恒 0（永远回滚），混合文本中文丢光也能通过。
# CJK-aware 组合相似度后，两个方向都必须正确。
ZH_THREE_STEPS = "第一步：读取数据源。第二步：计算比率。第三步：生成评估报告。"


def test_rouge_validator_pure_chinese_passes() -> None:
    """纯中文 3 步保 2 步 → 高相似度，不再被恒 0 分数误杀。"""
    validator = RougeValidator()
    compressed = "第一步：读取数据源。第二步：计算比率。"
    result = validator.validate(ZH_THREE_STEPS, compressed)
    assert result.passed is True
    assert result.rouge_l_score > 0.7


def test_rouge_validator_pure_chinese_over_compression_fails() -> None:
    """纯中文删掉 3/4 步骤 → 必须回滚（此前恒 0 也回滚，但属误杀路径）。"""
    validator = RougeValidator()
    result = validator.validate(ZH_THREE_STEPS, "第一步：读取数据源。")
    assert result.passed is False
    assert result.rouge_l_score < 0.7


def test_rouge_validator_mixed_dropped_chinese_fails() -> None:
    """混合文本丢光中文 → 必须回滚（此前 rouge=1.000 静默放行）。"""
    validator = RougeValidator()
    original = "Read the config file and compute the ratio. " + ZH_THREE_STEPS
    compressed = "Read the config file and compute the ratio."
    result = validator.validate(original, compressed)
    assert result.passed is False


def test_rouge_validator_chinese_keyword_guard_still_works() -> None:
    """纯中文相似度修复后，中文关键词守卫依然按位生效。"""
    validator = RougeValidator()
    result = validator.validate(ZH_THREE_STEPS, ZH_THREE_STEPS)
    assert result.passed is True
    ok = validator.validate(
        ZH_THREE_STEPS, ZH_THREE_STEPS + " 关键结论：评估报告已通过。",
        required_keywords=["评估报告"],
    )
    assert ok.passed is True
    missing = validator.validate(
        ZH_THREE_STEPS, ZH_THREE_STEPS, required_keywords=["不存在的关键词"]
    )
    assert missing.passed is False
