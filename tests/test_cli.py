"""Smoke tests for the tether CLI."""

from pathlib import Path

from tether.cli import build_parser, main
from tether.reporting import summarize_events


def test_cli_run_mock_completes(tmp_path: Path) -> None:
    """`tether run --mock` finishes and writes the event stream."""
    exit_code = main([
        "run",
        "--goal", "demo task",
        "--workspace", str(tmp_path),
        "--mock",
        "--max-steps", "3",
    ])

    assert exit_code == 0
    assert (tmp_path / "logs" / "events.jsonl").exists()


def test_cli_report_writes_markdown(tmp_path: Path) -> None:
    """`tether report` turns a real run's events into a markdown file."""
    main([
        "run",
        "--goal", "report me",
        "--workspace", str(tmp_path),
        "--mock",
        "--max-steps", "3",
    ])
    out = tmp_path / "report.md"

    exit_code = main([
        "report",
        "--events", str(tmp_path / "logs" / "events.jsonl"),
        "--out", str(out),
    ])

    assert exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "# Tether run report" in text
    assert "report me" in text
    assert "completed" in text
    assert "Tool calls" in text


def test_cli_report_missing_file(tmp_path: Path) -> None:
    """A nonexistent events file exits 1 with a clear message."""
    exit_code = main([
        "report",
        "--events", str(tmp_path / "nope.jsonl"),
        "--out", str(tmp_path / "x.md"),
    ])
    assert exit_code == 1


def test_summarize_events_empty() -> None:
    """Empty event lists render a placeholder report."""
    assert "no events" in summarize_events([])


def test_summarize_reports_goal_achievement() -> None:
    """A goal_verified event renders an explicit ✓/✗ verdict."""
    events = [
        {"event": "task_started", "task_id": "t1", "goal": "g", "brain": "llm"},
        {
            "event": "tool_executed", "tool": "write_file",
            "success": True, "cached": False,
        },
        {"event": "task_completed", "steps": 2, "total_tokens": 10},
        {"event": "goal_verified", "passed": True, "status": "completed"},
    ]
    report = summarize_events(events)
    assert "Goal achievement" in report
    assert "✓" in report


def test_summarize_flags_failed_tools_and_incomplete_tasks() -> None:
    """Failed tool calls and non-completed tasks get prominent warnings."""
    events = [
        {"event": "task_started", "task_id": "t1", "goal": "g", "brain": "llm"},
        {
            "event": "tool_executed", "tool": "run_test",
            "success": False, "cached": False,
        },
        {
            "event": "tool_executed", "tool": "run_test",
            "success": False, "cached": False,
        },
        {
            "event": "task_ended", "status": "stopped", "steps": 2,
            "error": "Step cap reached: 2 steps executed >= max_steps 2",
        },
    ]
    report = summarize_events(events)
    assert "2 of 2 tool call(s) FAILED" in report
    assert "Task ended without completing" in report
    assert "✗ goal not achieved" in report


def test_summarize_warns_when_completed_without_tools() -> None:
    """'completed' with zero tool executions is flagged, not celebrated."""
    events = [
        {"event": "task_started", "task_id": "t1", "goal": "g", "brain": "mock"},
        {"event": "task_completed", "steps": 1, "total_tokens": 0},
    ]
    report = summarize_events(events)
    assert "completed without executing any tools" in report


def test_cli_parses_defaults() -> None:
    """Defaults are sane so a bare `tether run --goal x` works."""
    args = build_parser().parse_args(["run", "--goal", "x"])
    assert args.max_steps == 20
    assert args.max_consecutive_failures == 3
    assert args.mock is False
    assert args.workspace == "."
    assert args.mcp_cmd is None
