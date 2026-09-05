"""Smoke tests for the tether CLI."""

from pathlib import Path

from tether.cli import build_parser, main


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


def test_cli_parses_defaults() -> None:
    """Defaults are sane so a bare `tether run --goal x` works."""
    args = build_parser().parse_args(["run", "--goal", "x"])
    assert args.max_steps == 20
    assert args.max_consecutive_failures == 3
    assert args.mock is False
    assert args.workspace == "."
