"""Phase 8 tests: benchmark configs, metrics, persistence, datasets."""

import asyncio
import json
from pathlib import Path

from tether.benchmarks.config import default_configs
from tether.benchmarks.datasets import (
    DRIFT_CHANGE_TYPES,
    generate_drift_samples,
    load_humaneval,
)
from tether.benchmarks.metrics import TaskResult, compute_metrics
from tether.benchmarks.runner import BenchmarkRunner


def test_config_validation() -> None:
    """Default experiment configs carry correct default values."""
    configs = default_configs()

    assert set(configs) == {
        "compression", "memory", "drift", "recovery", "intercept", "e2e"
    }
    comp = configs["compression"]
    assert comp.baselines == ["full", "last_n", "budget"]
    assert comp.budget_levels == [0, 1, 2, 3, 4]
    assert comp.seed == 42
    assert comp.num_samples is None
    assert configs["recovery"].runs_per_scenario == 5
    assert configs["drift"].samples_per_type == 10
    # e2e defaults to a small sample count: real API calls cost money.
    assert configs["e2e"].num_samples == 5
    assert configs["e2e"].num_samples == configs["e2e"].default_num_samples


def test_metrics_computation() -> None:
    """compute_metrics aggregates success/tokens/latency correctly."""
    results = [
        TaskResult(
            task_id="t1", variant="full", success=True, steps=5,
            prompt_tokens=100, completion_tokens=50,
            latency_ms=100.0,
        ),
        TaskResult(
            task_id="t2", variant="full", success=False, steps=7,
            prompt_tokens=300, completion_tokens=100,
            latency_ms=300.0,
        ),
    ]
    metrics = compute_metrics(results)

    assert metrics["n_samples"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["avg_steps"] == 6.0
    assert metrics["avg_tokens"]["prompt"] == 200.0
    assert metrics["avg_tokens"]["total"] == 275.0
    assert metrics["avg_latency_ms"] == 200.0
    # p95 = higher value (2 samples, nearest rank).
    assert metrics["p95_latency_ms"] == 300.0


def test_result_save_load(tmp_path: Path) -> None:
    """Runner saves results that round-trip through JSON."""
    configs = default_configs(num_samples=3)
    config = configs["intercept"]
    config.output_dir = tmp_path

    runner = BenchmarkRunner(config)
    results = asyncio.run(runner.run_experiment())
    json_path = runner.save_results()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "intercept"
    saved = payload["variants"]["interceptor"]["results"]
    assert len(saved) == len(results) == 3
    assert saved[0]["task_id"] == results[0].task_id
    assert payload["variants"]["interceptor"]["metrics"]["n_samples"] == 3


def test_dataset_loader() -> None:
    """HumanEval loads non-empty tasks with the expected fields."""
    tasks = load_humaneval(num_samples=5)
    assert len(tasks) == 5
    for task in tasks:
        assert "task_id" in task
        assert "prompt" in task
        assert task["prompt"].startswith(("from", "\n", "def"))


def test_drift_dataset_generation() -> None:
    """Generated drift dataset covers all 10 change types."""
    samples = generate_drift_samples(samples_per_type=2, seed=1)

    assert len(samples) == len(DRIFT_CHANGE_TYPES) * 2
    types = {s["change_type"] for s in samples}
    assert types == set(DRIFT_CHANGE_TYPES)
    # Every sample carries an expected drift level.
    assert all(s["expected_level"] for s in samples)


# ----------------------------------------------------------------------
# Phase 9: real-LLM e2e experiment
# ----------------------------------------------------------------------

_E2E_TASK = {
    "task_id": "HumanEval/999",
    "prompt": "def add_two(a, b):\n"
              "    \"\"\"Return the sum of a and b.\n"
              "    >>> add_two(1, 2)\n"
              "    3\n"
              "    \"\"\"\n",
    "entry_point": "add_two",
    "test": "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 1) == 0\n",
    "canonical_solution": "    return a + b\n",
}


def test_humaneval_check_passes_correct_solution() -> None:
    """A correct completion passes the official-test subprocess check."""
    from tether.benchmarks.datasets import run_humaneval_check

    result = run_humaneval_check(_E2E_TASK, "    return a + b\n")
    assert result["passed"] is True
    assert result["exit_code"] == 0


def test_humaneval_check_fails_wrong_solution() -> None:
    """A wrong completion fails; fenced code is unwrapped first."""
    from tether.benchmarks.datasets import run_humaneval_check, strip_code_fence

    assert strip_code_fence("```python\n    return a + b\n```") == "    return a + b"
    result = run_humaneval_check(_E2E_TASK, "    return a - b\n")
    assert result["passed"] is False
    assert result["exit_code"] != 0
    assert result["stderr_tail"]  # assertion traceback captured


def test_e2e_mock_pipeline(tmp_path: Path) -> None:
    """e2e runs end-to-end with the offline MockProvider and persists."""
    configs = default_configs(num_samples=2)
    config = configs["e2e"]
    assert config.num_samples == 2
    config.output_dir = tmp_path

    runner = BenchmarkRunner(config)
    results = asyncio.run(runner.run_experiment())
    json_path = runner.save_results()

    # 2 tasks x 3 arms.
    assert len(results) == 6
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "e2e"
    assert set(payload["variants"]) == {"full", "last_n", "budget"}
    for data in payload["variants"].values():
        for r in data["results"]:
            assert r["extra"]["is_mock"] is True
            assert r["extra"]["provider"] == "mock"
            assert r["extra"]["exit_code"] in (0, 1, -1)
