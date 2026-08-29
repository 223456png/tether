#!/usr/bin/env python3
"""Tether Benchmark Runner.

用法：
    python scripts/run_benchmark.py --experiment compression
    python scripts/run_benchmark.py --experiment memory
    python scripts/run_benchmark.py --experiment drift
    python scripts/run_benchmark.py --experiment recovery
    python scripts/run_benchmark.py --experiment intercept
    python scripts/run_benchmark.py --experiment e2e   # 真实 LLM (需 DEEPSEEK_API_KEY)
    python scripts/run_benchmark.py --all             # 不含 e2e（避免意外 API 费用）
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Allow running from the repo without installation.
_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT / "src", _ROOT / ".libs"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

from tether.benchmarks.config import default_configs  # noqa: E402
from tether.benchmarks.report import ReportGenerator  # noqa: E402
from tether.benchmarks.runner import BenchmarkRunner  # noqa: E402

_EXPERIMENTS = ["compression", "memory", "drift", "recovery", "intercept", "e2e"]
# --all deliberately excludes e2e: real API calls cost money.
_OFFLINE_EXPERIMENTS = ["compression", "memory", "drift", "recovery", "intercept"]


async def main() -> None:
    """Parse args, run the selected experiments, write reports."""
    parser = argparse.ArgumentParser(description="Tether benchmark runner")
    parser.add_argument(
        "--experiment",
        choices=_EXPERIMENTS + ["all"],
        default="all",
        help="Which experiment to run (all = the 5 offline experiments)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Shorthand for --experiment all",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of tasks to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiment = "all" if args.all else args.experiment
    selected = _OFFLINE_EXPERIMENTS if experiment == "all" else [experiment]
    configs = default_configs(args.num_samples)
    results_root = configs[selected[0]].output_dir
    report = ReportGenerator(results_root)

    for name in selected:
        config = configs[name]
        config.seed = args.seed
        logger.info("Running experiment: {}", name)
        runner = BenchmarkRunner(config)
        await runner.run_experiment()
        runner.save_results()
        md = report.generate_markdown(name)
        md_path = results_root / name / "report.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        logger.info("✅ Experiment {} complete -> {}", name, md_path)

    # Summary covers every experiment with results on disk (e.g. a real-LLM
    # e2e run from an earlier invocation), not just this invocation's runs.
    available = [
        name for name in _EXPERIMENTS
        if (results_root / name / "results.json").exists()
    ]
    summary = report.generate_table(available)
    summary_path = results_root / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary report -> {}", summary_path)


if __name__ == "__main__":
    asyncio.run(main())
