"""Experiment configurations for the Tether benchmark suite."""

from dataclasses import dataclass, field
from pathlib import Path

_BENCH_DIR = Path(__file__).parent
_RESULTS_DIR = _BENCH_DIR / "results"
_DATASETS_DIR = _BENCH_DIR / "datasets"


@dataclass
class ExperimentConfig:
    """Base configuration shared by all experiments."""

    name: str
    description: str
    dataset_path: Path
    num_samples: int | None = None  # None = run everything
    seed: int = 42
    output_dir: Path = _RESULTS_DIR
    tether_config: dict = field(default_factory=dict)


@dataclass
class CompressionExperimentConfig(ExperimentConfig):
    """Context compression experiment: full vs last-N vs BudgetAllocator."""

    baselines: list[str] = field(default_factory=lambda: ["full", "last_n", "budget"])
    budget_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class MemoryExperimentConfig(ExperimentConfig):
    """Memory ablation: none vs flat vs three-layer."""

    variants: list[str] = field(default_factory=lambda: ["no_memory", "flat", "layered"])


@dataclass
class DriftExperimentConfig(ExperimentConfig):
    """Drift detection: 10 change types x N samples each."""

    change_types: list[str] = field(default_factory=list)
    samples_per_type: int = 10


@dataclass
class RecoveryExperimentConfig(ExperimentConfig):
    """Recovery across the 10 interruption scenarios."""

    runs_per_scenario: int = 5


@dataclass
class InterceptExperimentConfig(ExperimentConfig):
    """Duplicate-call interception effectiveness."""

    task_steps: int = 5
    duplicate_calls: int = 3


@dataclass
class E2EExperimentConfig(ExperimentConfig):
    """Real-LLM end-to-end: compressed context -> generation -> pass@1.

    Uses the same three arms as the compression experiment, but the
    compressed context is sent to a real LLM whose completion is executed
    against the official HumanEval tests. Falls back to MockProvider
    (flagged in the report) when no API key is configured.
    """

    baselines: list[str] = field(default_factory=lambda: ["full", "last_n", "budget"])
    budget_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    default_num_samples: int = 5  # small by default: real API calls cost money
    execution_timeout: float = 10.0
    llm_model: str = "deepseek-chat"


def default_configs(num_samples: int | None = None) -> dict:
    """Build the six standard experiment configs.

    ``e2e`` defaults to ``default_num_samples`` (not all problems) when
    ``num_samples`` is None because each task x arm is a real API call.
    """
    return {
        "compression": CompressionExperimentConfig(
            name="compression",
            description="Context compression: full vs last-N vs BudgetAllocator",
            dataset_path=_DATASETS_DIR / "humaneval",
            num_samples=num_samples,
        ),
        "memory": MemoryExperimentConfig(
            name="memory",
            description="Memory ablation: none vs flat vs three-layer",
            dataset_path=_DATASETS_DIR / "humaneval",
            num_samples=num_samples,
        ),
        "drift": DriftExperimentConfig(
            name="drift",
            description="DriftDetector accuracy over 10 change types",
            dataset_path=_DATASETS_DIR / "drift",
            num_samples=num_samples,
        ),
        "recovery": RecoveryExperimentConfig(
            name="recovery",
            description="Recovery success across 10 interruption scenarios",
            dataset_path=_DATASETS_DIR / "recovery",
            num_samples=num_samples,
        ),
        "intercept": InterceptExperimentConfig(
            name="intercept",
            description="Duplicate-call interception rate",
            dataset_path=_DATASETS_DIR / "humaneval",
            num_samples=num_samples,
        ),
        "e2e": E2EExperimentConfig(
            name="e2e",
            description=(
                "Real-LLM end-to-end pass@1: compressed context -> DeepSeek "
                "generation -> official HumanEval test execution"
            ),
            dataset_path=_DATASETS_DIR / "humaneval",
            num_samples=num_samples if num_samples is not None else 5,
        ),
    }
