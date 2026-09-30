"""Metric computation for benchmark results."""

from dataclasses import asdict, dataclass, field


@dataclass
class TaskResult:
    """Outcome of one benchmark task."""

    task_id: str
    variant: str  # experiment arm: full / last_n / budget / ...
    success: bool
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    file_read_count: int = 0
    disk_read_count: int = 0
    stale_read_count: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    steps_lost: int = 0
    recovery_latency_ms: float = 0.0
    saved_tokens: int = 0
    intercepted: int = 0
    duplicates: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens_computed(self) -> int:
        """Prompt + completion tokens (falls back to prompt when unset)."""
        return self.total_tokens or self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        """JSON-friendly serialization."""
        return asdict(self)


def _mean(values: list[float]) -> float:
    """Arithmetic mean (0.0 for empty lists)."""
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[rank - 1]


# ---- 墙钟观测字段（非种子确定性，重跑必变，不入版控结果）----
LATENCY_RESULT_FIELDS = ("latency_ms", "recovery_latency_ms")
LATENCY_METRIC_KEYS = (
    "avg_latency_ms", "p95_latency_ms", "avg_drift_detect_ms", "avg_recovery_ms",
)


def _strip_nested_latency(extra: dict) -> dict:
    """深拷贝 extra，剥掉嵌套的墙钟字段（如 drift.detect_ms）。"""
    cleaned = dict(extra)
    drift = cleaned.get("drift")
    if isinstance(drift, dict):
        cleaned["drift"] = {k: v for k, v in drift.items() if k != "detect_ms"}
    return cleaned


def strip_latency_result(result: dict) -> dict:
    """从单条 TaskResult 序列化中剥掉墙钟字段（含 extra 嵌套）。"""
    cleaned = {k: v for k, v in result.items() if k not in LATENCY_RESULT_FIELDS}
    if isinstance(cleaned.get("extra"), dict):
        cleaned["extra"] = _strip_nested_latency(cleaned["extra"])
    return cleaned


def strip_latency_metrics(metrics: dict) -> dict:
    """从聚合指标中剥掉墙钟字段。"""
    return {k: v for k, v in metrics.items() if k not in LATENCY_METRIC_KEYS}


def compute_metrics(results: list[TaskResult]) -> dict:
    """Aggregate a list of TaskResults into summary metrics.

    Only the metric families relevant to the experiment's TaskResults
    produce meaningful numbers; irrelevant fields default to 0.
    """
    if not results:
        return {"n_samples": 0}

    total = len(results)
    successes = sum(1 for r in results if r.success)
    avg_file_reads = _mean([r.file_read_count for r in results])
    avg_steps = _mean([r.steps for r in results])

    metrics: dict = {
        "n_samples": total,
        # Task level
        "success_rate": successes / total,
        "avg_steps": avg_steps,
        "avg_tokens": {
            "prompt": _mean([r.prompt_tokens for r in results]),
            "completion": _mean([r.completion_tokens for r in results]),
            "total": _mean([r.total_tokens_computed for r in results]),
        },
        # Memory level
        "avg_file_reads": avg_file_reads,
        "avg_file_reads_per_step": (
            avg_file_reads / avg_steps if avg_steps else 0.0
        ),
        "avg_disk_reads": _mean([r.disk_read_count for r in results]),
        "avg_stale_reads": _mean([r.stale_read_count for r in results]),
        # Cost level
        "total_cost": sum(r.cost for r in results),
        "avg_cost_per_task": sum(r.cost for r in results) / total,
        # Latency
        "avg_latency_ms": _mean([r.latency_ms for r in results]),
        "p95_latency_ms": _percentile([r.latency_ms for r in results], 95),
    }

    # Drift detection (only when populated by the drift experiment).
    drift_fields = [r.extra.get("drift") for r in results if r.extra.get("drift")]
    if drift_fields:
        tp = sum(1 for d in drift_fields if d.get("tp"))
        tn = sum(1 for d in drift_fields if d.get("tn"))
        fp = sum(1 for d in drift_fields if d.get("fp"))
        fn = sum(1 for d in drift_fields if d.get("fn"))
        metrics.update({
            "drift_accuracy": (tp + tn) / total,
            "drift_false_positive": fp / (fp + tn) if (fp + tn) else 0.0,
            "drift_false_negative": fn / (fn + tp) if (fn + tp) else 0.0,
            "avg_drift_detect_ms": _mean(
                [r.extra["drift"]["detect_ms"] for r in results]
            ),
        })

    # Recovery (only when populated by the recovery experiment).
    recovery_results = [r for r in results if r.extra.get("recovery")]
    if recovery_results:
        metrics.update({
            "recovery_success_rate": (
                sum(1 for r in recovery_results if r.success) / total
            ),
            "avg_steps_lost": _mean([r.steps_lost for r in results]),
            "avg_recovery_ms": _mean([r.recovery_latency_ms for r in results]),
        })

    # Interception (only when populated by the intercept experiment).
    intercept_results = [r for r in results if r.extra.get("intercept")]
    if intercept_results:
        dup_total = sum(r.duplicates for r in results)
        metrics.update({
            "intercept_count": sum(r.intercepted for r in results),
            "intercept_rate": (
                sum(r.intercepted for r in results) / dup_total if dup_total else 0.0
            ),
            "tokens_saved_by_intercept": sum(r.saved_tokens for r in results),
        })

    return metrics
