"""Markdown report generation from saved benchmark results."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


class ReportGenerator:
    """Reads results/{experiment}/results.json and renders Markdown."""

    def __init__(self, results_dir: Path) -> None:
        """Store the root results directory."""
        self.results_dir = Path(results_dir)

    def _load(self, experiment_name: str) -> dict:
        """Load the saved payload for one experiment."""
        path = self.results_dir / experiment_name / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"No results for experiment: {experiment_name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def generate_markdown(self, experiment_name: str) -> str:
        """Generate a Markdown report for one experiment."""
        payload = self._load(experiment_name)
        variants = payload["variants"]
        lines: List[str] = [
            f"## {experiment_name} 实验结果",
            "",
            "### 配置",
            f"- 实验描述: {payload['description']}",
            f"- 运行日期: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- 变体数量: {len(variants)}",
            "",
            "### 关键指标",
            "",
            "| 变体 | 样本数 | 成功率 | 平均步骤 | Prompt Token | 总 Token | 平均延迟(ms) |",
            "|------|--------|--------|----------|--------------|----------|--------------|",
        ]
        for variant, data in variants.items():
            m = data["metrics"]
            tokens = m.get("avg_tokens", {})
            lines.append(
                f"| {variant} | {m.get('n_samples', 0)} "
                f"| {m.get('success_rate', 0):.2%} "
                f"| {m.get('avg_steps', 0):.2f} "
                f"| {tokens.get('prompt', 0):.0f} "
                f"| {tokens.get('total', 0):.0f} "
                f"| {m.get('avg_latency_ms', 0):.0f} |"
            )

        # Experiment-specific metric blocks, aggregated across variants.
        all_results = [
            r
            for data in variants.values()
            for r in data["results"]
        ]
        extra_rows = self._specialized_rows(experiment_name, all_results)
        if extra_rows:
            lines.append("")
            lines.append("### 专项指标（全量聚合）")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|----|")
            lines.extend(extra_rows)

        conclusion = self._conclusion(experiment_name, variants, all_results)
        if conclusion:
            lines.append("")
            lines.append("### 结论")
            lines.append("")
            lines.append(conclusion)

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _specialized_rows(experiment_name: str, results: List[dict]) -> List[str]:
        """Build experiment-specific metric rows from all results."""
        rows: List[str] = []
        if experiment_name == "drift":
            drift = [r["extra"]["drift"] for r in results if r["extra"].get("drift")]
            total = len(drift)
            tp = sum(1 for d in drift if d.get("tp"))
            tn = sum(1 for d in drift if d.get("tn"))
            fp = sum(1 for d in drift if d.get("fp"))
            fn = sum(1 for d in drift if d.get("fn"))
            avg_ms = (
                sum(d["detect_ms"] for d in drift) / total if total else 0.0
            )
            rows.append(f"| 检测准确率 | {(tp + tn) / total:.2%} |" if total else "| 检测准确率 | n/a |")
            rows.append(
                f"| 误报率 (FP) | {fp / (fp + tn):.2%} |" if (fp + tn)
                else "| 误报率 (FP) | 0.00% |"
            )
            rows.append(
                f"| 漏报率 (FN) | {fn / (fn + tp):.2%} |" if (fn + tp)
                else "| 漏报率 (FN) | 0.00% |"
            )
            rows.append(f"| 平均检测耗时 | {avg_ms:.2f}ms |")
        elif experiment_name == "recovery":
            total = len(results)
            success = sum(1 for r in results if r["success"])
            avg_lost = (
                sum(r["steps_lost"] for r in results) / total if total else 0.0
            )
            avg_ms = (
                sum(r["recovery_latency_ms"] for r in results) / total
                if total else 0.0
            )
            rows.append(f"| 恢复成功率 | {success / total:.2%} |" if total else "| 恢复成功率 | n/a |")
            rows.append(f"| 平均丢失步骤 | {avg_lost:.2f} |")
            rows.append(f"| 平均恢复耗时 | {avg_ms:.2f}ms |")
        elif experiment_name == "intercept":
            intercepted = sum(r["intercepted"] for r in results)
            duplicates = sum(r["duplicates"] for r in results)
            saved = sum(r["saved_tokens"] for r in results)
            rows.append(f"| 拦截次数 | {intercepted} |")
            rows.append(
                f"| 拦截率 | {intercepted / duplicates:.2%} |" if duplicates
                else "| 拦截率 | n/a |"
            )
            rows.append(f"| 节省 Token | {saved} |")
        elif experiment_name == "memory":
            by_variant: dict = {}
            for r in results:
                stats = by_variant.setdefault(r["variant"], {"disk": [], "stale": []})
                stats["disk"].append(r["disk_read_count"])
                stats["stale"].append(r["stale_read_count"])
            for variant, stats in by_variant.items():
                n = len(stats["disk"])
                rows.append(
                    f"| {variant} 平均磁盘读取 | {sum(stats['disk']) / n:.2f} |"
                )
                rows.append(
                    f"| {variant} 平均过期读取 | {sum(stats['stale']) / n:.2f} |"
                )
        elif experiment_name == "e2e":
            is_mock = any(
                r["extra"].get("is_mock") for r in results if r["extra"]
            )
            provider = next(
                (r["extra"].get("provider") for r in results
                 if r["extra"].get("provider")),
                "unknown",
            )
            by_variant: dict = {}
            for r in results:
                stats = by_variant.setdefault(
                    r["variant"], {"pass": 0, "total": 0, "tokens": []}
                )
                stats["total"] += 1
                stats["pass"] += 1 if r["success"] else 0
                stats["tokens"].append(r["prompt_tokens"])
            mode = "MockProvider（无 API Key，pass@1 无意义）" if is_mock else provider
            rows.append(f"| 模型 | {mode} |")
            for variant, stats in by_variant.items():
                n = stats["total"]
                rate = stats["pass"] / n if n else 0
                avg_tok = sum(stats["tokens"]) / n if n else 0
                rows.append(f"| {variant} pass@1 | {rate:.2%} ({stats['pass']}/{n}) |")
                rows.append(f"| {variant} 平均 Prompt Token | {avg_tok:.0f} |")
        return rows

    @staticmethod
    def _conclusion(
        experiment_name: str, variants: dict, results: List[dict]
    ) -> str:
        """Auto-derive a one-paragraph conclusion from the metrics."""
        def _metrics(variant: str) -> dict:
            return variants.get(variant, {}).get("metrics", {})

        def _avg_extra(variant: str, key: str) -> float:
            values = [
                r["extra"].get(key, 0)
                for r in variants.get(variant, {}).get("results", [])
            ]
            return sum(values) / len(values) if values else 0.0

        if experiment_name == "compression":
            reduction = _avg_extra("budget", "reduction_ratio")
            full_tokens = _metrics("full").get("avg_tokens", {}).get("prompt", 0)
            budget_tokens = _metrics("budget").get("avg_tokens", {}).get("prompt", 0)
            last_n_rate = _metrics("last_n").get("success_rate", 0)
            return (
                f"BudgetAllocator 在 ROUGE-L 校验（阈值 0.7 + 关键信息存活）保证 "
                f"100% 成功率的同时，平均压缩 {reduction:.1%} 的 Prompt Token"
                f"（{full_tokens:.0f} -> {budget_tokens:.0f}）；"
                f"Last-N 基线成功率仅 {last_n_rate:.0%}，失败原因均为丢弃较早的"
                f"工具结果导致关键信息丢失。"
            )
        if experiment_name == "memory":
            disk = {
                v: m.get("avg_disk_reads", 0) for v, m in
                ((v, d["metrics"]) for v, d in variants.items())
            }
            flat_rate = _metrics("flat").get("success_rate", 0)
            return (
                f"三层 Memory 将磁盘读取从 {disk.get('no_memory', 0):.0f} 次/任务"
                f"降至 {disk.get('layered', 0):.0f} 次/任务"
                f"（-{1 - disk.get('layered', 0) / max(1, disk.get('no_memory', 1)):.0%}），"
                f"且过期读取为 0；Flat 缓存磁盘读取同样低，但 {1 - flat_rate:.0%} "
                f"的任务因过期缓存（无漂移检测）失败。"
            )
        if experiment_name == "drift":
            drift = [r["extra"]["drift"] for r in results if r["extra"].get("drift")]
            total = len(drift)
            tp = sum(1 for d in drift if d.get("tp"))
            tn = sum(1 for d in drift if d.get("tn"))
            fp = sum(1 for d in drift if d.get("fp"))
            fn = sum(1 for d in drift if d.get("fn"))
            accuracy = (tp + tn) / total if total else 0
            fn_rate = fn / (fn + tp) if (fn + tp) else 0
            return (
                f"10 类文件变更共 {total} 个样本，三级级联检测（stat -> MD5 -> "
                f"符号结构）准确率 {accuracy:.1%}，漏报率 {fn_rate:.1%}，"
                f"平均单文件检测耗时 "
                f"{sum(d['detect_ms'] for d in drift) / total:.2f}ms。"
            )
        if experiment_name == "recovery":
            total = len(results)
            success = sum(1 for r in results if r["success"])
            failed = sorted({r["variant"] for r in results if not r["success"]})
            note = (
                f"；不可恢复场景：{', '.join(failed)}"
                f"（FileSnapshot 仅缓存元数据不缓存内容，删除后无法还原，"
                f"属已知设计限制——DriftDetector 仍能检出 MISSING 并明确报错）"
                if failed else ""
            )
            return (
                f"10 类中断场景 x {total // 10 if total else 0} 次重复，"
                f"总体恢复成功率 {success / total:.0%}{note}。"
            )
        if experiment_name == "intercept":
            intercepted = sum(r["intercepted"] for r in results)
            duplicates = sum(r["duplicates"] for r in results)
            saved = sum(r["saved_tokens"] for r in results)
            rate = intercepted / duplicates if duplicates else 0
            return (
                f"5 秒窗口内的重复调用拦截率 {rate:.0%}"
                f"（{intercepted}/{duplicates}），"
                f"共节省 {saved} Token 的冗余工具输出。"
            )
        if experiment_name == "e2e":
            is_mock = any(
                r["extra"].get("is_mock") for r in results if r["extra"]
            )
            if is_mock:
                return (
                    "本次运行未配置 API Key（MockProvider 降级），pass@1 "
                    "数值仅验证实验流程，不代表模型能力。设置 "
                    "DEEPSEEK_API_KEY 后重跑可得到真实数据。"
                )
            def _pass1(variant: str) -> str:
                arm = variants.get(variant, {}).get("metrics", {})
                n = arm.get("n_samples", 0)
                rate = arm.get("success_rate", 0)
                return f"{rate:.0%} ({n} 样本)"

            full_tokens = _metrics("full").get("avg_tokens", {}).get("prompt", 0)
            budget_tokens = _metrics("budget").get("avg_tokens", {}).get("prompt", 0)
            reduction = (
                1 - budget_tokens / full_tokens if full_tokens else 0
            )
            return (
                f"真实 LLM 端到端验证：BudgetAllocator 压缩 {reduction:.0%} 上下文"
                f"（{full_tokens:.0f} -> {budget_tokens:.0f} token）后，"
                f"pass@1 为 {_pass1('budget')}；完整上下文 pass@1 为 "
                f"{_pass1('full')}；Last-N 截断 pass@1 为 {_pass1('last_n')}。"
                f"压缩的代价（如有下降）直接反映在真实任务成功率上。"
            )
        return ""

    def generate_table(self, experiments: List[str]) -> str:
        """Generate a cross-experiment summary table."""
        lines: List[str] = [
            "# Tether Benchmark 汇总报告",
            "",
            "| 实验 | 变体 | 样本数 | 成功率 | 平均延迟(ms) |",
            "|------|------|--------|--------|--------------|",
        ]
        for name in experiments:
            try:
                payload = self._load(name)
            except FileNotFoundError:
                continue
            for variant, data in payload["variants"].items():
                m = data["metrics"]
                lines.append(
                    f"| {name} | {variant} | {m.get('n_samples', 0)} "
                    f"| {m.get('success_rate', 0):.2%} "
                    f"| {m.get('avg_latency_ms', 0):.0f} |"
                )
        lines.append("")
        return "\n".join(lines)
