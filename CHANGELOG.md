# Changelog

All notable changes to Tether are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **压缩保真校验对中文双向失效（第二轮审计 P0-1）**：`rouge_score` 的
  tokenizer 把一切非 ASCII 字符当分隔符——纯中文上下文相似度恒 0（永远
  回滚，可验证压缩静默失效）；混合内容丢光中文却得 1.000（信息静默丢失）。
  校验器改为 CJK-aware 组合相似度：ASCII 词走 rouge_score 的 ROUGE-L，
  CJK 字符走字符级 LCS F1（difflib），按原文构成加权——两个方向都被
  正确拦截，英文行为不变。中文回归测试锁定三种场景
- **token 估算对中文低估 2-3 倍（P0-2）**：`len(text) // 3` 让 CJK 重
  上下文在预算门放行后真实 token 超限。改为分段估算（CJK ≈1 token/字，
  其余 ≈1/4 字符），双向保守：中文不再低估，英文不再高估 33%
- **checkpoint load 遇 schema 漂移行直接崩（P1-1）**：坏行现在跳过并
  告警，恢复流程优先用更早的有效状态而不是硬失败
- **读路径漂移检测是死代码（P1-3）**：`_read_file_with_drift_check` 无
  任何调用点，文档宣称的读时漂移检测从未触发。改造为 `_track_read_file`
  并接入 read_file 成功后的执行路径（与 write 路径对称）：MATCH/METADATA
  快照保留，漂移快照失效并刷新内容快照。`read_file` 同时补
  `errors="replace"`（二进制内容不再炸穿 agent 循环）
- **MCP 工具重名静默丢弃却谎报注册成功（P2）**：`ToolRegistry.register`
  返回实际结果，`register_mcp_tools` 只列出真正注册的名字，重名跳过时
  显式告警
- **drift 检测 TOCTOU（P2）**：exists() 与 stat() 之间文件消失不再抛
  FileNotFoundError 穿透 agent 循环，按 MISSING 处理
- 审计对象说明：本轮审计基于 3890a11（旧 main），其中"步数上限 →
  COMPLETED / Quickstart 假完成 / 报告缺达成度"三项已在 606ada6 修复
- **Fake completion in the offline mock brain.** The mock thinker used to pick
  random, goal-blind action strings and let the loop report success without
  ever pursuing the goal (the documented Quickstart created no file yet ended
  "completed"). It now executes a scripted plan for goals shaped
  "Create <file> containing '<text>'" — actually writing, reading back, and
  verifying against the workspace — and for any other goal runs a short demo
  whose final answer states plainly that the goal was not pursued.
- **Step-cap semantics now match token-budget semantics.** Hitting
  `--max-steps` transitions the task to STOPPED (reason in `error_message`,
  checkpoint saved) instead of COMPLETED — resource exhaustion is never
  reported as success.
- **Run reports state whether the goal was achieved.** `summarize_events`
  renders a "Goal achievement" ✓/✗ verdict (from the new `goal_verified`
  event), warns when a run completes without executing any tools, and adds
  prominent warnings for failed tool calls and non-completed tasks.
- **LLM retry semantics.** 4xx responses other than 429 (bad key, wrong model,
  not found) now fail fast instead of burning the retry budget and masking the
  real cause. The `_RETRYABLE_STATUS` allow-list is finally enforced.
- **MCP transport timeout recovery.** A single read timeout no longer wedges
  the transport permanently: reads run on a dedicated reader thread feeding a
  queue, so the next request still succeeds on the same connection.
- **MCP process leak.** `TetherRuntime.connect_mcp` now closes the spawned
  server if tool registration fails, instead of leaking the subprocess.
- **Single-sourced version.** `tether.__version__` drives the package metadata
  (previously `pyproject.toml` said 0.2.0 while `__init__.py` said 0.1.0, and
  the MCP client hard-coded yet another string).

### Added

- `TetherRuntime.aclose()` — public shutdown API for MCP clients; the CLI and
  tests no longer touch the private `_mcp_clients` attribute.
- Regression tests: MCP read-timeout recovery, and LLM retry semantics
  (fail-fast on 4xx, retry on 429).
- CI: coverage reporting (`pytest-cov`), Python 3.11 in the test matrix,
  pip dependency caching, and concurrency cancellation.
- `CHANGELOG.md`, `.env.example`, `SECURITY.md`.

### Changed

- `pyproject.toml` metadata: `readme`, SPDX `license` + `license-files`,
  `authors`, `keywords`, `classifiers`, and `[project.urls]`.

## [0.2.0] - 2026-09-05

### Added

- MCP client (stdio JSON-RPC, zero dependencies): any MCP server's tools join
  the runtime registry as `mcp_{server}_{tool}`.
- `update_plan` builtin tool — the model maintains its own plan in the
  never-pruned TaskSummary layer, so it survives compression.
- `tether report` subcommand — aggregates a run's `events.jsonl` into a
  one-page Markdown report.
- Agent-level integration eval (20 scripted tasks × 5 types) driving the real
  runtime loop, offline and deterministic.
- SSE streaming end-to-end (provider → runtime callback → CLI `--stream`).
- Human-in-the-loop approval gate (`--require-approval`) and a hard token
  budget (`--max-tokens`).
- Per-section budget caps + head/tail truncation (compression 681 → 531
  tokens, −22% at 100% validation success).
- Content-backed snapshots that restore externally deleted files
  (recovery 90% → 100%).
- Checkpoint compaction (bounded JSONL history).

### Changed

- The LLM-driven agent loop replaces the random mock as the primary path; the
  offline mock brain is retained for tests and zero-cost demos.

## [0.1.0] - 2026-08-29

### Added

- Initial release: async state machine, three-layer memory, budget-allocated
  context assembly with ROUGE-L validation and rollback, three-level file
  drift detection (stat → MD5 → AST), drift-aware recovery, tool registry with
  duplicate-call interception, LLM provider layer, and six offline benchmark
  experiments.

[Unreleased]: https://github.com/223456png/tether/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/223456png/tether/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/223456png/tether/releases/tag/v0.1.0
