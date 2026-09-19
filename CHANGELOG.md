# Changelog

All notable changes to Tether are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
