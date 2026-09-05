---
design_type: phase
created_at: 2026-09-05
---

# Phase 10: LLM 驱动的主循环 + 错误即观察 + 真实测试运行器

## Intent Contract

```
intent: 把 llm/ 层真正接入主循环（此前 _think 是随机 mock），让 Tether 从"上下文管理库"
        变成可运行的 agent 运行时；同时补齐工程安全性（注册表隔离、路径边界、缓存语义）
constraints: 86→93 个测试全绿且全部离线；mock 路径保持向后兼容（现有 monkeypatch 测试不改语义）；
             零新增运行时依赖；ruff 全绿
success_criteria: 注入 LLMProvider 后循环由结构化 tool call 驱动、模型停止调用工具即完成；
                  工具异常可配置为"观察"回传模型（带连续失败熔断）；run_test 为真实 pytest 子进程；
                  CLI 可一键演示
risk_level: medium（子进程执行 pytest；行为默认值保持向后兼容）
```

## Verification Contract

```
verify_steps:
  - run tests: python -m pytest tests/ -q → 93 passed
  - run lint: ruff check src tests → all checks passed
  - check: tests/test_agent_loop.py 用 ScriptedProvider 驱动完整循环（工具调用/完成信号/token 计量/停止）
  - check: tools 层测试覆盖路径穿越拒绝、原子写、拦截器失效、真实 pytest 通过/失败两态
```

## Scope

| In | Out |
|----|-----|
| `LLMProvider.complete(tools=...)` + `ToolCall` 解析 | 流式输出（SSE） |
| `TetherRuntime(llm_provider, max_steps, max_consecutive_failures)` | 多任务并发调度 |
| `AgentDecision` 统一 LLM 结构化路径与 legacy 字符串路径 | 重写 legacy 正则解析器（mock 专用，保留） |
| 错误即观察 + 连续失败熔断 | 自动重试队列（模型自适应优先） |
| 真实 pytest 子进程 `run_test` | Docker 沙箱（Windows 无 Docker） |
| per-runtime `ToolRegistry` + 默认注册表合并 | 插件市场 / 动态加载 |
| workspace 路径边界 + 原子写 + 拦截器失效语义 | 加密/审计日志 |
| JSONL 事件流 + CLI 入口 | Web UI |

## Decisions

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 1 | 工具调用格式 | OpenAI function calling（JSON arguments），LLM 路径零字符串解析 | 改进版正则（无法处理嵌套/转义）；JSON mode（兼容性差） |
| 2 | 完成信号 | 模型停止调用工具即视为完成（ReAct 惯例）+ max_steps 兜底 | 特殊 DONE 标记（模型遵从性差） |
| 3 | 错误处理默认值 | 默认 0 = 首错即失败（mock 大脑无法自适应，保持旧测试语义） | 默认观察模式（mock 路径会把 5 步预算烧在重试上） |
| 4 | 错误回传形态 | 观察文本进入 tool history（`ERROR: ...`），模型下一轮可见 | 立即 FAILED（模型失去自适应机会）；静默重试（掩盖根因） |
| 5 | run_test 实现 | `asyncio.to_thread(subprocess.run)`，cwd=workspace，超时杀进程 | `create_subprocess_exec`（受 Windows 事件循环策略影响）；持久 pytest 进程（复杂度不成比例） |
| 6 | 注册表作用域 | 每 runtime 一个实例；装饰器工具进模块级默认注册表，构造时合并且不可覆盖内置 | 全局单例（不同 workspace 互相覆盖工具） |
| 7 | 缓存正确性 | 只缓存 `side_effect_free` 且成功的结果；变更类工具执行后全量失效 | 按文件粒度失效（映射不可靠，宁可多跑一次只读） |
| 8 | token 估算 | `len//3`（离线可复现，保守） | tiktoken（引入网络/二进制依赖，破坏离线确定性） |

## Surface

**新增**：`runtime/events.py`（EventRecorder，JSONL 事件流）、`cli.py`（`tether run`）、
`tests/test_agent_loop.py`（脚本化 provider 驱动的循环集成测试）、`tests/test_cli.py`、
`py.typed`。

**修改**：`runtime/runtime.py`（AgentDecision 循环、错误观察、记忆写入钩子、事件埋点）、
`llm/base.py` `llm/openai_compat.py`（tools 参数 + ToolCall 解析 + 截断判定修正）、
`tools/registry.py`（去单例）、`tools/builtin/__init__.py`（路径边界、原子写、真实 run_test）、
`tools/intercept.py`（缓存语义）、`context/budget.py`（selected_episodic_ids 统计）、
`checkpoint/recovery.py`（lesson 笔记）。

## 后续方向（Roadmap）

1. BudgetConfig 分段比例落地（目前仅 total_budget 生效）——按 section 预算裁剪替代全局档位。
2. 工具结果截断升级为头+尾保留，并把各档位阈值做成可配置。
3. LLM 调用流式输出（SSE），让 CLI 实时显示思考过程。
4. 检查点压实（保留最近 N 条全量快照），避免长任务 JSONL 无界增长。
5. 执行沙箱化（容器/受限子进程），把"不是安全边界"的 TODO 变成保证。
