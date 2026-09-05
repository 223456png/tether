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

1. BudgetConfig 分段比例落地（目前仅 total_budget 生效）——按 section 预算裁剪替代全局档位。✅ 已完成：allocate() 第二遍按 ratio 上限裁剪超限 section（tool 保最新、file 逐级降级、episodic 走收缩梯子），stats["section_trimmed"] 可观测；enforce_section_caps=False 可关闭。
2. 工具结果截断升级为头+尾保留，并把各档位阈值做成可配置。✅ 已完成：_truncate_head_tail 头尾 2:1 分配可见预算，错误堆栈的结论（尾部）不再被砍掉；tool_truncate_chars / tool_minimal_chars 可配置；benchmark 数字重跑后与原表完全一致（681/537/574）。
3. LLM 调用流式输出（SSE），让 CLI 实时显示思考过程。（待做）
4. 检查点压实（保留最近 N 条全量快照），避免长任务 JSONL 无界增长。✅ 已完成：save_full 后自动 _compact，keep_last_checkpoints（默认 20）。
5. 执行沙箱化（容器/受限子进程），把"不是安全边界"的 TODO 变成保证。（待做）

### Round 3 补充决策

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 9 | 分段上限的实现位置 | allocate() 的第二遍（档位选择之后），render_at_level 保持纯档位语义 | 修改档位定义本身（破坏 benchmark 档位扫描的对照性） |
| 10 | cap 触发后的 tool 裁剪 | 从最新往旧保留（recency-first），至少保留 1 条 | 均匀抽样（丢失最新状态）；按 token 加权（复杂度不成比例） |
| 11 | 头尾比例 | 2:1（头部含触发输入，尾部含结论/traceback） | 1:1（头部信息密度通常更高）；只保留尾部 |
| 12 | 压实触发时机 | save_full 末尾、超过阈值即重写（保留最新 N 条 v2） | 定期后台压实（引入并发）；按大小触发（行为不可预测） |

### Round 4 补充决策：被删文件的还原能力

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 13 | 内容存储范围 | 仅 agent 触碰过（read/write）的文件 + ≤64KB 上限，opt-in include_content | 全部文件存内容（存储爆炸）；完全不存（file_deleted 永远 0%） |
| 14 | 还原执行点 | RecoveryManager._detect_file_drift 内：MISSING → 尝试从快照写回 → MATCH | 单独的还原阶段（多一遍扫描）；留给上层 runtime（恢复逻辑应自洽） |
| 15 | 还原后的重放计划 | 内容写回后 md5 一致 → MATCH → 不需要重放该文件相关步骤 | 保守重放（文件字节级一致，重放是浪费） |
| 16 | benchmark 口径 | 恢复实验改为内容感知快照（如实反映 runtime 新行为），README/Limitations 同步改写并保留"未触碰文件不可还原"的限制 | 只改数字不改口径（报告与能力脱节） |
| 17 | 压缩搜索空间 | 全局档位阶梯 + 工具段 recency-fit 减半梯子（25%/12.5%/6.25%，由 tool_result_ratio 推导，无调参魔法数），全部候选过同一 ROUGE-L + 关键词门槛，取最低 token 存活者 | 只扫全局档位（分段独立后浪费搜索空间）；调参式 fraction 网格（对 benchmark 过拟合）。效果：−15.8% → −22.1%（681→531），成功率保持 100%，延迟 29ms→139ms 为搜索的真实代价 |

### Round 6 补充决策：MCP 接入 / 模型自规划 / 运行报表

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 18 | MCP 传输实现 | 标准库 Popen + 行式 JSON-RPC，读线程 + executor 超时，asyncio.to_thread 包装 | asyncio.create_subprocess_exec（受 Windows 事件循环策略影响）；引入 mcp SDK（重依赖，违背零依赖约束） |
| 19 | MCP 工具命名 | `mcp_{server}_{tool}` 加前缀注册，register overwrite=False | 直接用远端名（与内置工具撞名会静默覆盖） |
| 20 | MCP 工具缓存语义 | 一律 side_effect_free=False（远端语义未知，宁可真执行） | 默认可缓存（旧结果风险不可控） |
| 21 | 规划载体 | `update_plan` 工具写 TaskSummary.current_plan（永不裁剪层），模型显式调用 | 解析模型回复文本中的计划（脆弱）；单独 planning pass（多一倍 LLM 调用） |
| 22 | CLI --mock 语义 | 无 key/--mock 时 llm_provider=None → 回退 mock thinker（会真实执行工具，可演示） | 把 MockProvider 当大脑（从不返回 tool call，第一步即"完成"，无演示价值）——本轮修复的退化 |
| 23 | 运行报表 | reporting.py 纯函数聚合 events.jsonl → markdown，CLI `tether report` | 读日志正则抽取（脆弱）；引入 dashboard 框架（过重） |

### Round 7 补充决策：审批门 / 成本熔断

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 24 | 审批门拦截点 | _execute_call 内、缓存检查之前；只拦 mutating 工具（默认 write_file/run_test，approval_tools 可配） | 拦全部工具（读也要人批，不可用）；放在 LLM 回复解析层（mock 路径失控） |
| 25 | 拒绝后的语义 | DENIED 作为观察回流（任务继续，模型可换路子），不计入连续失败熔断 | 拒绝即失败（用户否决 ≠ 工具故障）；静默跳过（模型不知道被拒） |
| 26 | 成本熔断时机 | 每轮 think 之前检查累计 token ≥ max_total_tokens → STOPPED（存检查点，可恢复） | FAILED（超预算不是故障）；think 内部抛异常（污染错误通道） |

### Round 8 补充决策：Agent 级集成评测 / SSE 流式

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 27 | agent 评测的"大脑" | 确定性规则策略（ScriptedPolicy + 观察通道解析），驱动真实 TetherRuntime 循环，按最终工作区状态判定成败 | MockProvider 当大脑（从不调工具，无评测意义）；预录制响应序列（非闭环，测不到观察通道）；真实 LLM（不可离线复现） |
| 28 | 策略的观察通道 | 与真实 LLM 同源：解析组装后 context 的 [RECENT TOOLS] 完整 entry（多行） | 内部直读 runtime 状态（绕过被测通道，评测失真）。首跑即抓到"多行工具输出被单行截断"的通道缺陷，评测价值自证 |
| 29 | 流式管线位置 | provider.complete(on_delta=...) 按能力探测传递（inspect 签名），runtime 存 on_llm_delta 回调，CLI --stream 打印 | 改 LLMProvider 协议强制全部实现流式（破坏现有 provider）；runtime 内置打印（不可测/不可换） |
