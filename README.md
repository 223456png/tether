# Tether

[![CI](https://github.com/223456png/tether/actions/workflows/ci.yml/badge.svg)](https://github.com/223456png/tether/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**面向长时运行 LLM 编码 Agent 的记忆与上下文管理层——做可验证安全性的 context engineering，所有数字可复现。**

Tether 是一个实验性的 Agent 运行时，回答一个很实际的问题：编码 Agent 一干就是几个小时，怎么让它的上下文保持精简、记忆保持正确、进度可恢复，而且不悄悄丢信息？

实现上是一条可独立测试的组件流水线：LLM 工具循环 → 三层记忆 → 预算化压缩 → 文件漂移检测 → 智能恢复 → 重复调用拦截 → MCP 生态接入。README 里的每个结论背后都有你能自己跑的 benchmark。

> **定位**：Tether 不追求 SOTA 压缩率（LLMLingua 这类学习式压缩器能到 10–20 倍）。它是确定性、零模型的流水线，价值在可验证的安全：每次压缩都过校验（ROUGE-L + 关键词存活），不达标就回滚。下面的 benchmark 量的就是这个取舍。

> **为什么这事在 2026 年重要**：业界已经把 compaction 当成长上下文增长的默认答案，也习惯了它的标志性失败模式——压缩之后 Agent 忘了"完成"的标准是什么，只对着可见的测试优化、原地打转补丁，最后报告一个假完成。Tether 的对策是结构性的，不是 prompt 层的：任务摘要和模型自维护的计划（`update_plan`）放在**永不裁剪**的记忆层，每轮回到上下文；每次压缩都有校验、丢信息就回滚；文件漂移跨重启检测（stat → MD5 → AST），恢复时知道 Agent 离开期间到底发生了什么。

## 架构

```mermaid
flowchart LR
    subgraph Runtime
        SM[State Machine<br/>think → tool → checkpoint] --> CP[Checkpoint Manager<br/>JSONL + workspace fingerprint]
        SM --> TR[Tool Registry<br/>+ duplicate-call interceptor]
    end
    subgraph Memory
        TS[TaskSummary] --> CA
        FS[FileSnapshot] --> CA
        EN[EpisodicNotes] --> CA
    end
    CA[Context Assembler] --> BA[BudgetAllocator<br/>5 trim levels]
    BA --> RV{ROUGE-L + keyword<br/>validation}
    RV -- fail --> RB[Rollback to full context]
    RV -- pass --> LLM([LLM Provider])
    DD[DriftDetector<br/>stat → MD5 → AST] --> SM
    RC[RecoveryManager] --> SM
```

| 阶段 | 组件 | 做什么 |
|-------|-----------|--------------|
| 1 | `runtime/` `checkpoint/` | 异步状态机、JSONL checkpoint、工具超时熔断 |
| 2–3 | `memory/` `context/` | 三层记忆（TaskSummary / FileSnapshot / EpisodicNotes）+ 预算化上下文组装（5 级裁剪 + 分节上限 + 头尾截断） |
| 4 | `context/validator.py` | ROUGE-L + 关键词压缩校验，失败回滚 |
| 5 | `filesystem/drift.py` | 三级文件漂移检测（stat → MD5 → AST 符号与签名） |
| 6 | `checkpoint/recovery.py` | 10 种中断场景的漂移感知恢复规划，已删文件支持内容还原 |
| 7 | `tools/` | 工具注册表 + 重复调用拦截（5 秒窗口，副作用安全） |
| 8 | `benchmarks/` | 7 个 benchmark 实验，含基线与消融（含 Agent 级集成评测） |
| 9 | `llm/` | OpenAI 兼容 provider 层（实测 DeepSeek）+ 离线 mock |
| 10 | `runtime/runtime.py` `llm/` | LLM 驱动的 Agent 循环：上下文组装 → OpenAI 式工具调用 → done 信号（离线测试保留 mock 降级） |
| 11 | `mcp.py` `tools/builtin` | MCP 客户端（stdio JSON-RPC，零依赖）：任意 MCP server 的工具进注册表；`update_plan` 让模型维护自己的计划 |

## Benchmark 结果

下面所有数字都由仓库内代码产出。六个离线实验跑 `python scripts/run_benchmark.py --all` 复现；真实 LLM 的 e2e 见[复现 e2e 实验](#复现-e2e-实验)。各实验的 Markdown 报告在 `src/tether/benchmarks/results/`；六个离线实验的原始 JSON/CSV 由同一命令（固定种子）确定性再生成；e2e 需要 API key，只提交了 Markdown 报告。

> **数字量的是什么**：压缩/记忆/漂移/恢复/拦截实验直接驱动*组件*（离线、固定种子、可复现）；**实验 7** 用确定性目标导向策略驱动*真实的* `TetherRuntime` 循环，补上了这个缺口。模型智能只由 e2e 实验（6）度量。

### 1. 上下文压缩（20 个 HumanEval 任务，离线校验）

| 策略 | Prompt tokens | 信息保全 |
|----------|--------------|----------------------|
| 全量上下文（基线） | 681 | 100% |
| Last-N 截断 | 537（−21%） | **65%**，早期工具结果被丢 |
| **BudgetAllocator（本文）** | **503（−22.5%）** | **100%**，ROUGE-L 校验，失败回滚 |

### 2. 真实 LLM 端到端 pass@1（20 个 HumanEval 任务 × 3 组，DeepSeek，greedy 解码）

每组把真实的 Agent 上下文（任务摘要、文件快照、工具历史、情节笔记）加任务发给 `deepseek-v4-flash`，产出对着官方 HumanEval 测试执行。

| 组 | pass@1 | 平均 prompt tokens | 平均总 tokens（含推理） | 平均延迟 |
|-----|--------|-------------------|-----------------------------------|-------------|
| 全量上下文 | 65%（13/20） | 795 | 4548 | 77.2s |
| Last-N | 80%（16/20） | 673 | 2760 | 49.0s |
| **BudgetAllocator（本文）** | **80%（16/20）** | 708（−11%） | 3609（**−21%**） | 62.3s |

结论：n=20 时 pass@1 差异在噪声内，但压缩组**从不劣于**全量上下文，总 token 成本省 21%——与"短上下文减少推理模型分心"（lost in the middle）的方向一致。多数未通过是轻量 flash 模型在 8192 token 上限处推理预算耗尽（空输出），不是代码写错。

### 3. 记忆消融（20 任务）

| 记忆配置 | 平均磁盘读/任务 | 过期读 | 任务成功率 |
|---------------|----------------------|-------------|--------------|
| 无记忆 | 30 | 0 | 100%（慢） |
| 平铺缓存 | 1 | 15.95 | **45%**，过期数据导致失败 |
| **三层（本文）** | **2** | **0** | **100%** |

### 4. 文件漂移检测（10 种变更类型 × 10 样本）

**100/100 准确**（0 误报、0 漏报），平均 **1.3ms**/文件。覆盖内容编辑、注释/空白变更、追加、删除、重命名、函数增删改、签名变更、并发多文件编辑。

### 5. 恢复（10 个中断场景 × 5 轮）

**整体 100%** 恢复成功，平均丢失 0.6 步。所有场景都能恢复，包括 `file_deleted`：Agent 碰过的文件（读或写）都有内容快照，外部删除可以通过写回内容撤销；Agent 没碰过的文件没有快照、恢复不了——检测器照样会标记，恢复失败时给出明确原因。

### 6. 重复调用拦截（20 个模拟任务）

**100% 拦截**（5 秒窗口内 60/60 次重复调用），省下 5100 token 的冗余工具输出。

### 7. Agent 级集成评测（20 个脚本任务 × 5 种类型，离线）

用确定性的目标导向策略代替 LLM，驱动**真实的 `TetherRuntime` 循环**（上下文组装 → 结构化工具调用 → 记忆写入 → checkpoint → 事件流）跑合成文件操作任务，按最终工作区状态判分。

| 任务类型 | 结果 |
|-----------|--------|
| create-and-verify | 4/4 |
| edit-existing | 4/4 |
| search-then-fix（闭环，根据观察反应） | 4/4 |
| plan-execute（`update_plan` → 执行） | 4/4 |
| test-and-report（真实 pytest） | 4/4 |
| **合计** | **100%（20/20），平均 3.4 步/任务** |

这个实验补上了其他实验留下的缺口：整条流水线作为一个整体、离线确定性跑通。首跑就抓到一个真 bug（多行工具输出在观察通道里被截断）——这正是集成评测存在的意义。

## 快速开始

```bash
git clone https://github.com/223456png/tether.git
cd tether
pip install -e ".[dev]"
python -m pytest tests/ -q          # 150 个测试，全离线
```

也可以从命令行驱动一个 Agent 任务（不需要 API key，会降级到离线 mock 大脑），再把事件流转成一页报告：

```bash
tether run --goal "Create hello.txt containing 'hi' and verify it" --workspace ./ws
tether report --events ./ws/logs/events.jsonl          # markdown 摘要
```

离线 mock 大脑会**真的执行这个目标**：写入 hello.txt → 读回验证 → 完成，报告末尾给出明确结论：

```
## Goal achievement

- ✓ **goal achieved** (verified against the final workspace state)
```

对 mock 无法执行的目标，它只跑一段简短 demo 并在最终答案里明说"没有追求该目标"——离线大脑绝不假装完成任务。

接任意 MCP server 的工具用 `--mcp-cmd`（可重复）：

```bash
tether run --goal "..." --workspace ./ws --mcp-cmd "python path/to/mcp_server.py"
```

真实使用的安全护栏：`--require-approval` 让写类工具在控制台确认后执行；`--max-tokens N` 给出硬成本上限（到量即停，状态 STOPPED，checkpoint 已保存）。

跑离线 benchmark：

```bash
python scripts/run_benchmark.py --all            # 6 个离线实验
python scripts/run_benchmark.py --experiment compression --num-samples 5
```

### 运行 LLM 驱动的 Agent 循环

```python
import asyncio
from pathlib import Path

from tether.llm.factory import create_provider_from_env
from tether.runtime.runtime import TetherRuntime

# 设置了 DEEPSEEK_API_KEY（或 TETHER_LLM_*）→ 真实 provider；
# 否则离线 MockProvider（is_mock=True），这段代码永远能跑。
provider, is_mock = create_provider_from_env()

runtime = TetherRuntime(
    goal="Create hello.txt containing 'hi' and verify it reads back",
    workspace_dir=Path("./my-workspace"),
    llm_provider=provider,
    max_steps=30,
)
asyncio.run(runtime.run())
print(runtime.state.status, runtime.state.final_answer)
print(runtime.state.total_tokens, "tokens")
```

每一轮：`ContextAssembler` 在 token 预算内从三层记忆组装系统上下文 → 模型返回 OpenAI 式工具调用或最终答案 → 工具经注册表执行（带重复调用拦截与工作区边界检查）→ 状态、记忆和 JSONL 事件流（`logs/events.jsonl`）更新 → checkpoint。

循环里值得知道的行为：

- **错误是观察，不是判决。** `max_consecutive_failures=N`（CLI 默认 3）时，工具异常或超时会回喂工具历史，让模型换参数重试；只有连续 N 次失败才熔断。默认 `0` 保持 mock 大脑的严格 fail-fast（它不会适应）。
- **`run_test` 是真的。** 在 workspace 里起 `python -m pytest <file>` 并返回通过/失败摘要——失败时 traceback 尾部作为错误返回，Agent 能看到错在哪、去修。
- **人审门。** `--require-approval`（或 `approval_gate` 回调）时，写类工具（`write_file`、`run_test`）执行前暂停确认；拒绝变成 `DENIED` 观察返回给模型适应。
- **硬成本上限。** `--max-tokens N` 在累计 token 用量到 N 时停机（状态 STOPPED，checkpoint 已保存）。`--max-steps N` 撞上限同样停机为 STOPPED——步数和 token 是同一种"资源耗尽"语义，绝不会伪装成 completed。
- **模型自己维护计划。** `update_plan` 工具把 Agent 的计划写进永不裁剪的 TaskSummary 层，压缩后依然在，每轮回到上下文。
- **流式输出。** `--stream`（或 `on_llm_delta` 回调）把模型回答以 SSE 实时吐出——不支持流式的 provider 照常调用。

没有 API key 时循环跑在脚本/mock 大脑上，所有代码路径在 CI 里零成本可测。

### 复现 e2e 实验

```bash
export DEEPSEEK_API_KEY=sk-...            # 任意 OpenAI 兼容 key，经 TETHER_LLM_* 变量同样可用
export TETHER_LLM_MODEL=deepseek-v4-flash
python scripts/run_benchmark.py --experiment e2e --num-samples 20
```

没有 API key 时降级到离线 `MockProvider`，所有结果打上 `is_mock: true`——流水线在 CI 里零成本可跑。

## 局限（引用数字前先读）

- **e2e 样本量小**（20 任务 × 3 组）。≤15 个百分点的差异都在噪声内；按方向一致报告，不当作显著。
- **benchmark 驱动的是组件，不是整合后的循环**（见上文"数字量的是什么"）。
- **代码执行没有沙箱。** HumanEval 产出和测试跑在带超时的普通子进程里（业界常规，但别对着不可信的模型用）。工具限制文件访问在 workspace 内，但循环本身不是安全边界。
- **`file_deleted` 恢复只覆盖碰过的文件。** 读或写过的文件有内容快照（每个 ≤64KB）可在外部删除后还原；没碰过的文件恢复不了。
- **漂移测试集是自建的**（10 种变更 × 10 样本）。100% 准确说的是这十类变更全覆盖，不代表生产级泛化。
- **HumanEval 子集离线内置**（20 题），因为构建环境访问不了上游仓库。
- **压缩率约 22.5%，对比学习式压缩器的 10–20 倍**；换来的是硬性语义保全保证（每次压缩过 ROUGE-L + 关键词校验，否则回滚）。校验是 CJK-aware 的：`rouge_score` 把中文全当分隔符（纯中文恒 0 分、混合丢中文不扣分），Tether 的校验器对 CJK 走字符级 LCS 组合评分，中文上下文同样可验证。

## Roadmap

剩下的想法：

1. **沙箱化执行**——测试跑进容器，把"不是安全边界"的声明变成真正的保证。
2. **跨任务情节记忆**——笔记目前是任务内的，共享存储能让经验跨任务传递。
3. **对话压缩**——工具历史现在是窗口化的，LLM/确定性摘要可以拉长回忆视野。

近期已完成：

- ✅ 分节预算上限 + 头尾截断（压缩 −22.5%，校验 100% 通过）
- ✅ checkpoint 压缩——JSONL 历史有界
- ✅ MCP 客户端——任意 MCP server 的工具进注册表
- ✅ `update_plan`——模型在记忆里维护自己的计划
- ✅ 审批门 + token 预算——human-in-the-loop 加硬成本上限
- ✅ SSE 流式——provider → runtime 回调 → CLI `--stream` 全链路
- ✅ Agent 级集成评测——20 个脚本任务过真实循环，100% 通过（上文实验 7）

## 项目结构

```
src/tether/
├── runtime/        # 状态机 + LLM/mock 循环 + JSONL 事件流
├── checkpoint/     # JSONL checkpoint + 智能恢复
├── memory/         # TaskSummary / FileSnapshot / EpisodicNotes
├── context/        # BudgetAllocator（分级 + 分节上限）+ ROUGE-L 校验器
├── filesystem/     # DriftDetector（stat → MD5 → AST）
├── tools/          # 运行时注册表 + 副作用安全拦截器 + 真实 run_test
├── llm/            # OpenAI 兼容 provider（function calling）+ 离线 mock
├── mcp.py          # MCP 客户端（stdio JSON-RPC）
├── reporting.py    # events.jsonl → 单页 markdown 运行报告
├── cli.py          # tether run / tether report 命令行入口
└── benchmarks/     # 7 个实验、指标、报告、数据集（含 Agent 级评测）
tests/              # 150 个测试（全离线，含脚本 provider 循环 + MCP 往返）
```

## License

MIT，见 [LICENSE](LICENSE)。
