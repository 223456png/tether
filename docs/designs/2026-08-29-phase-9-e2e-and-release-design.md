---
design_type: phase
created_at: 2026-08-29
---

# Phase 9: 真实 LLM 端到端实验 + 开源发布

## Intent Contract

```
intent: 用真实 LLM (DeepSeek, OpenAI 兼容) 端到端实验替换代理指标（pass@1），并将项目发布为可验证的开源仓库（README + CI + GitHub）
constraints: 54 个现有测试必须全绿；无 API Key 时所有路径可跑（mock 降级）；不修改 Phase 1-7 核心行为；零新增运行时依赖
success_criteria: e2e 实验产出真实 pass@1 数据（≥5 样本 x 3 变体）；GitHub 仓库 CI 绿；README 含真实数据表
risk_level: medium（外部 API 依赖 + 子进程代码执行；无安全/隐私风险，成本 <$0.1）
```

## Verification Contract

```
verify_steps:
  - run tests: python -m pytest tests/ -q → 59+ passed（54 旧 + ≥5 新）
  - check: DEEPSEEK_API_KEY 未设置时 python scripts/run_benchmark.py --experiment e2e → mock 降级跑通
  - confirm: 设置 Key 后同命令 → 产出真实 pass@1 报告（JSON + CSV + Markdown）
  - confirm: GitHub Actions 首次运行 → 绿
```

## Governance Contract

```
approval_gates:
  - 设计批准（已完成：用户选择方案 A）
  - 真实 LLM 实验运行前（花费用户 API 费用，确认样本数）
  - GitHub 推送前（公开仓库，确认仓库名/可见性）
rollback: 全部为新增文件 + runner 内新分支；LLM 调用失败自动回退 mock；推送前可整体删除
ownership: 用户（API 费用与仓库归属）
```

## Scope

| In | Out |
|----|-----|
| LLM Provider 抽象（base / deepseek / mock / factory） | openai SDK 依赖（用标准库 http 直连） |
| E2E 实验（HumanEval → 真实 LLM → 子进程执行 → pass@1） | 多 agent / planning 重构 |
| README.md（英文，含真实数据表 + Limitations） | SWE-bench（需 Docker） |
| GitHub Actions CI（3.10/3.12 矩阵） | 中英双语 README |
| GitHub 插件直推发布 | 本地 git 安装 |

## Decisions

| # | Decision | Choice | Rejected Alternatives |
|---|----------|--------|----------------------|
| 1 | LLM 接入方式 | 标准库 `urllib` + `asyncio.to_thread`（零依赖） | openai SDK（重依赖）；httpx（新依赖） |
| 2 | E2E 评估指标 | pass@1（HumanEval 官方测试用例） | ROUGE-L 代理（已有）；pass@k（样本量不足） |
| 3 | 代码执行方式 | 子进程 + 10s 超时（HumanEval 标准做法，README 注明非沙箱） | Docker 沙箱（Windows 无 Docker） |
| 4 | 无 Key 行为 | 自动降级 MockProvider + 报告标注 | 直接报错（破坏 CI/离线可用性） |
| 5 | 发布方式 | GitHub MCP `push_files`（本机无 git） | winget 安装 git（额外依赖） |

## Surface

**新增 `tether/llm/`**：`base.py`（LLMProvider 协议 + LLMResponse 数据类，携带 token 用量）、`openai_compat.py`（DeepSeekProvider：REST 直连 + 指数退避重试 x3）、`mock.py`（MockProvider 包装现有随机行为）、`factory.py`（环境变量 `DEEPSEEK_API_KEY` / `TETHER_LLM_BASE_URL` / `TETHER_LLM_MODEL` → provider 实例）。

**扩展 `benchmarks/`**：config.py 新增 `E2EExperimentConfig`；datasets.py 增加 HumanEval 测试用例执行器（子进程）；runner.py 新增 `_run_e2e`（三变体：full / last_n / budget，压缩上下文后真实生成 + 执行验证）；report.py 增加 e2e 结论。

**仓库级**：README.md、LICENSE（MIT）、.github/workflows/ci.yml、.gitignore。

## Risks & Open Questions

- DeepSeek 网络可达性与上一轮 HumanEval 下载失败同类（网络受限环境）→ 失败自动回退 mock，不阻塞其他交付物
- 子进程执行 LLM 生成的代码存在本机风险 → HumanEval 标准做法 + 超时 + README 明确非沙箱警告；生成的代码仅含函数体
- bundled HumanEval subset（20 题）是否含 `test` / `entry_point` 字段 → 实现前先验证数据结构
