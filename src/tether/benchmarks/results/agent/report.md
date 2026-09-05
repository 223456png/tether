## agent 实验结果

### 配置
- 实验描述: Agent-level integration eval: TetherRuntime end-to-end over scripted goal-directed tasks (offline, deterministic)
- 运行日期: 2026-09-05T13:01:51+00:00
- 变体数量: 5

### 关键指标

| 变体 | 样本数 | 成功率 | 平均步骤 | Prompt Token | 总 Token | 平均延迟(ms) |
|------|--------|--------|----------|--------------|----------|--------------|
| create_verify | 4 | 100.00% | 3.00 | 300 | 345 | 72 |
| edit_existing | 4 | 100.00% | 4.00 | 400 | 455 | 94 |
| search_fix | 4 | 100.00% | 4.00 | 400 | 455 | 94 |
| plan_execute | 4 | 100.00% | 4.00 | 400 | 455 | 100 |
| test_report | 4 | 100.00% | 2.00 | 200 | 235 | 424 |

### 专项指标（全量聚合）

| 指标 | 值 |
|------|----|
| 任务完成率 | 100.00% (20/20) |
| create_verify | 4/4 通过 |
| edit_existing | 4/4 通过 |
| search_fix | 4/4 通过 |
| plan_execute | 4/4 通过 |
| test_report | 4/4 通过 |

### 结论

确定性策略驱动真实 TetherRuntime 循环完成 20 个合成任务，端到端成功率 100%（全部任务类型的端到端验证均通过）。本实验测的是运行时整环（上下文组装 → 结构化工具调用 → 记忆 → 检查点），不衡量模型智能。
