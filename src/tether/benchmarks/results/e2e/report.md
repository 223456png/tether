## e2e 实验结果

### 配置
- 实验描述: Real-LLM end-to-end pass@1: compressed context -> DeepSeek generation -> official HumanEval test execution
- 运行日期: 2026-08-29T08:10:09+00:00
- 变体数量: 3

### 关键指标

| 变体 | 样本数 | 成功率 | 平均步骤 | Prompt Token | 总 Token | 平均延迟(ms) |
|------|--------|--------|----------|--------------|----------|--------------|
| full | 20 | 65.00% | 0.00 | 795 | 4548 | 77200 |
| last_n | 20 | 80.00% | 0.00 | 673 | 2760 | 49019 |
| budget | 20 | 80.00% | 0.00 | 708 | 3609 | 62310 |

### 专项指标（全量聚合）

| 指标 | 值 |
|------|----|
| 模型 | openai_compat |
| full pass@1 | 65.00% (13/20) |
| full 平均 Prompt Token | 795 |
| last_n pass@1 | 80.00% (16/20) |
| last_n 平均 Prompt Token | 673 |
| budget pass@1 | 80.00% (16/20) |
| budget 平均 Prompt Token | 708 |

### 结论

真实 LLM 端到端验证：BudgetAllocator 压缩 11% 上下文（795 -> 708 token）后，pass@1 为 80% (20 样本)；完整上下文 pass@1 为 65% (20 样本)；Last-N 截断 pass@1 为 80% (20 样本)。压缩的代价（如有下降）直接反映在真实任务成功率上。
