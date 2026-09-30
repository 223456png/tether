# Tether Benchmark 汇总报告

| 实验 | 变体 | 样本数 | 成功率 | 平均延迟(ms) |
|------|------|--------|--------|--------------|
| compression | full | 20 | 100.00% | — |
| compression | last_n | 20 | 65.00% | — |
| compression | budget | 20 | 100.00% | — |
| memory | no_memory | 20 | 100.00% | — |
| memory | flat | 20 | 45.00% | — |
| memory | layered | 20 | 100.00% | — |
| drift | content_logic | 10 | 100.00% | — |
| drift | comment_whitespace | 10 | 100.00% | — |
| drift | append_content | 10 | 100.00% | — |
| drift | file_deleted | 10 | 100.00% | — |
| drift | file_renamed | 10 | 100.00% | — |
| drift | function_renamed | 10 | 100.00% | — |
| drift | function_added | 10 | 100.00% | — |
| drift | function_removed | 10 | 100.00% | — |
| drift | signature_changed | 10 | 100.00% | — |
| drift | multi_file | 10 | 100.00% | — |
| recovery | process_crash | 5 | 100.00% | — |
| recovery | timeout | 5 | 100.00% | — |
| recovery | api_rate_limit | 5 | 100.00% | — |
| recovery | context_overflow | 5 | 100.00% | — |
| recovery | user_interrupt | 5 | 100.00% | — |
| recovery | file_modified | 5 | 100.00% | — |
| recovery | file_deleted | 5 | 100.00% | — |
| recovery | tool_failure | 5 | 100.00% | — |
| recovery | oom | 5 | 100.00% | — |
| recovery | dependency_failure | 5 | 100.00% | — |
| intercept | interceptor | 20 | 100.00% | — |
| agent | create_verify | 4 | 100.00% | — |
| agent | edit_existing | 4 | 100.00% | — |
| agent | search_fix | 4 | 100.00% | — |
| agent | plan_execute | 4 | 100.00% | — |
| agent | test_report | 4 | 100.00% | — |
