# Tether Benchmark 汇总报告

| 实验 | 变体 | 样本数 | 成功率 | 平均延迟(ms) |
|------|------|--------|--------|--------------|
| compression | full | 20 | 100.00% | 0 |
| compression | last_n | 20 | 65.00% | 17 |
| compression | budget | 20 | 100.00% | 139 |
| memory | no_memory | 20 | 100.00% | 0 |
| memory | flat | 20 | 45.00% | 0 |
| memory | layered | 20 | 100.00% | 0 |
| drift | content_logic | 10 | 100.00% | 2 |
| drift | comment_whitespace | 10 | 100.00% | 1 |
| drift | append_content | 10 | 100.00% | 2 |
| drift | file_deleted | 10 | 100.00% | 0 |
| drift | file_renamed | 10 | 100.00% | 0 |
| drift | function_renamed | 10 | 100.00% | 1 |
| drift | function_added | 10 | 100.00% | 2 |
| drift | function_removed | 10 | 100.00% | 1 |
| drift | signature_changed | 10 | 100.00% | 1 |
| drift | multi_file | 10 | 100.00% | 1 |
| recovery | process_crash | 5 | 100.00% | 0 |
| recovery | timeout | 5 | 100.00% | 0 |
| recovery | api_rate_limit | 5 | 100.00% | 0 |
| recovery | context_overflow | 5 | 100.00% | 0 |
| recovery | user_interrupt | 5 | 100.00% | 0 |
| recovery | file_modified | 5 | 100.00% | 0 |
| recovery | file_deleted | 5 | 100.00% | 0 |
| recovery | tool_failure | 5 | 100.00% | 0 |
| recovery | oom | 5 | 100.00% | 0 |
| recovery | dependency_failure | 5 | 100.00% | 0 |
| intercept | interceptor | 20 | 100.00% | 0 |
