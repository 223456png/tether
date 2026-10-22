# 安全说明

## 报告安全问题

发现安全问题时请私下告诉我，不要开公开 issue。可以在这里提交草稿安全公告：
<https://github.com/223456png/tether/security/advisories/new>，或者直接联系维护者。

## 威胁模型与已知边界

Tether 是一个实验性的研究型运行时，不是加固过的沙箱。

- **代码执行没有沙箱。** `run_test` 会在工作区内起 `python -m pytest`，HumanEval 评测也会在普通子进程里执行模型生成的代码。别把 Tether 接到不可信的模型或工作区上。容器化执行在 roadmap 里。
- **工具的文件访问被限制在工作区内。** `read_file`、`write_file`、`search_code`、`run_test` 都会把路径解析到工作区下面，拒绝绝对路径、`..` 和指向根目录外的软链。这是一道防误操作的护栏，不是用来对抗恶意模型的安全边界。
- **`--mcp-cmd` 会执行本地命令。** 每个 `--mcp-cmd` 都会以工作区为当前目录起一个子进程，权限和你本人一样。只接你信得过的 MCP server。
- **API key。** `DEEPSEEK_API_KEY` / `TETHER_LLM_*` 请通过环境变量传（见 `.env.example`），不要提交到仓库。Tether 只会把它们发给配置好的 LLM 端点。

## 支持的版本

Tether 还在 1.0 之前，只维护最新的 `main`。
