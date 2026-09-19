# Security Policy

## Reporting a vulnerability

Please report security issues privately — open a draft advisory at
<https://github.com/223456png/tether/security/advisories/new> or contact the
maintainer directly. Do not open a public issue for a security problem.

## Threat model and known boundaries

Tether is an **experimental research runtime**, not a hardened sandbox.

- **Code execution is not sandboxed.** `run_test` spawns `python -m pytest`
  inside the workspace, and the HumanEval benchmark executes model-generated
  completions in plain subprocesses. Do not point Tether at untrusted models
  or untrusted workspaces. Containerized execution is on the roadmap.
- **Tool file access is workspace-bounded.** `read_file`, `write_file`,
  `search_code` and `run_test` resolve paths under the workspace and refuse
  escapes (absolute paths, `..`, symlinks pointing outside the root). This is
  a correctness guardrail, not a security boundary against a hostile model.
- **`--mcp-cmd` executes a local command.** Each `--mcp-cmd` value is spawned
  as a subprocess with the workspace as its working directory. Only connect
  MCP servers you trust; the command runs with your user's privileges.
- **API keys.** Provide `DEEPSEEK_API_KEY` / `TETHER_LLM_*` through environment
  variables (see `.env.example`) and never commit them. Tether sends them only
  to the configured LLM endpoint.

## Supported versions

Tether is pre-1.0; only the latest `main` is supported.
