"""ToolRegistry: per-runtime tool registry (no process-global state).

Each ``TetherRuntime`` owns its own registry so runtimes bound to
different workspaces never clobber each other's tools. A module-level
*default* registry backs the ``@register_tool`` decorator; runtimes
merge its entries (without overwriting builtins) at construction.
"""

import inspect
import threading

from tether.tools.base import Tool


class ToolRegistry:
    """Registry of Tool instances scoped to one owner (usually a runtime)."""

    def __init__(self) -> None:
        """Create an empty registry with its own registration history."""
        self._tools: dict[str, Tool] = {}
        self._history: list[str] = []

    def register(self, tool: type[Tool] | Tool, overwrite: bool = True) -> bool:
        """Register a Tool subclass (instantiated here) or a ready instance.

        Returns True when the tool was registered, False when an existing
        registration was kept (``overwrite=False`` with a name collision).

        ``overwrite=False`` keeps the existing entry on name collisions —
        used when merging decorator-registered tools into a runtime that
        already bound its own builtin with the same name.
        """
        if inspect.isclass(tool):
            tool = tool()
        if not tool.name:
            raise ValueError("Tool must define a non-empty 'name'")
        if not overwrite and tool.name in self._tools:
            return False
        self._tools[tool.name] = tool
        self._history.append(tool.name)
        self._history = self._history[-100:]
        return True

    def get(self, name: str) -> Tool | None:
        """Return the tool registered under ``name`` (None if absent)."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_tools_schema(self) -> list[dict]:
        """Return name/description/parameters for every tool (function calling)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters_schema(),
            }
            for tool in self._tools.values()
        ]

    def get_openai_tools_schema(self) -> list[dict]:
        """Return tools in OpenAI function-calling payload format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_parameters_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def get_tools_prompt(self) -> str:
        """Return a plain-text tool listing for non-function-calling models."""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    def merge(self, other: "ToolRegistry", overwrite: bool = False) -> list[str]:
        """Copy ``other``'s tools into this registry; returns merged names."""
        for tool in other._tools.values():
            self.register(tool, overwrite=overwrite)
        return list(self._tools.keys())


# ---------------------------------------------------------------------
# Module-level default registry (backing the @register_tool decorator)
# ---------------------------------------------------------------------
_default_registry: ToolRegistry | None = None
_default_lock = threading.Lock()


def get_default_registry() -> ToolRegistry:
    """Return the process-wide default registry (created on first use)."""
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = ToolRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Drop the default registry (used by tests for isolation)."""
    global _default_registry
    with _default_lock:
        _default_registry = None
