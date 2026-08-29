"""ToolRegistry: thread-safe singleton registry of Tool instances."""

import inspect
import threading
from typing import Dict, List, Optional, Type, Union

from tether.tools.base import Tool


class ToolRegistry:
    """Global singleton registry shared by all runtimes in the process."""

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        """Create the singleton instance on first access (double-checked)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._tools: Dict[str, Tool] = {}
                    instance._history: List[str] = []
                    cls._instance = instance
        return cls._instance

    def register(self, tool: Union[Type[Tool], Tool]) -> None:
        """Register a Tool subclass (instantiated here) or a ready instance."""
        if inspect.isclass(tool):
            tool = tool()
        if not tool.name:
            raise ValueError("Tool must define a non-empty 'name'")
        self._tools[tool.name] = tool
        self._history.append(tool.name)
        self._history = self._history[-100:]

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool registered under ``name`` (None if absent)."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_tools_schema(self) -> List[dict]:
        """Return name/description/parameters for every tool (function calling)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters_schema(),
            }
            for tool in self._tools.values()
        ]

    def get_tools_prompt(self) -> str:
        """Return a plain-text tool listing for non-function-calling models."""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (used by tests for isolation)."""
        with cls._lock:
            cls._instance = None
