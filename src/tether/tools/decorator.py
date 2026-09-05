"""@register_tool decorator: register Tool classes or plain async functions.

Registrations go into the module-level default registry (process-wide,
decorator-friendly); each TetherRuntime merges those entries into its
own per-runtime registry at construction time.
"""

import inspect
from collections.abc import Callable
from typing import Any

from tether.tools.base import Tool, ToolResult
from tether.tools.registry import get_default_registry


class FunctionTool(Tool):
    """Adapter that wraps a plain async function as a Tool."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
    ) -> None:
        """Inspect the function signature to build the parameter schema."""
        self._func = func
        self.name = name
        self.description = description
        self._params = [
            p for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]

    def get_parameters_schema(self) -> dict:
        """Build a JSON Schema from the function signature."""
        properties = {
            p.name: {"type": "string"}
            for p in self._params
        }
        required = [p.name for p in self._params if p.default is p.empty]
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    async def execute(self, **kwargs) -> ToolResult:
        """Call the wrapped function and normalize its result."""
        result = await self._func(**kwargs)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(success=True, output=str(result))


def register_tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Any], Any]:
    """Register a Tool subclass or an async function into the registry.

    Usage::

        @register_tool(name="read_file", description="Read a file")
        class ReadFile(Tool): ...

        @register_tool(name="search", description="Search code")
        async def search_code(pattern: str) -> ToolResult: ...
    """

    def decorator(cls_or_func: Any) -> Any:
        # Tool subclass: customize naming and register an instance.
        if inspect.isclass(cls_or_func) and issubclass(cls_or_func, Tool):
            tool_name = name or cls_or_func.__name__
            tool = cls_or_func()
            tool.name = tool_name
            if description:
                tool.description = description
            get_default_registry().register(tool)
            return cls_or_func

        # Async function: wrap in a FunctionTool and register it.
        if inspect.iscoroutinefunction(cls_or_func):
            tool_name = name or cls_or_func.__name__
            tool_desc = description or (inspect.getdoc(cls_or_func) or "")
            get_default_registry().register(
                FunctionTool(cls_or_func, tool_name, tool_desc)
            )
            return cls_or_func

        return cls_or_func

    return decorator
