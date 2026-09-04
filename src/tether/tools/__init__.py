"""Tools package: registry, decorator, interceptor, builtins."""

from tether.tools.base import Tool, ToolResult
from tether.tools.decorator import FunctionTool, register_tool
from tether.tools.intercept import CallInterceptor, CallRecord
from tether.tools.registry import (
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)

__all__ = [
    "CallInterceptor",
    "CallRecord",
    "FunctionTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "get_default_registry",
    "register_tool",
    "reset_default_registry",
]
