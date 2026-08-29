"""Tools package: registry, decorator, interceptor, builtins."""

from tether.tools.base import Tool, ToolResult
from tether.tools.decorator import FunctionTool, register_tool
from tether.tools.intercept import CallInterceptor, CallRecord
from tether.tools.registry import ToolRegistry

__all__ = [
    "CallInterceptor",
    "CallRecord",
    "FunctionTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "register_tool",
]
