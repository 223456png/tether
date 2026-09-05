"""Tool base classes: ToolResult and the Tool ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Uniform return value for every tool execution."""

    success: bool
    output: str = ""
    error: str | None = None
    metadata: dict | None = None  # tool-specific metadata


class Tool(ABC):
    """Abstract base class for all Tether tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        """Return the JSON Schema description of this tool's parameters."""

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
