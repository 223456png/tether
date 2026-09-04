"""CallInterceptor: blocks duplicate tool calls inside a time window.

Only *side-effect-free* tools are ever cached: replaying a write or a
test run from cache would hand the agent stale state as if it were the
live result. Failed results are never recorded either, so a retry after
a transient error executes for real.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from tether.tools.base import ToolResult

_MAX_RECORDS_PER_TOOL = 100


@dataclass
class CallRecord:
    """One recorded tool invocation."""

    tool_name: str
    params_hash: str  # MD5 of the sorted-params JSON
    timestamp: float
    result: Optional[ToolResult] = None


class CallInterceptor:
    """Detects identical tool calls repeated within ``window_seconds``.

    This is the anti-loop defense: an agent retrying the exact same
    read-only call with the exact same arguments gets the cached result
    instead of a fresh (wasteful) execution.
    """

    def __init__(self, window_seconds: int = 5) -> None:
        """Store the dedup window and init per-tool record lists."""
        self.window_seconds = window_seconds
        self._records: Dict[str, List[CallRecord]] = {}
        self._lock = threading.Lock()
        self._intercept_count = 0

    def _compute_hash(self, params: dict) -> str:
        """Hash params order-independently (sorted-key JSON -> MD5)."""
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(sorted_params.encode()).hexdigest()

    def check(
        self, tool_name: str, params: dict, cacheable: bool = True
    ) -> Optional[ToolResult]:
        """Return the cached result if this exact call happened recently.

        ``None`` means "not a duplicate, execute normally". Non-cacheable
        tools (writes, test runs) always return ``None``: their result
        depends on state the call itself may have changed.
        """
        if not cacheable:
            return None
        with self._lock:
            records = self._records.get(tool_name, [])
            now = time.time()
            # Drop expired records.
            records = [
                r for r in records if now - r.timestamp <= self.window_seconds
            ]
            self._records[tool_name] = records

            param_hash = self._compute_hash(params)
            for record in records:
                if record.params_hash == param_hash:
                    self._intercept_count += 1
                    logger.info(
                        "🚫 Intercepted duplicate call: {}({})",
                        tool_name, params,
                    )
                    return record.result
            return None

    def record(
        self, tool_name: str, params: dict, result: ToolResult, cacheable: bool = True
    ) -> None:
        """Record a completed call for future dedup checks.

        Skipped entirely for non-cacheable tools and for failed results
        (a retry after an error must really execute).
        """
        if not cacheable or not result.success:
            return
        with self._lock:
            param_hash = self._compute_hash(params)
            record = CallRecord(
                tool_name=tool_name,
                params_hash=param_hash,
                timestamp=time.time(),
                result=result,
            )
            self._records.setdefault(tool_name, []).append(record)
            # Keep memory bounded.
            self._records[tool_name] = self._records[tool_name][-_MAX_RECORDS_PER_TOOL:]

    def invalidate_all(self) -> None:
        """Drop every cached result (called after workspace mutations).

        A write can invalidate reads the interceptor has no way to map,
        so the safe move is to empty the read cache completely — the
        next duplicate read costs one real execution, never stale data.
        """
        with self._lock:
            self._records.clear()
            logger.debug("Interceptor cache invalidated (workspace mutation)")

    def clear(self, tool_name: Optional[str] = None) -> None:
        """Clear records (all tools, or one tool when ``tool_name`` given)."""
        with self._lock:
            if tool_name:
                self._records[tool_name] = []
            else:
                self._records.clear()
