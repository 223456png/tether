"""FileSnapshot: per-file cache layer used for drift detection."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from tether.memory.base import MemoryEntry, new_entry_id

# Suffix -> language mapping for common file types.
_SUFFIX_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}

# Regex-based symbol extraction (AST-based extraction arrives in Phase 5).
_SYMBOL_PATTERNS = {
    "python": [
        re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.MULTILINE),
        re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.MULTILINE),
        re.compile(r"^\s*async\s+def\s+([A-Za-z_]\w*)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\bclass\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\bconst\s+([A-Za-z_$][\w$]*)\s*="),
    ],
    "typescript": [
        re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\bclass\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\bconst\s+([A-Za-z_$][\w$]*)\s*="),
        re.compile(r"\binterface\s+([A-Za-z_$][\w$]*)"),
    ],
    "java": [
        re.compile(r"\bclass\s+([A-Za-z_]\w*)"),
        re.compile(r"\binterface\s+([A-Za-z_]\w*)"),
        re.compile(r"\b(?:public|private|protected)\s+[\w<>\[\],\s]+\s+([A-Za-z_]\w*)\s*\("),
    ],
    "go": [
        re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
        re.compile(r"\btype\s+([A-Za-z_]\w*)\s+struct"),
    ],
}
_DEFAULT_SYMBOL_PATTERN = _SYMBOL_PATTERNS["javascript"]

# Signature patterns (name + parameters) used for drift severity checks.
_SIGNATURE_PATTERNS = {
    "python": [
        re.compile(r"^\s*async\s+def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE),
        re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE),
    ],
}


def infer_language(file_path: Path) -> str:
    """Infer the programming language from the file suffix."""
    return _SUFFIX_LANGUAGE.get(file_path.suffix.lower(), "unknown")


def extract_symbols(text: str, language: str) -> list[str]:
    """Extract top-level symbols (function/class names) via regex.

    Returns a de-duplicated list preserving first-seen order.
    """
    patterns = _SYMBOL_PATTERNS.get(language, _DEFAULT_SYMBOL_PATTERN)
    seen: set = set()
    symbols: list[str] = []
    for pattern in patterns:
        for name in pattern.findall(text):
            if name not in seen:
                seen.add(name)
                symbols.append(name)
    return symbols


def extract_signatures(text: str, language: str) -> list[str]:
    """Extract function signatures (``name(params)``) for drift severity.

    Signature changes (e.g. added parameters) keep the symbol set stable
    but alter signatures; the drift detector compares both. Whitespace in
    parameter lists is normalized so reformatting does not count as drift.
    """
    patterns = _SIGNATURE_PATTERNS.get(language)
    if not patterns:
        return []
    seen: set = set()
    signatures: list[str] = []
    for pattern in patterns:
        for name, params in pattern.findall(text):
            normalized = " ".join(params.split())
            signature = f"{name}({normalized})"
            if signature not in seen:
                seen.add(signature)
                signatures.append(signature)
    return signatures


def compute_md5(file_path: Path) -> str:
    """Compute the MD5 hex digest of a file (streamed in chunks)."""
    digest = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FileSnapshot(MemoryEntry, BaseModel):
    """Cached metadata about a workspace file, used by the drift detector."""

    entry_id: str = Field(default_factory=new_entry_id)
    task_id: str
    path: str
    md5: str
    size: int
    mtime: float
    language: str
    summary: str
    symbols: list[str] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "FileSnapshot":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, task_id: str, file_path: Path) -> "FileSnapshot":
        """Build a snapshot from a real file on disk.

        Computes MD5 (streamed, 8KB chunks), size, mtime; infers language
        from the suffix and extracts symbols via regex. ``summary`` is a
        placeholder of "filename: top symbols" (LLM summaries come later).
        """
        file_path = Path(file_path)
        stat = file_path.stat()
        language = infer_language(file_path)
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        symbols = extract_symbols(text, language)
        if symbols:
            summary = f"{file_path.name}: {', '.join(symbols[:5])}"
        else:
            summary = file_path.name
        return cls(
            task_id=task_id,
            path=file_path.as_posix(),
            md5=compute_md5(file_path),
            size=stat.st_size,
            mtime=stat.st_mtime,
            language=language,
            summary=summary,
            symbols=symbols,
            signatures=extract_signatures(text, language),
        )
