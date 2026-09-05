"""Dataset loading and generation for benchmarks."""

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

_DATASETS_DIR = Path(__file__).parent / "datasets"
_HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"
)

# The 10 drift change types (drift-detection specific dataset).
DRIFT_CHANGE_TYPES = [
    "content_logic",       # code logic changed
    "comment_whitespace",  # comments/whitespace only
    "append_content",      # content appended
    "file_deleted",        # file removed
    "file_renamed",        # path changed
    "function_renamed",    # function renamed -> STRUCTURE
    "function_added",      # new function -> STRUCTURE
    "function_removed",    # function deleted -> STRUCTURE
    "signature_changed",   # signature changed -> STRUCTURE
    "multi_file",          # several files changed at once
]

# Expected DriftLevel per change type.
EXPECTED_DRIFT_LEVEL = {
    "content_logic": "CONTENT",
    "comment_whitespace": "CONTENT",
    "append_content": "CONTENT",
    "file_deleted": "MISSING",
    "file_renamed": "MISSING",
    "function_renamed": "STRUCTURE",
    "function_added": "STRUCTURE",
    "function_removed": "STRUCTURE",
    "signature_changed": "STRUCTURE",
    "multi_file": "CONTENT",
}

# The 10 recovery scenarios (error messages feed RecoveryManager inference).
RECOVERY_SCENARIOS = [
    ("process_crash", None),
    ("timeout", "Tool timeout: 'run_test' exceeded 10s"),
    ("api_rate_limit", "API rate limit: HTTP 429 from provider"),
    ("context_overflow", "ContextOverflow: token limit of 16000 exceeded"),
    ("user_interrupt", "KeyboardInterrupt: user requested stop"),
    ("file_modified", None),  # realized as a content edit
    ("file_deleted", None),  # realized as a file removal
    ("tool_failure", "Tool 'edit_file' failed: invalid arguments"),
    ("oom", "MemoryError: out of memory during generation"),
    ("dependency_failure", "pip install failed: dependency resolution error"),
]

_BASE_CODE = (
    "def alpha(x):\n"
    "    return x + 1\n"
    "\n"
    "\n"
    "def beta(y):\n"
    "    return y * 2\n"
    "\n"
    "\n"
    "class Service:\n"
    "    def run(self):\n"
    "        return 'ok'\n"
)


def load_humaneval(num_samples: int | None = None) -> list[dict]:
    """Load HumanEval problems.

    Tries the official remote JSONL first (short timeout); falls back to
    the bundled offline subset (20 problems) when the network is
    unavailable.
    """
    tasks = _try_download_humaneval()
    source = "remote"
    if not tasks:
        local = _DATASETS_DIR / "humaneval" / "humaneval_subset.jsonl"
        tasks = _read_jsonl(local)
        source = f"bundled subset ({len(tasks)} problems)"
    if num_samples:
        tasks = tasks[:num_samples]
    logger.info("HumanEval loaded | source={} tasks={}", source, len(tasks))
    return tasks


def _try_download_humaneval() -> list[dict]:
    """Best-effort remote fetch; returns [] on any failure."""
    try:
        import urllib.request

        with urllib.request.urlopen(_HUMANEVAL_URL, timeout=5) as resp:
            data = resp.read().decode("utf-8")
        path = _DATASETS_DIR / "humaneval" / "HumanEval.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return _read_jsonl(path)
    except Exception as exc:  # noqa: BLE001 - any network error -> fallback
        logger.debug("HumanEval download failed ({}); using bundled subset", exc)
        return []


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into dicts."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generate_drift_samples(
    samples_per_type: int = 10, seed: int = 42
) -> list[dict]:
    """Generate the drift-detection dataset (10 types x N samples).

    Each sample describes a base file, a mutation, and the expected
    DriftLevel after the mutation is applied.
    """
    rng = random.Random(seed)
    samples: list[dict] = []

    for change_type in DRIFT_CHANGE_TYPES:
        for i in range(samples_per_type):
            suffix = rng.randrange(1000, 9999)
            samples.append({
                "change_type": change_type,
                "sample_id": f"{change_type}_{i}",
                "base_code": _BASE_CODE,
                "mutation": change_type,
                "suffix": suffix,
                "expected_level": EXPECTED_DRIFT_LEVEL[change_type],
            })
    return samples


def apply_mutation(workspace: Path, sample: dict) -> dict:
    """Apply a drift sample's mutation to a fresh workspace file.

    Writes ``src/sample_mod.py`` with the base code and applies the
    mutation in place; returns the relative path plus rename target
    when applicable.
    """
    import os

    rel = f"src/{sample['sample_id']}_mod.py"
    path = workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample["base_code"], encoding="utf-8")

    def _touch(future: int = 120) -> None:
        """Push mtime forward to simulate time passing since the snapshot."""
        stat = path.stat()
        os.utime(path, (stat.st_atime + future, stat.st_mtime + future))

    mutation = sample["mutation"]
    info = {"path": rel, "renamed_to": None}

    if mutation == "content_logic":
        path.write_text(
            sample["base_code"].replace("return x + 1", "return x + 2"),
            encoding="utf-8",
        )
        _touch()
    elif mutation == "comment_whitespace":
        path.write_text(
            "# touched comment\n\n" + sample["base_code"],
            encoding="utf-8",
        )
        _touch()
    elif mutation == "append_content":
        with path.open("a", encoding="utf-8") as f:
            f.write("\n\n# appended trailing note\n")
        _touch()
    elif mutation == "file_deleted":
        path.unlink()
    elif mutation == "file_renamed":
        target = workspace / f"src/{sample['sample_id']}_renamed.py"
        path.rename(target)
        info["renamed_to"] = target.relative_to(workspace).as_posix()
    elif mutation == "function_renamed":
        path.write_text(
            sample["base_code"].replace("def alpha(", "def alpha2("),
            encoding="utf-8",
        )
        _touch()
    elif mutation == "function_added":
        path.write_text(
            sample["base_code"] + "\n\ndef added_func():\n    return 42\n",
            encoding="utf-8",
        )
        _touch()
    elif mutation == "function_removed":
        path.write_text(
            sample["base_code"].replace(
                "def beta(y):\n    return y * 2\n\n\n", ""
            ),
            encoding="utf-8",
        )
        _touch()
    elif mutation == "signature_changed":
        path.write_text(
            sample["base_code"].replace("def beta(y):", "def beta(y, z=0):"),
            encoding="utf-8",
        )
        _touch()
    elif mutation == "multi_file":
        # Two extra files changed alongside the main one.
        extra = workspace / f"src/{sample['sample_id']}_extra.py"
        extra.write_text("def extra_fn():\n    return 1\n", encoding="utf-8")
        path.write_text(
            sample["base_code"].replace("return 'ok'", "return 'done'"),
            encoding="utf-8",
        )
        _touch()
    return info


# ----------------------------------------------------------------------
# HumanEval execution verification (Phase 9: real-LLM e2e experiment)
# ----------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Extract code from a markdown-fenced block, if present.

    Unfenced text is returned verbatim: stripping whitespace would eat
    the function-body indentation HumanEval completions rely on.
    """
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip("\n")
    return text


def build_humaneval_program(task: dict, completion: str) -> str:
    """Assemble the executable HumanEval check program.

    Standard HumanEval layout: prompt + completion + test harness +
    ``check(entry_point)``. A syntactically invalid completion makes the
    subprocess fail, which counts as a miss — exactly what pass@1
    should measure.
    """
    code = strip_code_fence(completion)
    return f"{task['prompt']}\n{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"


def run_humaneval_check(
    task: dict, completion: str, timeout_seconds: float = 10.0
) -> dict:
    """Execute one completion against the task's official tests.

    Runs ``python <tmpfile>`` in a subprocess (NOT a sandbox — see the
    README limitations section) and reports pass/fail with the exit
    status and stderr tail for debugging.

    Returns ``{"passed": bool, "exit_code": int, "stderr_tail": str}``.
    """
    program = build_humaneval_program(task, completion)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(program)
        tmp_path = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "passed": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "exit_code": -1,
            "stderr_tail": f"timeout after {timeout_seconds}s",
        }
    finally:
        tmp_path.unlink(missing_ok=True)
