"""Agent-level integration benchmark: task specs + a scripted policy.

The experiment drives the *real* ``TetherRuntime`` loop (context
assembly -> tool call -> memory writes -> checkpoint -> events) with a
deterministic, goal-directed policy standing in for the LLM. Success is
verified against the final workspace state, so a task fails when the
harness misbehaves (lost tool results, stale reads, broken writes) —
this measures the runtime, not model intelligence.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tether.llm.base import LLMResponse, ToolCall


def _tool_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        prompt_tokens=100,
        completion_tokens=10,
        tool_calls=[ToolCall(name=name, arguments=arguments, id="call")],
        provider="policy",
    )


def _final(text: str) -> LLMResponse:
    return LLMResponse(
        content=text, prompt_tokens=100, completion_tokens=25, provider="policy"
    )


@dataclass
class Step:
    """One scripted policy step: a tool call, optionally built from the
    previous step's observation (``dynamic`` receives the raw tool output)."""

    tool: str
    args: dict = field(default_factory=dict)
    dynamic: Callable[[str], dict] | None = None


class ScriptedPolicy:
    """A FIFO step machine; finishes with a summary once steps run out."""

    def __init__(self, steps: list[Step], answer: str = "task complete") -> None:
        self._steps = steps
        self._answer = answer
        self._i = 0

    def next_action(self, last_output: str) -> LLMResponse:
        if self._i >= len(self._steps):
            return _final(self._answer)
        step = self._steps[self._i]
        self._i += 1
        args = step.dynamic(last_output) if step.dynamic else dict(step.args)
        return _tool_response(step.tool, args)


class PolicyProvider:
    """LLMProvider stand-in driven by a ScriptedPolicy.

    The observation fed to the policy is the tail of the assembled
    context's recent-tools section — the same feedback channel a real
    LLM sees, so tool results really have to survive the loop.
    """

    name = "policy"
    model = "scripted-policy-1"

    def __init__(self, policy: ScriptedPolicy) -> None:
        self.policy = policy
        self.turns = 0

    @staticmethod
    def _last_observation(messages: list[dict]) -> str:
        system = messages[0].get("content", "") if messages else ""
        if "[RECENT TOOLS]" not in system:
            return ""
        lines = system.split("[RECENT TOOLS]", 1)[1].splitlines()
        # The last entry may span multiple lines (multi-line tool output):
        # take everything from its "- " header to the end of the section,
        # mirroring what a real LLM sees.
        last_start = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("- ") and "-> " in line:
                last_start = idx
        if last_start is None:
            return ""
        entry = "\n".join(lines[last_start:])
        return entry.split("-> ", 1)[1]

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.turns += 1
        return self.policy.next_action(self._last_observation(messages))


# ---------------------------------------------------------------------------
# Task templates
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    """One benchmark task template.

    ``steps(i)`` builds the scripted steps for run ``i`` (any per-run
    state stays local to that call); ``verify`` checks the end state of
    the workspace + runtime after the loop finishes.
    """

    name: str
    goal: str
    setup: Callable[[Path], None]
    steps: Callable[[int], list[Step]]
    verify: Callable[[Path, object, int], bool]
    answer: str = "task complete"


def _search_steps() -> list[Step]:
    """search -> read (path parsed from the search output) -> write fix."""
    holder: dict = {}

    def read_dynamic(observation: str) -> dict:
        import re

        match = re.search(r"([\w./\\-]+\.\w+):\d+:", observation)
        holder["path"] = match.group(1).replace("\\", "/") if match else ""
        return {"path": holder["path"]}

    def write_dynamic(content: str) -> dict:
        return {"path": holder["path"], "content": content.replace("BUG-7", "FIXED-7")}

    return [
        Step("search_code", {"pattern": "BUG-7"}),
        Step("read_file", dynamic=read_dynamic),
        Step("write_file", dynamic=write_dynamic),
    ]


def build_agent_tasks() -> list[AgentTask]:
    """The five task templates (parameterized per run index)."""
    tasks: list[AgentTask] = []

    # 1. create-and-verify -------------------------------------------------
    def setup_create(ws: Path) -> None:
        ws.mkdir(parents=True, exist_ok=True)

    def verify_create(ws: Path, runtime, run_idx: int) -> bool:
        target = ws / f"note-{run_idx}.txt"
        return target.exists() and f"todo-{run_idx}" in target.read_text(encoding="utf-8")

    tasks.append(AgentTask(
        name="create_verify",
        goal="Create a note file and read it back to confirm",
        setup=setup_create,
        steps=lambda i: [
            Step("write_file", {"path": f"note-{i}.txt", "content": f"todo-{i}: ship it"}),
            Step("read_file", {"path": f"note-{i}.txt"}),
        ],
        verify=verify_create,
    ))

    # 2. edit-existing -----------------------------------------------------
    def setup_edit(ws: Path) -> None:
        (ws / "config.ini").write_text("mode=dev\n", encoding="utf-8")

    def verify_edit(ws: Path, runtime, run_idx: int) -> bool:
        text = (ws / "config.ini").read_text(encoding="utf-8")
        return "mode=prod" in text and f"# rev {run_idx}" in text

    tasks.append(AgentTask(
        name="edit_existing",
        goal="Switch config.ini from mode=dev to mode=prod",
        setup=setup_edit,
        steps=lambda i: [
            Step("read_file", {"path": "config.ini"}),
            Step("write_file", {"path": "config.ini", "content": f"mode=prod\n# rev {i}\n"}),
            Step("read_file", {"path": "config.ini"}),
        ],
        verify=verify_edit,
    ))

    # 3. search-then-fix (closed-loop: reacts to the search observation) ---
    def make_setup_search(bug_file: str):
        def setup(ws: Path) -> None:
            (ws / "src").mkdir(parents=True, exist_ok=True)
            for name in ("alpha", "beta"):
                content = (
                    "def run():\n    return 'BUG-7'\n"
                    if name == bug_file
                    else "def run():\n    return 'ok'\n"
                )
                (ws / f"src/{name}.py").write_text(content, encoding="utf-8")
        return setup

    def verify_search(ws: Path, runtime, run_idx: int) -> bool:
        fixed_any = False
        for name in ("alpha", "beta"):
            text = (ws / f"src/{name}.py").read_text(encoding="utf-8")
            if "BUG-7" in text:
                return False  # the bug is still there
            if "FIXED-7" in text:
                fixed_any = True
        return fixed_any  # exactly the buggy file got the fix

    tasks.append(AgentTask(
        name="search_fix",
        goal="Find the file containing BUG-7 and fix it",
        setup=make_setup_search("alpha"),
        steps=lambda i: _search_steps(),
        verify=verify_search,
    ))

    # 4. plan-execute ------------------------------------------------------
    def setup_plain(ws: Path) -> None:
        ws.mkdir(parents=True, exist_ok=True)

    def verify_plan(ws: Path, runtime, run_idx: int) -> bool:
        target = ws / "report.txt"
        summary = runtime.memory_store.load_task_summary(runtime.state.task_id)
        plan_ok = summary is not None and summary.current_plan == [
            "write the report", "read it back", "report done",
        ]
        return plan_ok and target.exists() and "plan-ran" in target.read_text(encoding="utf-8")

    tasks.append(AgentTask(
        name="plan_execute",
        goal="Plan the work, write report.txt and confirm",
        setup=setup_plain,
        steps=lambda i: [
            Step("update_plan", {"plan": ["write the report", "read it back", "report done"]}),
            Step("write_file", {"path": "report.txt", "content": f"plan-ran-{i}"}),
            Step("read_file", {"path": "report.txt"}),
        ],
        verify=verify_plan,
    ))

    # 5. test-and-report ---------------------------------------------------
    def setup_test(ws: Path) -> None:
        tests = ws / "tests"
        tests.mkdir(parents=True, exist_ok=True)
        (tests / "test_pass.py").write_text(
            "def test_passes():\n    assert 2 + 2 == 4\n", encoding="utf-8"
        )

    def verify_test(ws: Path, runtime, run_idx: int) -> bool:
        answer = (runtime.state.final_answer or "").lower()
        ran = any("run_test" in action for _, action, _ in runtime._tool_history)
        return ran and answer.startswith("tests passed")

    tasks.append(AgentTask(
        name="test_report",
        goal="Run tests/test_pass.py and report the outcome",
        setup=setup_test,
        steps=lambda i: [Step("run_test", {"path": "tests/test_pass.py"})],
        answer="Tests passed, all good",
        verify=verify_test,
    ))

    return tasks
