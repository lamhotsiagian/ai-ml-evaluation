"""A LangGraph ReAct agent instrumented for trajectory evaluation.

The agent itself is deliberately ordinary -- plan, call tools, observe, answer.
What makes it an *evaluation* target is that every decision is recorded as a
typed step, so the trajectory can be scored independently of the final answer.
That separation is the whole point of Chapter 5: an agent that reaches the right
answer through three wasted tool calls and one dangerous write is not a passing
agent, and outcome-only evaluation cannot see the difference.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Callable, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from evalcore.config import Settings, get_settings
from evalcore.llm import build_chat_model

StepKind = Literal["plan", "tool_call", "observation", "answer", "error", "limit"]


@dataclass
class TrajectoryStep:
    """One recorded decision in an agent run."""

    index: int
    kind: StepKind
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "step": self.index, "kind": self.kind, "tool": self.tool or "",
            "args": json.dumps(self.arguments)[:160], "content": self.content[:160],
            "latency_ms": round(self.latency_ms, 1), "error": self.error or "",
        }


@dataclass
class AgentRun:
    """Everything an agent produced, including how it got there."""

    task: str
    final_answer: str
    steps: list[TrajectoryStep]
    succeeded: bool
    total_latency_ms: float
    n_llm_calls: int
    stop_reason: str

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps if s.kind == "tool_call" and s.tool]

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_sequence)

    @property
    def n_errors(self) -> int:
        return sum(1 for s in self.steps if s.kind == "error" or s.error)

    def to_records(self) -> list[dict[str, Any]]:
        return [step.as_row() for step in self.steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "final_answer": self.final_answer,
            "succeeded": self.succeeded, "stop_reason": self.stop_reason,
            "n_tool_calls": self.n_tool_calls, "n_llm_calls": self.n_llm_calls,
            "n_errors": self.n_errors, "total_latency_ms": round(self.total_latency_ms, 1),
            "steps": [asdict(step) for step in self.steps],
        }


class AgentState(TypedDict):
    """LangGraph state. ``add_messages`` appends rather than replaces."""

    messages: Annotated[list[BaseMessage], add_messages]
    steps: list[TrajectoryStep]
    iterations: int
    stop_reason: str


_AGENT_SYSTEM = """You are a task-completing agent with tools.

Operating rules:
- Call a tool only when you cannot answer from what you already know or have observed.
- Never call the same tool with the same arguments twice; if a result was unhelpful,
  change the arguments or change approach.
- Read tool errors and adapt. Repeating a failing call is a failure, not persistence.
- When you have enough information, answer directly and stop calling tools.
- If the task cannot be completed with the available tools, say so plainly and stop."""


class EvaluableAgent:
    """A ReAct agent whose every step is captured for evaluation."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        max_iterations: int = 8,
        system_prompt: str = _AGENT_SYSTEM,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self._llm = build_chat_model(role="generation", model=model, settings=self.settings)
        self._llm_with_tools = self._llm.bind_tools(self.tools) if self.tools else self._llm
        self._graph = self._build_graph()
        self._llm_calls = 0

    # -- graph -------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason_node)
        graph.add_node("act", self._act_node)
        graph.add_edge(START, "reason")
        graph.add_conditional_edges("reason", self._route, {"act": "act", "end": END})
        graph.add_edge("act", "reason")
        return graph.compile()

    async def _reason_node(self, state: AgentState) -> dict[str, Any]:
        """One LLM turn: either request tools or produce the final answer."""
        if state["iterations"] >= self.max_iterations:
            step = TrajectoryStep(
                index=len(state["steps"]), kind="limit",
                content=f"iteration limit {self.max_iterations} reached",
            )
            return {"steps": state["steps"] + [step], "stop_reason": "iteration_limit"}

        started = time.perf_counter()
        response = await self._llm_with_tools.ainvoke(state["messages"])
        self._llm_calls += 1
        latency = (time.perf_counter() - started) * 1000

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            steps = state["steps"] + [
                TrajectoryStep(
                    index=len(state["steps"]) + offset, kind="tool_call",
                    tool=call["name"], arguments=dict(call.get("args", {})),
                    latency_ms=latency if offset == 0 else 0.0,
                )
                for offset, call in enumerate(tool_calls)
            ]
            return {"messages": [response], "steps": steps,
                    "iterations": state["iterations"] + 1, "stop_reason": ""}

        step = TrajectoryStep(
            index=len(state["steps"]), kind="answer",
            content=str(response.content), latency_ms=latency,
        )
        return {"messages": [response], "steps": state["steps"] + [step],
                "iterations": state["iterations"] + 1, "stop_reason": "answered"}

    async def _act_node(self, state: AgentState) -> dict[str, Any]:
        """Execute every requested tool call, recording failures as observations.

        A tool exception is fed back to the model as a ToolMessage rather than
        crashing the run. Recovery from a tool error is a capability worth
        measuring, and an agent that cannot recover should fail the evaluation
        on its trajectory -- not disappear from the results as an exception.
        """
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        messages: list[BaseMessage] = []
        steps = list(state["steps"])

        for call in tool_calls:
            name = call["name"]
            arguments = dict(call.get("args", {}))
            started = time.perf_counter()
            selected = self.tools_by_name.get(name)
            if selected is None:
                content = f"ERROR: unknown tool '{name}'. Available: {sorted(self.tools_by_name)}"
                error: str | None = "unknown_tool"
            else:
                try:
                    result = await selected.ainvoke(arguments)
                    content, error = str(result), None
                except Exception as exc:  # noqa: BLE001 - tool errors are observations
                    content, error = f"ERROR: {type(exc).__name__}: {exc}", str(exc)
            latency = (time.perf_counter() - started) * 1000

            steps.append(TrajectoryStep(
                index=len(steps), kind="observation", tool=name, arguments=arguments,
                content=content[:2000], latency_ms=latency, error=error,
            ))
            messages.append(ToolMessage(content=content[:4000], tool_call_id=call.get("id", name)))

        return {"messages": messages, "steps": steps, "stop_reason": ""}

    @staticmethod
    def _route(state: AgentState) -> str:
        if state["stop_reason"]:
            return "end"
        last = state["messages"][-1]
        return "act" if getattr(last, "tool_calls", None) else "end"

    # -- execution ---------------------------------------------------------
    async def arun(self, task: str) -> AgentRun:
        self._llm_calls = 0
        started = time.perf_counter()
        initial: AgentState = {
            "messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=task)],
            "steps": [], "iterations": 0, "stop_reason": "",
        }
        # recursion_limit guards against a cyclic graph independently of our own
        # iteration counter; both are needed because a tool can also loop.
        final = await self._graph.ainvoke(initial, {"recursion_limit": self.max_iterations * 3})

        answer = ""
        for message in reversed(final["messages"]):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                answer = str(message.content)
                break

        stop_reason = final.get("stop_reason") or "answered"
        return AgentRun(
            task=task,
            final_answer=answer,
            steps=final["steps"],
            succeeded=bool(answer) and stop_reason != "iteration_limit",
            total_latency_ms=(time.perf_counter() - started) * 1000,
            n_llm_calls=self._llm_calls,
            stop_reason=stop_reason,
        )

    def run(self, task: str) -> AgentRun:
        import asyncio
        return asyncio.run(self.arun(task))


# ---------------------------------------------------------------------------
# Deterministic evaluation tools
# ---------------------------------------------------------------------------
def build_evaluation_tools(corpus_search: Callable[[str, int], list[str]] | None = None) -> list[BaseTool]:
    """Tools with deterministic, verifiable behaviour.

    Every tool here returns a value that a test can assert on exactly. Agent
    evaluation with nondeterministic tools measures the tools as much as the
    agent, and the resulting suite is unusable as a regression gate.
    """

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '(1200 * 0.18) + 45'."""
        allowed = set("0123456789.+-*/() eE%")
        if not set(expression) <= allowed:
            raise ValueError(f"expression contains unsupported characters: {expression!r}")
        if "__" in expression:
            raise ValueError("expression rejected")
        # eval with empty builtins over a character-allowlisted arithmetic string.
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{value}"

    @tool
    def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
        """Convert between supported units (km, mi, kg, lb, c, f)."""
        table = {
            ("km", "mi"): lambda v: v * 0.621371,
            ("mi", "km"): lambda v: v / 0.621371,
            ("kg", "lb"): lambda v: v * 2.20462,
            ("lb", "kg"): lambda v: v / 2.20462,
            ("c", "f"): lambda v: v * 9 / 5 + 32,
            ("f", "c"): lambda v: (v - 32) * 5 / 9,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key not in table:
            raise ValueError(f"unsupported conversion {from_unit}->{to_unit}; supported: {sorted(table)}")
        return f"{table[key](value):.4f} {to_unit}"

    @tool
    def knowledge_search(query: str, k: int = 3) -> str:
        """Search the indexed corpus for passages relevant to a query."""
        if corpus_search is None:
            raise RuntimeError("knowledge_search requires an indexed corpus; build the index first")
        passages = corpus_search(query, k)
        if not passages:
            return "NO_RESULTS"
        return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))

    @tool
    def date_difference(start_iso: str, end_iso: str) -> str:
        """Days between two ISO dates (YYYY-MM-DD)."""
        from datetime import date
        start = date.fromisoformat(start_iso)
        end = date.fromisoformat(end_iso)
        return str((end - start).days)

    tools: list[BaseTool] = [calculator, unit_convert, date_difference]
    if corpus_search is not None:
        tools.append(knowledge_search)
    return tools
