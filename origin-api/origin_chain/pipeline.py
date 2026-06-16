"""OriginChain → Cosavu ContextAPI → NGen pipeline (two-model split).

    1. Groq (llama-4) is given the OriginChain tools and the question, and performs
       the tool call(s) — ask / SQL / vector / FTS / graph / schema. (NGen can't:
       its /infer endpoint is a closed agent with its own fixed toolset.)
    2. Each tool runs against OriginChain via the MCP server — the result is the
       *context*.
    3. That context is sent through Cosavu's ContextAPI (Stan) to optimize it
       (fewer tokens, same intent) before it goes any further.
    4. The optimized context is handed to NGen-4 Mini, which writes the final
       user-facing answer.

    from origin_chain import OriginChainPipeline

    with OriginChainPipeline() as pipe:
        result = pipe.run("Which 5 customers spent the most last quarter?")
        print(result.answer)                 # written by NGen-4 Mini
        print(result.tokens_saved, "tokens saved by ContextAPI")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import OriginChainError
from .llm import GroqToolCaller, OrchestratorLLM, mcp_tools_to_openai
from .mcp_client import OriginChainMCP


def _import_cosavu():
    """Import the cosavu_context client, adding the sibling cosavu-api folder to
    sys.path if it isn't already importable (repo layout: ../cosavu-api)."""
    try:
        import cosavu_context  # noqa: F401
        return cosavu_context
    except ImportError:
        sibling = Path(__file__).resolve().parents[2] / "cosavu-api"
        if sibling.is_dir() and str(sibling) not in sys.path:
            sys.path.insert(0, str(sibling))
        import cosavu_context  # noqa: F401
        return cosavu_context


TOOL_SYSTEM_PROMPT = (
    "You are connected to an OriginChain datastore through the provided tools. "
    "When a question needs data, call the most appropriate tool (natural-language "
    "ask, SQL, vector search, full-text search, graph, or schema listing). The data "
    "IS available through the tools — never claim you lack access. Once you have the "
    "data you need, stop calling tools."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a data assistant. Answer the user's question using ONLY the data "
    "provided below it. Be concise and factual; do not invent values."
)


@dataclass
class ToolStep:
    """One tool round-trip: the OriginChain source context, the ContextAPI-optimized
    context, the savings, and the optimizer's sampling parameters."""

    tool: str
    arguments: Dict[str, Any]
    source_context: str = ""        # raw, as returned by OriginChain
    optimized_context: str = ""     # what was actually fed to the LLM
    original_tokens: int = 0
    optimized_tokens: int = 0
    optimized: bool = False
    sampling: Optional[Dict[str, Any]] = None  # Stan temp / top_p / reasoning_mode
    compression: Optional[float] = None

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.optimized_tokens


@dataclass
class PipelineResult:
    answer: str
    steps: List[ToolStep] = field(default_factory=list)
    thinking: str = ""              # the answer model's reasoning

    @property
    def tokens_saved(self) -> int:
        return sum(s.tokens_saved for s in self.steps)


class OriginChainPipeline:
    """Wires OriginChain (data), Cosavu ContextAPI (optimization), Groq (tool
    calling) and NGen (final answer).

    Args:
        mcp:         An ``OriginChainMCP`` (created from env if omitted).
        tool_llm:    A ``GroqToolCaller`` that performs the OriginChain tool calls.
        answer_llm:  An ``OrchestratorLLM`` (NGen) that writes the final answer.
        cosavu:      A ``cosavu_context.Cosavu`` client (created from env if omitted).
        model_tier:  Stan tier used to optimize tool results.
        optimize:    Set False to skip ContextAPI (passthrough — for comparison).
    """

    def __init__(
        self,
        mcp: Optional[OriginChainMCP] = None,
        tool_llm: Optional[GroqToolCaller] = None,
        answer_llm: Optional[OrchestratorLLM] = None,
        cosavu: Optional[Any] = None,
        model_tier: str = "stan-1.5-mini-thinking",
        optimize: bool = True,
        optimize_min_tokens: int = 200,
    ) -> None:
        self.mcp = mcp or OriginChainMCP()
        self.tool_llm = tool_llm or GroqToolCaller()
        self.answer_llm = answer_llm or OrchestratorLLM()
        self.model_tier = model_tier
        self.optimize = optimize
        # ContextAPI/Stan compresses bloated prose; running small or already-lean
        # results (e.g. a short list of identifiers) through it can drop real data.
        # Only optimize when the raw result is at least this many estimated tokens.
        self.optimize_min_tokens = optimize_min_tokens
        if cosavu is not None:
            self.cosavu = cosavu
        elif optimize:
            self.cosavu = _import_cosavu().Cosavu()
        else:
            self.cosavu = None

    def __enter__(self) -> "OriginChainPipeline":
        self.mcp.start()
        return self

    def __exit__(self, *exc) -> None:
        self.mcp.close()

    # ── main entry point ───────────────────────────────────────────────────
    def run(self, question: str, max_tool_rounds: int = 4) -> PipelineResult:
        """Answer ``question``: Groq calls OriginChain tools, every result is
        optimized through Cosavu's ContextAPI, and NGen writes the final answer."""
        if not question or not question.strip():
            raise OriginChainError("question is empty")

        self.mcp.start()
        tools = mcp_tools_to_openai(self.mcp.list_tools())

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: List[ToolStep] = []
        gathered: List[str] = []  # optimized context, in order

        # ── Stage 1: Groq performs the OriginChain tool call(s) ──────────────
        for _ in range(max_tool_rounds):
            msg = self.tool_llm.chat(messages, tools=tools)
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                raw = self.mcp.call_tool(name, args)
                context, step = self._optimize(name, args, raw)
                steps.append(step)
                gathered.append(f"[{name}]\n{context}")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": context}
                )

        # ── Stage 2: NGen-4 Mini writes the final answer from optimized data ─
        data_blob = "\n\n".join(gathered) if gathered else "(no data was retrieved)"
        answer_prompt = (
            f"QUESTION: {question}\n\nDATA (retrieved from OriginChain, optimized):\n{data_blob}"
        )
        answer, thinking = self.answer_llm.infer_verbose(
            answer_prompt, system_prompt=ANSWER_SYSTEM_PROMPT
        )
        return PipelineResult(answer=answer.strip(), steps=steps, thinking=thinking)

    # ── ContextAPI optimization ────────────────────────────────────────────
    def _optimize(self, name: str, args: Dict[str, Any], raw: str):
        step = ToolStep(tool=name, arguments=args, source_context=raw, optimized_context=raw)
        if not self.optimize or not self.cosavu or not raw.strip():
            return raw, step
        # Skip tiny/already-lean results — compressing them risks dropping data
        # for little or no token gain (Stan targets verbose prose, not short lists).
        if len(raw) // 4 < self.optimize_min_tokens:
            return raw, step
        try:
            result = self.cosavu.optimize(raw, model=self.model_tier)
        except Exception:  # noqa: BLE001 — never fail the pipeline on optimization
            return raw, step
        step.optimized = True
        step.original_tokens = result.original_tokens
        step.optimized_tokens = result.optimized_tokens
        step.optimized_context = result.text or raw
        step.sampling = result.sampling
        step.compression = result.compression
        return step.optimized_context, step
