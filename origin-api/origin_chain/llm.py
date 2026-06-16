"""Orchestration LLM wrapper — TNSA NGen (the GenChat convention).

GenChat serves its NGen models (NGen-4 Mini, NGen-4.1 Mini, …) from TNSA's hosted
inference API:

    POST https://api.tnsaai.com/infer
    { "model": "NGen-4-Mini", "prompt": "...", "system_prompt": "..." }
    -> { "response": "... text (may contain <think>…</think>) ..." }

It's a prompt-in / text-out interface — TNSA handles its own tool execution
server-side and does not expose OpenAI-style structured `tool_calls` to the
client. So to let NGen drive OriginChain tools we use a **prompt-based tool
protocol**: the tool list is described in the prompt, and the model replies with a
single JSON object to call a tool (parsed by the pipeline) or with a final answer.

Model defaults to ``NGen-4-Mini`` and can be overridden via ``ORIGIN_CHAIN_MODEL``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from .exceptions import ConfigError, LLMError

DEFAULT_MODEL = os.environ.get("ORIGIN_CHAIN_MODEL", "NGen-4-Mini")
DEFAULT_BASE_URL = os.environ.get("TNSA_API_URL", "https://api.tnsaai.com")

# Groq drives the tool-calling step (GenChat's instant-chat model — it supports
# OpenAI-style client-side tool_calls, which TNSA's NGen agent does not expose).
DEFAULT_TOOL_MODEL = os.environ.get(
    "ORIGIN_CHAIN_TOOL_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks, matching GenChat's NGen handling."""
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    return cleaned.strip()


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Render MCP tool descriptors into a compact prompt catalogue."""
    lines = []
    for t in tools:
        name = t.get("name")
        if not name:
            continue
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        params = ", ".join(props.keys()) or "(no args)"
        lines.append(f"- {name}({params}): {t.get('description', '').strip()}")
    return "\n".join(lines)


class OrchestratorLLM:
    """TNSA NGen inference wrapper used for the tool-calling loop.

    Args:
        model:    NGen model id. Defaults to ``NGen-4-Mini``.
        base_url: TNSA API base. Defaults to ``https://api.tnsaai.com``.
        api_key:  Optional TNSA key (``x-api-key``). Falls back to
                  ``TNSA_API_KEY`` / ``TNSA_API``; sent only if present.
        timeout:  Per-request timeout (seconds).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("TNSA_API_KEY") or os.environ.get("TNSA_API")
        self.timeout = timeout
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self._session.headers.update(headers)

    def infer(self, prompt: str, system_prompt: str = "") -> str:
        """One NGen inference turn. Returns the answer text (thinking stripped)."""
        return self.infer_verbose(prompt, system_prompt)[0]

    def infer_verbose(self, prompt: str, system_prompt: str = "") -> "tuple[str, str]":
        """Like :meth:`infer` but returns ``(answer, thinking)``.

        NGen `/infer` streams Server-Sent Events: ``{"thinking": …}`` reasoning
        tokens, ``{"token": …}`` answer tokens, and a final ``{"full_response": …}``.
        A plain-JSON response (the tool-model path) is also handled.
        """
        body: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        if system_prompt:
            body["system_prompt"] = system_prompt
        try:
            resp = self._session.post(
                f"{self.base_url}/infer", json=body, timeout=self.timeout, stream=True
            )
            resp.raise_for_status()
            if "text/event-stream" in resp.headers.get("content-type", ""):
                answer, thinking = self._consume_sse(resp)
                return strip_thinking(answer), thinking.strip()
            data = resp.json()
        except requests.RequestException as exc:
            raise LLMError(f"NGen inference failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"NGen returned invalid JSON: {exc}") from exc
        text = data.get("full_response") or data.get("response") or data.get("text") or ""
        return strip_thinking(text), str(data.get("thinking") or "").strip()

    @staticmethod
    def _consume_sse(resp: "requests.Response") -> "tuple[str, str]":
        """Accumulate ``(answer, thinking)`` from an NGen SSE stream."""
        full: Optional[str] = None
        tokens: List[str] = []
        thinking: List[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            if d.get("error"):
                raise LLMError(str(d["error"]))
            if d.get("thinking") is not None:
                thinking.append(str(d["thinking"]))
            if "full_response" in d:
                full = str(d["full_response"])
            for key in ("token", "content", "response", "text"):
                if key in d and d[key] is not None:
                    tokens.append(str(d[key]))
        answer = full if full is not None else "".join(tokens)
        return answer, "".join(thinking)

    def close(self) -> None:
        self._session.close()


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """If the model emitted a tool call, return ``{"tool", "arguments"}`` else None.

    Accepts a bare JSON object or one fenced in a ```json code block.
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "tool" in obj:
        return {"tool": obj["tool"], "arguments": obj.get("arguments") or {}}
    return None


def mcp_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool descriptors to the OpenAI/Groq `tools` schema."""
    converted = []
    for t in tools:
        name = t.get("name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema")
                    or t.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


class GroqToolCaller:
    """Groq chat-completions wrapper that drives the OriginChain tool calls.

    Groq's llama-4 (GenChat's instant-chat model) supports OpenAI-style
    client-side ``tool_calls`` — the piece TNSA's NGen agent does not expose — so
    it performs the actual OriginChain tool selection.

    Args:
        api_key: Groq key. Falls back to ``GROQ_API_KEY``.
        model:   Tool-calling model. Defaults to GenChat's instant model.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_TOOL_MODEL) -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ConfigError("GROQ_API_KEY is not configured (needed for tool calling).")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("the 'groq' package is required (pip install groq)") from exc
        self._client = Groq(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """One tool-calling turn. Returns the assistant message (may carry
        ``tool_calls``)."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — surface provider errors uniformly
            raise LLMError(str(exc)) from exc
        return resp.choices[0].message
