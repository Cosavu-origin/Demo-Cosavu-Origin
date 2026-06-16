"""Minimal stdio JSON-RPC client for the OriginChain MCP server.

Speaks line-delimited JSON-RPC 2.0 over a subprocess's stdin/stdout — the same
handshake the server expects (`initialize` -> `notifications/initialized` ->
`tools/list` / `tools/call`). No external MCP SDK required.

The server is launched as:

    OC_HOST=... OC_TENANT=... OC_TOKEN=...  npx -y @originchain/mcp-server

Credentials are passed through the environment only; they are never logged.

    from origin_chain import OriginChainMCP

    with OriginChainMCP() as oc:        # reads OC_HOST / OC_TENANT / OC_TOKEN
        tools = oc.list_tools()
        result = oc.call_tool("ask", {"question": "how many orders shipped today?"})
        print(result)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional

from .exceptions import ConfigError, MCPError

DEFAULT_COMMAND = ["npx", "-y", "@originchain/mcp-server"]
PROTOCOL_VERSION = "2024-11-05"


class OriginChainMCP:
    """A connection to the OriginChain MCP server.

    Args:
        host:    OriginChain host. Falls back to ``OC_HOST``.
        tenant:  Tenant id. Falls back to ``OC_TENANT``.
        token:   Bearer token. Falls back to ``OC_TOKEN``.
        command: Override the launch command (default: ``npx -y @originchain/mcp-server``).
        timeout: Seconds to wait for a single response.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        tenant: Optional[str] = None,
        token: Optional[str] = None,
        command: Optional[List[str]] = None,
        timeout: float = 60.0,
    ) -> None:
        self.host = host or os.environ.get("OC_HOST")
        self.tenant = tenant or os.environ.get("OC_TENANT")
        self.token = token or os.environ.get("OC_TOKEN")
        if not (self.host and self.tenant and self.token):
            raise ConfigError(
                "OriginChain needs OC_HOST, OC_TENANT and OC_TOKEN "
                "(pass them in or set the environment variables)."
            )
        self.command = command or DEFAULT_COMMAND
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._lock = threading.Lock()
        self._initialized = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> "OriginChainMCP":
        if self._proc is not None:
            return self
        env = dict(os.environ, OC_HOST=self.host, OC_TENANT=self.tenant, OC_TOKEN=self.token)
        # Resolve the launcher to a full path so Windows finds npx.cmd / npm.cmd etc.
        command = list(self.command)
        resolved = shutil.which(command[0])
        if resolved:
            command[0] = resolved
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # server logs/diagnostics are not surfaced
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as exc:
            raise MCPError(
                f"could not launch {self.command[0]!r}; is Node.js / npx installed and on PATH?"
            ) from exc
        self._handshake()
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None
            self._initialized = False

    def __enter__(self) -> "OriginChainMCP":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ── public API ──────────────────────────────────────────────────────────
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the tools the server exposes (name, description, input schema)."""
        self._ensure_started()
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Invoke an OriginChain tool and return its textual result.

        Raises ToolCallError if the server reports the tool call failed.
        """
        self._ensure_started()
        result = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        text = _content_to_text(result.get("content", []))
        if result.get("isError"):
            from .exceptions import ToolCallError

            raise ToolCallError(text or f"tool {name!r} reported an error")
        return text

    # ── internals ────────────────────────────────────────────────────────────
    def _ensure_started(self) -> None:
        if self._proc is None or not self._initialized:
            self.start()

    def _handshake(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "origin-chain", "version": "1.0"},
            },
        )
        self._notify("notifications/initialized")
        self._initialized = True

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("MCP server is not running")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except BrokenPipeError as exc:
            raise MCPError("MCP server closed its input pipe") from exc

    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            req_id = self._next_id()
            msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                msg["params"] = params
            self._send(msg)
            return self._read_response(req_id)

    def _read_response(self, req_id: int) -> Dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise MCPError("MCP server is not running")
        # Read lines until we see the matching response id (skip notifications/logs).
        while True:
            line = proc.stdout.readline()
            if line == "":
                code = proc.poll()
                raise MCPError(f"MCP server exited unexpectedly (code {code})")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore non-JSON output on stdout
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                err = msg["error"] or {}
                raise MCPError(f"{err.get('message', 'JSON-RPC error')} (code {err.get('code')})")
            return msg.get("result", {}) or {}


def _content_to_text(content: List[Dict[str, Any]]) -> str:
    """Flatten MCP content blocks into a single text string."""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif "text" in block:
            parts.append(str(block["text"]))
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p).strip()
