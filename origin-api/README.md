# OriginChain — LLM tool calls, optimized by Cosavu

Ask a question; an LLM performs an OriginChain tool call against your datastore, the
result is squeezed through **Cosavu's ContextAPI** (Stan), and a second model writes
the answer from that leaner context. Less context, same answer, fewer tokens.

```
question ─▶ Groq (tool call) ─▶ OriginChain ─▶ raw context
                                                 │
                                      Cosavu ContextAPI (optimize)
                                                 │
                            lean context ─▶ NGen-4 Mini ─▶ answer
```

**Two models, by necessity.** Both come from GenChat's stack:

- **Groq** (`llama-4`, GenChat's instant model) performs the **tool call** — it
  supports OpenAI-style client-side `tool_calls`.
- **NGen-4 Mini** (`api.tnsaai.com/infer`) writes the **final answer** from the
  optimized context.

Why split? TNSA's NGen `/infer` is a *closed agent* with its own fixed server-side
tools — it won't drive external (OriginChain) tools, so Groq handles tool selection
and NGen does the user-facing generation. OriginChain itself is reached over its MCP
server (`@originchain/mcp-server`), launched as a subprocess.

## Requirements

```bash
pip install -r requirements.txt   # groq + requests
```

Plus **Node.js / `npx`** on your PATH (used to launch the OriginChain MCP server).
The pipeline also uses the sibling [`cosavu-api`](../cosavu-api) package
(`cosavu_context`) — it's added to the path automatically from the repo layout.

## Configuration

| Env var          | For                          | Example                                        |
| ---------------- | ---------------------------- | ---------------------------------------------- |
| `OC_HOST`        | OriginChain datastore host   | `https://<tenant>.<region>.db.originchain.ai`  |
| `OC_TENANT`      | OriginChain tenant id        | `3k4z...`                                       |
| `OC_TOKEN`       | OriginChain bearer token     | `…` (keep secret)                              |
| `GROQ_API_KEY`   | Tool-calling LLM (Groq)      | `gsk_...`                                       |
| `COSAVU_API_KEY` | Cosavu ContextAPI            | `csvu_...`                                      |
| `ORIGIN_CHAIN_MODEL`      | (optional) NGen answer model | defaults to `NGen-4-Mini`             |
| `ORIGIN_CHAIN_TOOL_MODEL` | (optional) Groq tool model   | defaults to GenChat's instant model   |
| `TNSA_API_KEY`   | (optional) TNSA `x-api-key`  | sent only if set                               |

## Quickstart

```python
from origin_chain import OriginChainPipeline

with OriginChainPipeline() as pipe:            # reads the env vars above
    result = pipe.run("Which 5 customers spent the most last quarter?")
    print(result.answer)
    print(f"ContextAPI saved {result.tokens_saved} tokens before the LLM")
```

See [`example.py`](example.py) for a runnable version.

### Rich terminal chat

[`chat.py`](chat.py) is an interactive REPL that shows the internals of every turn —
**Source Context** (raw from OriginChain), **Optimized Context** (what ContextAPI sent
to the LLM, with token savings), **Sampling Parameters** (Stan's temperature / top_p /
reasoning mode), the model's **Thinking**, and the **Answer**:

```bash
pip install -r requirements.txt
export OC_HOST=... OC_TENANT=... OC_TOKEN=...
export GROQ_API_KEY=gsk_...  COSAVU_API_KEY=csvu_...
python chat.py
```

## What it does, step by step

1. Connects to the OriginChain MCP server and lists its tools (natural-language
   `ask`, SQL, vector search, full-text search, graph, schema listing).
2. Hands those tools to **Groq** with your question; Groq emits a `tool_call`.
3. Runs that tool against OriginChain and gets back the raw result — the **context**.
4. Sends that context to Cosavu's ContextAPI (`stan-1.5-mini-thinking` by default)
   to trim tokens while keeping intent (Groq may chain more tool calls).
5. Hands the optimized context to **NGen-4 Mini**, which writes the final answer.

`result.steps` records each tool call with its before/after token counts;
`result.tokens_saved` is the total trimmed before the LLM ever saw the data.

## Lower-level building blocks

Use the pieces directly if you don't want the full loop:

```python
from origin_chain import OriginChainMCP, OrchestratorLLM, mcp_tools_to_openai

# Talk to OriginChain directly
with OriginChainMCP() as oc:                   # reads OC_HOST / OC_TENANT / OC_TOKEN
    print([t["name"] for t in oc.list_tools()])
    rows = oc.call_tool("ask", {"question": "how many orders shipped today?"})
    print(rows)
```

| Symbol                | What it does                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `OriginChainPipeline` | The full Groq → OriginChain → ContextAPI → NGen loop.              |
| `OriginChainMCP`      | Minimal stdio JSON-RPC client for the OriginChain MCP server.      |
| `GroqToolCaller`      | Groq wrapper that performs the OriginChain `tool_calls`.           |
| `OrchestratorLLM`     | TNSA NGen (`/infer`) wrapper that writes the final answer.         |
| `mcp_tools_to_openai` | Convert MCP tool descriptors to the OpenAI/Groq `tools` schema.    |

Errors derive from `OriginChainError` (`MCPError`, `ToolCallError`, `LLMError`,
`ConfigError`).

## Notes

- The folder is `origin-api`, but the importable package is `origin_chain`. Add this
  folder to your `PYTHONPATH` (or run from inside it), then `import origin_chain`.
- Credentials (`OC_TOKEN` etc.) are passed to the MCP server through its environment
  only and are never logged. The MCP server's own stderr is discarded.
- To compare with and without optimization, construct the pipeline with
  `OriginChainPipeline(optimize=False)`.
