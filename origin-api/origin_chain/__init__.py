"""OriginChain integration — LLM tool calls over OriginChain, optimized by Cosavu.

Groq (llama-4) performs the OriginChain tool calls (ask / SQL / vector / full-text /
graph / schema); each result (the *context*) is run through Cosavu's ContextAPI to
trim tokens; then NGen-4 Mini writes the final answer from the optimized context.
(NGen's /infer is a closed agent with its own fixed tools, so Groq handles the
client-side tool calling.)

    from origin_chain import OriginChainPipeline

    with OriginChainPipeline() as pipe:        # reads OC_* + COSAVU_API_KEY (+ TNSA_API_KEY)
        result = pipe.run("Top 5 customers by spend last quarter?")
        print(result.answer)
        print(f"ContextAPI saved {result.tokens_saved} tokens")

Lower-level pieces are exported too:

    from origin_chain import OriginChainMCP, OrchestratorLLM
"""

from .config import load_env
from .exceptions import (
    ConfigError,
    LLMError,
    MCPError,
    OriginChainError,
    ToolCallError,
)
from .llm import (
    DEFAULT_MODEL,
    DEFAULT_TOOL_MODEL,
    GroqToolCaller,
    OrchestratorLLM,
    format_tools_for_prompt,
    mcp_tools_to_openai,
    parse_tool_call,
    strip_thinking,
)
from .mcp_client import OriginChainMCP
from .pipeline import OriginChainPipeline, PipelineResult, ToolStep

__version__ = "1.0.0"

__all__ = [
    "load_env",
    "OriginChainPipeline",
    "PipelineResult",
    "ToolStep",
    "OriginChainMCP",
    "GroqToolCaller",
    "OrchestratorLLM",
    "mcp_tools_to_openai",
    "format_tools_for_prompt",
    "parse_tool_call",
    "strip_thinking",
    "DEFAULT_MODEL",
    "DEFAULT_TOOL_MODEL",
    "OriginChainError",
    "MCPError",
    "ToolCallError",
    "LLMError",
    "ConfigError",
    "__version__",
]
