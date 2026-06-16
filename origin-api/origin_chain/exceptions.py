"""Exceptions for the OriginChain integration."""

from __future__ import annotations


class OriginChainError(Exception):
    """Base class for all OriginChain errors."""


class MCPError(OriginChainError):
    """The OriginChain MCP server returned an error or failed to start."""


class ToolCallError(OriginChainError):
    """A tool call against OriginChain failed."""


class LLMError(OriginChainError):
    """The orchestration LLM call failed."""


class ConfigError(OriginChainError):
    """Missing or invalid configuration (env vars, credentials)."""
