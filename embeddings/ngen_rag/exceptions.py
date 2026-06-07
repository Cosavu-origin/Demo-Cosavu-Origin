"""Exceptions for the TNSA ngen-embeddings RAG toolkit."""

from __future__ import annotations


class EmbeddingsError(Exception):
    """Base class for all embeddings / RAG errors."""


class AuthError(EmbeddingsError):
    """Missing or invalid TNSA API key (HTTP 401)."""


class RateLimitError(EmbeddingsError):
    """Rate limit exceeded (HTTP 429). Retry with backoff."""


class BadRequestError(EmbeddingsError):
    """Malformed request, e.g. missing input/model (HTTP 400)."""


class APIError(EmbeddingsError):
    """Unexpected server or transport error (HTTP 5xx / network)."""
