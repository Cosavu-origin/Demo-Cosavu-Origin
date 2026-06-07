"""Exceptions raised by the Cosavu ContextAPI client."""

from __future__ import annotations


class CosavuError(Exception):
    """Base class for all ContextAPI client errors."""


class AuthError(CosavuError):
    """Missing or invalid API key (HTTP 401/403)."""


class RateLimitError(CosavuError):
    """Rate limit exceeded (HTTP 429). Retry with backoff."""


class BadRequestError(CosavuError):
    """The request was malformed or rejected (HTTP 400)."""


class APIError(CosavuError):
    """The service returned an unexpected error (HTTP 5xx or transport)."""
