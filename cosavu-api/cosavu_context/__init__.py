"""Cosavu ContextAPI — Python client for Stan prompt optimization.

Optimize bloated prompts into lean, same-intent ones before forwarding them to
your LLM of choice. Talks to https://api.cosavu.com.

    from cosavu_context import Cosavu, optimize, THINKING

    # one-shot
    lean = optimize("hey could you maybe summarize this for me ...").text

    # reusable client
    cosavu = Cosavu()                      # reads COSAVU_API_KEY
    result = cosavu.optimize(prompt, model=THINKING)
    print(result.text, result.reduction)
"""

from .core import (
    DEFAULT_MODEL,
    INSTANT,
    MODELS,
    PREDICTIVE,
    THINKING,
    Cosavu,
    optimize,
)
from .exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    CosavuError,
    RateLimitError,
)
from .models import OptimizedContext

__version__ = "1.0.0"

__all__ = [
    "Cosavu",
    "optimize",
    "OptimizedContext",
    "MODELS",
    "DEFAULT_MODEL",
    "INSTANT",
    "THINKING",
    "PREDICTIVE",
    "CosavuError",
    "AuthError",
    "BadRequestError",
    "RateLimitError",
    "APIError",
    "__version__",
]
