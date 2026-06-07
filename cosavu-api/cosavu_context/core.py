"""Core client for Cosavu ContextAPI.

ContextAPI sits between your app and any large LLM (GPT / Claude / etc.). You
hand it a bloated prompt; it returns a leaner one with the same intent and fewer
tokens, which you then forward to your model to cut cost and latency.

Typical use::

    from cosavu_context import Cosavu

    cosavu = Cosavu(api_key="csvu_...")            # or set COSAVU_API_KEY
    result = cosavu.optimize("Hey so um could you maybe ...")

    print(result.text)                              # lean prompt -> send to your LLM
    print(result.tokens_saved, result.reduction)    # 18, 0.34

Stan model tiers (pass via ``model=``):

    stan-1.5-mini-instant      fastest   | extractive compression
    stan-1.5-mini-thinking     balanced  | clean fluent rewrite (default)
    stan-1.5-mini-predictive   richest   | rewrite + adds implied context
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

from .exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    CosavuError,
    RateLimitError,
)
from .models import OptimizedContext

# ── public model tiers ───────────────────────────────────────────────────────
INSTANT = "stan-1.5-mini-instant"
THINKING = "stan-1.5-mini-thinking"
PREDICTIVE = "stan-1.5-mini-predictive"

MODELS = (INSTANT, THINKING, PREDICTIVE)
DEFAULT_MODEL = THINKING

DEFAULT_BASE_URL = "https://api.cosavu.com"


class Cosavu:
    """A thin, dependency-light client for the Cosavu ContextAPI ``/optimize``
    endpoint.

    Args:
        api_key:     Your Cosavu API token. Falls back to the ``COSAVU_API_KEY``
                     (or ``COSAVU_API``) environment variable.
        base_url:    Override the API host. Defaults to ``COSAVU_API_URL`` env or
                     ``https://api.cosavu.com``.
        timeout:     Per-request timeout in seconds.
        max_retries: Retries on 429 / 5xx / transport errors, with backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("COSAVU_API_KEY") or os.environ.get("COSAVU_API")
        if not self.api_key:
            raise AuthError(
                "No API key. Pass api_key=... or set the COSAVU_API_KEY environment variable."
            )
        self.base_url = (base_url or os.environ.get("COSAVU_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "X-API-Token": self.api_key,
                "User-Agent": "cosavu-context-python/1.0",
            }
        )

    # ── public API ───────────────────────────────────────────────────────────
    def optimize(self, prompt: str, model: str = DEFAULT_MODEL) -> OptimizedContext:
        """Optimize ``prompt`` and return an :class:`OptimizedContext`.

        Raises:
            BadRequestError: empty prompt or unknown model / malformed request.
            AuthError:       missing or invalid API key.
            RateLimitError:  rate limit exceeded after retries.
            APIError:        server or transport failure after retries.
        """
        if not prompt or not prompt.strip():
            raise BadRequestError("prompt is empty")
        if model not in MODELS:
            raise BadRequestError(
                f"unknown model {model!r}; choose one of {', '.join(MODELS)}"
            )

        payload = self._post("/optimize", {"prompt": prompt, "model_tier": model})
        return OptimizedContext.from_response(payload, model=model)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Cosavu":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── internals ────────────────────────────────────────────────────────────
    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = APIError(f"request to {url} failed: {exc}")
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise APIError(f"invalid JSON in response: {exc}") from exc

            if resp.status_code in (401, 403):
                raise AuthError(self._detail(resp) or "invalid or missing API token")
            if resp.status_code == 400:
                raise BadRequestError(self._detail(resp) or "bad request")
            if resp.status_code == 429:
                last_exc = RateLimitError(self._detail(resp) or "rate limit exceeded")
                self._backoff(attempt)
                continue
            if resp.status_code >= 500:
                last_exc = APIError(self._detail(resp) or f"server error {resp.status_code}")
                self._backoff(attempt)
                continue

            raise APIError(f"unexpected status {resp.status_code}: {self._detail(resp)}")

        assert last_exc is not None
        raise last_exc

    def _backoff(self, attempt: int) -> None:
        if attempt < self.max_retries:
            time.sleep(0.5 * (2 ** attempt))

    @staticmethod
    def _detail(resp: requests.Response) -> str:
        try:
            data = resp.json()
        except ValueError:
            return (resp.text or "").strip()
        if isinstance(data, dict):
            return str(data.get("detail") or data.get("error") or "").strip()
        return str(data)


# ── module-level convenience ──────────────────────────────────────────────────
def optimize(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OptimizedContext:
    """One-shot helper: optimize a single prompt without managing a client.

    >>> from cosavu_context import optimize
    >>> lean = optimize("please could you kindly summarize this ...").text
    """
    client = Cosavu(api_key=api_key, base_url=base_url)
    try:
        return client.optimize(prompt, model=model)
    finally:
        client.close()


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
]
