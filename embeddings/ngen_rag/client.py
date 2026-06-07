"""Client for TNSA's ngen-embeddings API.

Turns text into dense vectors via ``POST https://api.tnsaai.com/v1/embeddings``.
Auth is the ``x-api-key`` header.

    from ngen_rag import EmbeddingClient

    client = EmbeddingClient()                  # reads TNSA_API_KEY
    vec = client.embed("The quick brown fox")   # -> list[float]
    mat = client.embed(["a", "b", "c"])         # -> list[list[float]]
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Union

import requests

from .exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    RateLimitError,
)

DEFAULT_BASE_URL = "https://api.tnsaai.com/v1"
DEFAULT_MODEL = "ngen-embeddings-v1"


class EmbeddingClient:
    """Thin client for the ngen-embeddings endpoint.

    Args:
        api_key:     TNSA API key. Falls back to ``TNSA_API_KEY`` or ``TNSA_API``.
        base_url:    Override the API host. Defaults to ``TNSA_API_URL`` env or
                     ``https://api.tnsaai.com/v1``.
        model:       Embedding model id. Defaults to ``ngen-embeddings-v1``.
        timeout:     Per-request timeout in seconds.
        max_retries: Retries on 429 / 5xx / transport errors, with backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("TNSA_API_KEY") or os.environ.get("TNSA_API")
        if not self.api_key:
            raise AuthError(
                "No API key. Pass api_key=... or set the TNSA_API_KEY environment variable."
            )
        self.base_url = (base_url or os.environ.get("TNSA_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "User-Agent": "ngen-rag-python/1.0",
            }
        )

    # ── public API ───────────────────────────────────────────────────────────
    def embed(
        self, text: Union[str, Sequence[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Embed a single string or a batch.

        Returns a single vector for a ``str`` input, or a list of vectors (in the
        same order) for a list input.
        """
        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        if not inputs or any(not isinstance(t, str) or not t.strip() for t in inputs):
            raise BadRequestError("input must be a non-empty string or list of non-empty strings")

        payload = self._post(
            "/embeddings", {"model": self.model, "input": inputs[0] if single else inputs}
        )
        data = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in data]
        if not vectors:
            raise APIError("response contained no embeddings")
        return vectors[0] if single else vectors

    # convenience alias for batch readability
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        result = self.embed(list(texts))
        return result  # type: ignore[return-value]

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EmbeddingClient":
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

            if resp.status_code == 401:
                raise AuthError(self._detail(resp) or "invalid or missing API key")
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
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or "").strip()
            return str(data.get("detail") or err or "").strip()
        return str(data)
