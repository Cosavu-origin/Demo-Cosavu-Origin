"""Result types for the Cosavu ContextAPI client.

The wire response from ``/optimize`` carries some service-internal diagnostics
in its ``optimization_notes`` field. We deliberately do **not** surface those
raw notes; ``OptimizedContext`` exposes only the public, stable fields a caller
needs: the optimized text, token accounting, and the compression achieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _extract_compression(notes: Optional[str]) -> Optional[float]:
    """Pull just the public compression ratio (e.g. ``compression=42%``) out of
    the raw notes, ignoring everything else. Returns a 0..1 float or ``None``."""
    if not notes:
        return None
    m = re.search(r"compression=(\d+(?:\.\d+)?)%", notes)
    if not m:
        return None
    return round(float(m.group(1)) / 100.0, 4)


def _extract_sampling(notes: Optional[str]) -> Optional[Dict[str, Any]]:
    """Pull the public sampling parameters the optimizer reported — temperature,
    top_p and reasoning mode — out of the notes. Internal-only tokens (rewriter,
    device, …) are deliberately ignored. Returns a dict or ``None``."""
    if not notes:
        return None
    out: Dict[str, Any] = {}
    t = re.search(r"\btemp=(\d+(?:\.\d+)?)", notes)
    p = re.search(r"\btop_p=(\d+(?:\.\d+)?)", notes)
    mode = re.search(r"\bmode=([A-Za-z0-9_-]+)", notes)
    jepa = re.search(r"jepa_recoverability=(\d+(?:\.\d+)?)%", notes)
    if t:
        out["temperature"] = float(t.group(1))
    if p:
        out["top_p"] = float(p.group(1))
    if mode:
        out["reasoning_mode"] = mode.group(1)
    if jepa:
        out["jepa_recoverability"] = round(float(jepa.group(1)) / 100.0, 4)
    return out or None


@dataclass
class OptimizedContext:
    """A leaner, same-intent version of your prompt, ready to forward to any LLM.

    Attributes:
        text:              The optimized prompt. Send this to your big model.
        model:             The Stan model tier that produced it.
        original_tokens:   Estimated tokens of the input prompt.
        optimized_tokens:  Estimated tokens of ``text``.
        tokens_saved:      ``original_tokens - optimized_tokens``.
        reduction:         Fraction of input tokens removed (0..1).
        compression:       Target compression the optimizer applied (0..1), if reported.
        latency_ms:        Server-side optimization time in milliseconds.
    """

    text: str
    model: str
    original_tokens: int
    optimized_tokens: int
    latency_ms: float
    compression: Optional[float] = None
    sampling: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.optimized_tokens

    @property
    def reduction(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(self.tokens_saved / self.original_tokens, 4)

    @classmethod
    def from_response(cls, payload: Dict[str, Any], model: str) -> "OptimizedContext":
        """Build a clean result from the raw API JSON, dropping internal notes."""
        blocks = payload.get("blocks") or []
        text = "\n\n".join(b.get("content", "") for b in blocks).strip()
        # Fallback for older shapes that may inline the optimized prompt.
        if not text:
            text = (payload.get("optimized_text") or payload.get("text") or "").strip()

        return cls(
            text=text,
            model=model,
            original_tokens=int(payload.get("total_original_tokens", 0)),
            optimized_tokens=int(payload.get("total_optimized_tokens", 0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            compression=_extract_compression(payload.get("optimization_notes")),
            sampling=_extract_sampling(payload.get("optimization_notes")),
            raw={k: v for k, v in payload.items() if k != "optimization_notes"},
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text
