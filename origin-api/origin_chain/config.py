"""Tiny zero-dependency .env loader.

Walks up from a starting directory looking for `.env` files and loads
``KEY=VALUE`` lines into ``os.environ`` (without overwriting variables that are
already set, so real environment values always win). No external dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def load_env(start: Optional[str] = None, override: bool = False) -> List[str]:
    """Load the nearest `.env` (searching ``start`` and its parents) into the
    environment. Returns the list of files loaded.

    Args:
        start:    directory to start from (defaults to this file's folder, i.e.
                  the ``origin-api`` package root).
        override: if True, values in `.env` overwrite existing env vars.
    """
    base = Path(start) if start else Path(__file__).resolve().parent.parent
    loaded: List[str] = []
    seen: set[str] = set()
    # Search the start dir and each parent; first definition of a key wins.
    for directory in [base, *base.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            for k, v in _parse_env(candidate.read_text(encoding="utf-8")).items():
                if k in seen:
                    continue
                seen.add(k)
                if override or not os.environ.get(k):
                    os.environ[k] = v
            loaded.append(str(candidate))
    return loaded
