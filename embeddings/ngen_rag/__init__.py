"""TNSA ngen-embeddings — vectors + RAG toolkit.

Convert text to vectors with TNSA's ``ngen-embeddings-v1`` model, save them to a
local vector store, and run retrieval-augmented generation.

    from ngen_rag import EmbeddingClient, RAG

    # raw embeddings
    client = EmbeddingClient()                 # reads TNSA_API_KEY
    vec = client.embed("The quick brown fox")

    # RAG
    rag = RAG()
    rag.add_texts(["... your documents ..."])
    rag.save("kb.npz")
    prompt = rag.build_prompt("what does the doc say about X?")
"""

from .client import DEFAULT_MODEL, EmbeddingClient
from .exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    EmbeddingsError,
    RateLimitError,
)
from .rag import RAG, chunk_text
from .store import SearchResult, VectorStore

__version__ = "1.0.0"

__all__ = [
    "EmbeddingClient",
    "VectorStore",
    "SearchResult",
    "RAG",
    "chunk_text",
    "DEFAULT_MODEL",
    "EmbeddingsError",
    "AuthError",
    "BadRequestError",
    "RateLimitError",
    "APIError",
    "__version__",
]
