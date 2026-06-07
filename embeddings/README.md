# TNSA AI — Embeddings API

Generate dense vector embeddings from text using TNSA's `ngen-embeddings-v1` model. The API is OpenAI-compatible, so you can use the official `openai` SDK or plain HTTP.

## Base URL

```
https://api.tnsaai.com/v1
```

## Authentication

All requests require your TNSA API key. Pass it in the `x-api-key` header:

```
x-api-key: your_tnsa_api_key_here
```

> If you use the OpenAI SDK, it sends the key as `Authorization: Bearer <key>` by default. See [Using the OpenAI SDK](#2-openai-python-sdk) for how to send `x-api-key` instead.

Keep your API key secret. Don't commit it to source control — load it from an environment variable instead.

## Endpoint

### `POST /v1/embeddings`

Create an embedding vector for the given input text.

**Request body**

| Field   | Type             | Required | Description                                          |
| ------- | ---------------- | -------- | ---------------------------------------------------- |
| `model` | string           | yes      | Model ID. Use `ngen-embeddings-v1`.                  |
| `input` | string or array  | yes      | Text to embed. Pass an array to embed a batch.       |

**Response body**

```json
{
  "object": "list",
  "model": "ngen-embeddings-v1",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, ...]
    }
  ]
}
```

The embedding vector is at `data[0].embedding`.

## Examples

### 1. Python (`requests`)

```python
import os
import requests

url = "https://api.tnsaai.com/v1/embeddings"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["TNSA_API_KEY"],
}
data = {
    "model": "ngen-embeddings-v1",
    "input": "The quick brown fox jumps over the lazy dog",
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    embeddings = response.json()["data"]
    vector = embeddings[0]["embedding"]
    print(f"Dimension: {len(vector)}")
    print(f"Vector (first 5 values): {vector[:5]}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### 2. OpenAI Python SDK

The API is OpenAI-compatible. Point the client at the TNSA base URL and pass your key via `x-api-key`:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tnsaai.com/v1",
    api_key="not-used",  # placeholder; real auth is the header below
    default_headers={"x-api-key": os.environ["TNSA_API_KEY"]},
)

response = client.embeddings.create(
    model="ngen-embeddings-v1",
    input="The quick brown fox jumps over the lazy dog",
)

vector = response.data[0].embedding
print(f"Dimension: {len(vector)}")
print(f"Vector (first 5 values): {vector[:5]}")
```

> If `api.tnsaai.com` also accepts `Authorization: Bearer <key>`, you can simplify to `OpenAI(base_url=..., api_key=os.environ["TNSA_API_KEY"])`.

### 3. curl

```bash
curl -X POST https://api.tnsaai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "x-api-key: $TNSA_API_KEY" \
  -d '{
    "model": "ngen-embeddings-v1",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

## Batch embeddings

Pass a list of strings to embed multiple inputs in one call. Results come back in the same order, each with its own `index`:

```python
response = client.embeddings.create(
    model="ngen-embeddings-v1",
    input=[
        "The quick brown fox",
        "jumps over the lazy dog",
    ],
)

for item in response.data:
    print(item.index, len(item.embedding))
```

## Errors

The API returns standard HTTP status codes:

| Status | Meaning                                            |
| ------ | -------------------------------------------------- |
| `200`  | Success.                                           |
| `400`  | Malformed request (e.g. missing `input`/`model`).  |
| `401`  | Missing or invalid API key.                        |
| `429`  | Rate limit exceeded — retry with backoff.          |
| `5xx`  | Server error — retry with backoff.                 |

Error responses include a JSON body describing the problem:

```json
{ "error": { "message": "Invalid API key", "type": "authentication_error" } }
```

## Python toolkit: `ngen_rag`

This folder ships a small, importable package that turns text into vectors, saves
them, and runs retrieval-augmented generation on top of `ngen-embeddings-v1`.

```bash
pip install -r requirements.txt   # requests + numpy
export TNSA_API_KEY=tnsa_...       # Windows: setx TNSA_API_KEY tnsa_...
```

**Text → vectors**

```python
from ngen_rag import EmbeddingClient

client = EmbeddingClient()                       # reads TNSA_API_KEY
vec = client.embed("The quick brown fox")        # -> list[float]
mat = client.embed_batch(["a", "b", "c"])        # -> list[list[float]]
```

**Save vectors + search them**

```python
from ngen_rag import VectorStore

store = VectorStore()
store.add(texts=["a", "b"], vectors=mat[:2], metadatas=[{"id": 1}, {"id": 2}])
store.save("vectors.npz")

store = VectorStore.load("vectors.npz")
hits = store.search(client.embed("query"), k=3)   # cosine similarity
for h in hits:
    print(h.score, h.text, h.metadata)
```

**RAG (index → retrieve → prompt)**

```python
from ngen_rag import RAG

rag = RAG()                                       # chunks + embeds for you
rag.add_texts(["... long document ..."], metadatas=[{"source": "handbook"}])
rag.save("kb.npz")

rag = RAG.load("kb.npz")
hits = rag.retrieve("how do refunds work?", k=4)  # top-k chunks
prompt = rag.build_prompt("how do refunds work?") # context + question, ready to send

# embeddings do retrieval; you generate with any LLM:
# openai.chat.completions.create(model="gpt-5",
#     messages=[{"role": "user", "content": prompt}])
```

See [`example.py`](example.py) for a complete runnable walkthrough.

| Class / fn        | What it does                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `EmbeddingClient` | Calls `/v1/embeddings`; `embed()` / `embed_batch()`; retries 429/5xx. |
| `VectorStore`     | Holds vectors + text + metadata; `add` / `search` / `save` / `load`. |
| `RAG`             | Chunk → embed → store → `retrieve` / `build_context` / `build_prompt`. |
| `chunk_text`      | Overlapping character-based chunker.                               |

Errors derive from `EmbeddingsError` (`AuthError`, `BadRequestError`,
`RateLimitError`, `APIError`).

> The folder name is `embeddings`, but the importable package is `ngen_rag`. Add
> this folder to your `PYTHONPATH` (or run from inside it), then `import ngen_rag`.

## Local development

If you're running the service locally, swap the base URL for:

```
http://localhost:8000/v1
```

Everything else (headers, request/response shapes) stays the same.

## License

See [LICENSE](LICENSE).
