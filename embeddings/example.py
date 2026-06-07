"""TNSA ngen-embeddings — vectors + RAG quickstart.

    pip install -r requirements.txt
    export TNSA_API_KEY=tnsa_...        # Windows: setx TNSA_API_KEY tnsa_...
    python example.py
"""

from ngen_rag import EmbeddingClient, RAG

# ── 1. text -> vectors ────────────────────────────────────────────────────────
client = EmbeddingClient()  # reads TNSA_API_KEY from the environment
vec = client.embed("The quick brown fox jumps over the lazy dog")
print(f"single vector dim: {len(vec)}")

batch = client.embed_batch(["first sentence", "second sentence"])
print(f"batch: {len(batch)} vectors")

# ── 2. build a RAG index and save it ──────────────────────────────────────────
DOCS = [
    "Cosavu ContextAPI optimizes bloated prompts into lean ones before they reach "
    "a large language model, cutting token cost and latency.",
    "TNSA's ngen-embeddings-v1 turns text into dense vectors. The API is "
    "OpenAI-compatible and authenticated with an x-api-key header.",
    "Refunds are processed within 5 business days to the original payment method. "
    "Contact support@example.com to start a refund request.",
]
META = [{"source": "contextapi"}, {"source": "embeddings"}, {"source": "policy"}]

rag = RAG()
n = rag.add_texts(DOCS, metadatas=META)
print(f"indexed {n} chunks")
rag.save("kb.npz")

# ── 3. reload and query ───────────────────────────────────────────────────────
rag = RAG.load("kb.npz")
question = "How long do refunds take?"

print("\nTop matches:")
for hit in rag.retrieve(question, k=2):
    print(f"  [{hit.score:.3f}] ({hit.metadata.get('source')}) {hit.text[:80]}...")

# ── 4. assemble a RAG prompt -> send to any LLM ───────────────────────────────
prompt = rag.build_prompt(question, k=2)
print("\nRAG prompt to forward to your LLM:\n")
print(prompt)
# e.g. openai.chat.completions.create(model="gpt-5",
#          messages=[{"role": "user", "content": prompt}])
