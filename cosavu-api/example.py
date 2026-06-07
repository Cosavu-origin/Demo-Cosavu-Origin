"""Cosavu ContextAPI — quickstart example.

    pip install -r requirements.txt
    export COSAVU_API_KEY=csvu_...     # Windows: setx COSAVU_API_KEY csvu_...
    python example.py
"""

from cosavu_context import Cosavu, MODELS

BLOATED = (
    "Hey so um I was just wondering if you could maybe possibly help me out here, "
    "like, I'm building a production-grade retrieval-augmented generation system and "
    "I really really need you to walk me through, step by step in a very thorough way, "
    "the architectural decisions around chunking, embedding model selection, and the "
    "tradeoffs between latency, recall, and cost — please assume I know the basics, "
    "thank you so so much!"
)


def main() -> None:
    cosavu = Cosavu()  # reads COSAVU_API_KEY from the environment

    for model in MODELS:
        r = cosavu.optimize(BLOATED, model=model)
        print("=" * 78)
        print(f"MODEL: {r.model}")
        print(f"OPTIMIZED:\n  {r.text}")
        print(
            f"TOKENS: {r.original_tokens} -> {r.optimized_tokens} "
            f"(saved {r.tokens_saved}, {r.reduction:.0%})  |  {r.latency_ms:.0f} ms"
        )

    # How you'd actually use it: optimize, then forward the lean prompt to your LLM.
    print("=" * 78)
    lean = cosavu.optimize(BLOATED).text
    print("Forward this lean prompt to your model of choice:")
    print(f"  {lean}")


if __name__ == "__main__":
    main()
