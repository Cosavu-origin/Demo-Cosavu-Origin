"""OriginChain → Cosavu ContextAPI → LLM — quickstart.

    pip install -r requirements.txt          # groq + requests (+ Node.js for npx)

    # OriginChain (datastore)
    export OC_HOST=https://<tenant>.<region>.db.originchain.ai
    export OC_TENANT=<tenant>
    export OC_TOKEN=<token>
    # Tool calling: Groq (GenChat's instant model)
    export GROQ_API_KEY=gsk_...
    # Final answer: TNSA NGen-4 Mini (optional x-api-key)
    export TNSA_API_KEY=tnsa_...
    # Cosavu ContextAPI
    export COSAVU_API_KEY=csvu_...

    python example.py "Which 5 customers spent the most last quarter?"
"""

import sys

from origin_chain import OriginChainPipeline


def main() -> None:
    question = " ".join(sys.argv[1:]) or "List the tables available and what they contain."

    with OriginChainPipeline() as pipe:
        # Inspect what the OriginChain MCP server exposes.
        print("OriginChain tools:", [t.get("name") for t in pipe.mcp.list_tools()])

        result = pipe.run(question)

        print("\n=== ANSWER ===")
        print(result.answer)

        print("\n=== ContextAPI optimization ===")
        for step in result.steps:
            print(
                f"  tool={step.tool}  "
                f"{step.original_tokens} -> {step.optimized_tokens} tokens "
                f"(saved {step.tokens_saved})"
            )
        print(f"  total tokens saved before the LLM: {result.tokens_saved}")


if __name__ == "__main__":
    main()
