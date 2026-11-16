"""
Smoke test for the LiteLLM gateway, using the app's virtual key from .env.
Run after `docker compose up -d` and minting the key, before ingestion:

    python scripts/test_gateway.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import config, gateway  # noqa: E402


def main() -> None:
    print(f"Gateway: {config.LITELLM_BASE_URL}\n")
    try:
        print(f"[1] Embeddings via alias {config.EMBEDDING_MODEL!r} ...")
        vec = gateway.GatewayEmbedder().embed(["How long is an access token valid?"])[0]
        print(f"    OK - {len(vec)} dimensions, first values {[round(v, 4) for v in vec[:3]]}")

        print(f"\n[2] Chat via alias {config.LLM_MODEL!r} ...")
        text, tin, tout = asyncio.run(gateway.achat([{"role": "user", "content": "Reply with exactly: gateway works"}], max_tokens=50))
        print(f"    OK - {text.strip()!r} (tokens in={tin}, out={tout})")
    except Exception as exc:
        sys.exit(f"    ✗ {type(exc).__name__}: {exc}")
    print("\n✅ LiteLLM gateway works for chat and embeddings")


if __name__ == "__main__":
    main()
