"""
OpenAI-SDK clients for the LiteLLM gateway (sync for ingestion, async for serving).

The gateway speaks the OpenAI API, so the plain `openai` SDK is all the app needs.
"""
import asyncio

from openai import AsyncOpenAI, OpenAI

from rag import config


def base_url() -> str:
    # The LiteLLM proxy serves the OpenAI routes at both "/" and "/v1".
    url = config.LITELLM_BASE_URL
    return url if url.endswith("/v1") else f"{url}/v1"


def _check_key() -> None:
    if not config.LITELLM_API_KEY:
        raise RuntimeError(
            "LITELLM_API_KEY is not set. Mint a virtual key and put it in .env "
            "(README.md → 'Mint the virtual key')."
        )


_sync_client: OpenAI | None = None
_async_client: tuple[asyncio.AbstractEventLoop, AsyncOpenAI] | None = None


def sync_client() -> OpenAI:
    global _sync_client
    _check_key()
    if _sync_client is None:
        _sync_client = OpenAI(base_url=base_url(), api_key=config.LITELLM_API_KEY, max_retries=5)
    return _sync_client


def async_client() -> AsyncOpenAI:
    """One async client per event loop (its connection pool is bound to the loop)."""
    global _async_client
    _check_key()
    loop = asyncio.get_running_loop()
    if _async_client is None or _async_client[0] is not loop:
        _async_client = (loop, AsyncOpenAI(base_url=base_url(), api_key=config.LITELLM_API_KEY, max_retries=5))
    return _async_client[1]


class GatewayEmbedder:
    """Embeds through the gateway alias EMBEDDING_MODEL, one input per request
    (Vertex gemini-embedding-001 accepts a single input per call)."""

    @property
    def model(self) -> str:
        return config.EMBEDDING_MODEL

    def embed(self, texts: list[str], progress=None) -> list[list[float]]:
        vectors = []
        for i, text in enumerate(texts, 1):
            resp = sync_client().embeddings.create(model=self.model, input=[text])
            vectors.append(resp.data[0].embedding)
            if progress:
                progress(i, len(texts))
        return vectors

    async def aembed(self, text: str) -> list[float]:
        resp = await async_client().embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding


async def achat(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.2) -> tuple[str, int, int]:
    """Chat completion through the gateway alias LLM_MODEL. Returns (text, input_tokens, output_tokens)."""
    resp = await async_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
