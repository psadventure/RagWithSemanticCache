"""
BetterDB semantic cache on Valkey, behind a small interface the service uses
(tests swap in an in-memory fake with the same methods).

Optional: AWS ElastiCache Serverless with IAM auth (VALKEY_USE_IAM_AUTH=true).
"""
import json
from dataclasses import dataclass

import valkey.asyncio as valkey_async
from betterdb_semantic_cache import SemanticCache, SemanticCacheOptions
from betterdb_semantic_cache.types import AnalyticsOptions, CacheStoreOptions
from valkey.credentials import CredentialProvider

from rag import config


# ============================================================================
# IAM auth token (ElastiCache Serverless public endpoints)
# ============================================================================


def generate_iam_auth_token(user_id: str, cache_name: str, region: str) -> str:
    """SigV4-signed, short-lived (15 min) auth token for an ElastiCache endpoint,
    following AWS's documented pattern for IAM-authenticated connections."""
    import botocore.session
    from botocore.model import ServiceId
    from botocore.signers import RequestSigner

    session = botocore.session.get_session()
    signer = RequestSigner(
        ServiceId("elasticache"),
        region,
        "elasticache",
        "v4",
        session.get_credentials(),
        session.get_component("event_emitter"),
    )
    url = f"https://{cache_name}/?Action=connect&User={user_id}"
    presigned_url = signer.generate_presigned_url(
        {"method": "GET", "url": url, "body": {}, "headers": {}, "context": {}},
        region_name=region,
        expires_in=900,
        operation_name="",
    )
    # The client sends this (without the scheme) as the password; the server derives
    # everything it needs to verify the signature from it.
    return presigned_url.removeprefix("https://")


class ElastiCacheIAMCredentialProvider(CredentialProvider):
    """Fresh IAM token on every get_credentials() call; the Valkey client calls it
    whenever it (re)authenticates, including after the ~15 min token expiry."""

    def __init__(self, user_id: str, cache_name: str, region: str):
        self._user_id = user_id
        self._cache_name = cache_name
        self._region = region

    def get_credentials(self):
        return self._user_id, generate_iam_auth_token(self._user_id, self._cache_name, self._region)


def make_valkey_client() -> valkey_async.Valkey:
    if config.VALKEY_USE_IAM_AUTH:
        if not config.VALKEY_SERVERLESS_CACHE_NAME:
            raise RuntimeError("VALKEY_USE_IAM_AUTH=true needs VALKEY_SERVERLESS_CACHE_NAME.")
        return valkey_async.Valkey(
            host=config.VALKEY_HOST,
            port=config.VALKEY_PORT,
            ssl=True,
            credential_provider=ElastiCacheIAMCredentialProvider(
                user_id=config.VALKEY_IAM_USER_ID,
                cache_name=config.VALKEY_SERVERLESS_CACHE_NAME,
                region=config.AWS_REGION,
            ),
        )
    return valkey_async.Valkey(host=config.VALKEY_HOST, port=config.VALKEY_PORT)


# ============================================================================
# Cache backend
# ============================================================================


@dataclass
class CheckResult:
    hit: bool
    distance: float | None          # cosine distance to the closest cached question (None = cache empty)
    response: str | None = None
    cached_question: str | None = None
    metadata: dict | None = None


class BetterDBCache:
    """Thin wrapper around betterdb_semantic_cache.SemanticCache."""

    def __init__(self, embed_fn, client: valkey_async.Valkey | None = None):
        self.client = client or make_valkey_client()
        self.name = config.SEMANTIC_CACHE_NAME
        self.threshold = config.SEMANTIC_CACHE_THRESHOLD
        self._cache = SemanticCache(
            SemanticCacheOptions(
                name=self.name,
                client=self.client,
                embed_fn=embed_fn,
                default_threshold=self.threshold,
                use_default_cost_table=True,
                analytics=AnalyticsOptions(disabled=not config.BETTERDB_TELEMETRY),
            )
        )

    async def initialize(self) -> None:
        await self._cache.initialize()

    async def check(self, question: str) -> CheckResult:
        r = await self._cache.check(question)
        if not r.hit:
            return CheckResult(hit=False, distance=r.similarity)
        cached_question, metadata = None, {}
        if r.matched_key:
            prompt, meta = await self.client.hmget(r.matched_key, ["prompt", "metadata"])
            cached_question = prompt.decode() if isinstance(prompt, bytes) else prompt
            try:
                metadata = json.loads(meta) if meta else {}
            except ValueError:
                metadata = {}
        return CheckResult(True, r.similarity, r.response, cached_question, metadata)

    async def store(self, question: str, answer: str, *, model: str, input_tokens: int,
                    output_tokens: int, metadata: dict) -> None:
        await self._cache.store(
            question,
            answer,
            CacheStoreOptions(model=model, input_tokens=input_tokens,
                              output_tokens=output_tokens, metadata=metadata),
        )

    async def stats(self) -> dict:
        s = await self._cache.stats()
        try:
            entries = (await self._cache.index_info()).num_docs
        except Exception:
            entries = None
        return {
            "hits": s.hits,
            "misses": s.misses,
            "total": s.total,
            "hit_rate": s.hit_rate,
            "cost_saved_usd": s.cost_saved_micros / 1_000_000,
            "entries": entries,
        }

    async def clear(self) -> None:
        # flush() drops the index, entries and stats, and leaves the cache uninitialised.
        await self._cache.flush()
        await self._cache.initialize()

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        await self._cache.shutdown()
        await self.client.aclose()
