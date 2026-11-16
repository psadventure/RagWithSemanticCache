# RagWithSemanticCache — project notes

Extends `../CLAUDE.md` (working agreement, engineering defaults). Public repo: `psadventure/RagWithSemanticCache`.

RAG chatbot over 3 fictional Kestrel Markdown docs (`data/`), with a BetterDB semantic cache on Valkey,
a FastAPI backend + Streamlit UI, and a Google ADK agent (`adk web agent`). Every model call goes
through this repo's own Dockerised LiteLLM gateway → Vertex AI.

Ported from `../RAGAdvancedWithSemanticCache` (branch `feature/litellm-gateway`). Patterns
(gateway infra, ADK tools, env drift test) were copied from `../ragtechniques`. Both are read-only sources.

## Commands

```bash
cd infra && docker compose up -d && cd ..          # litellm + db + valkey
python scripts/test_gateway.py                    # gateway smoke test
python -m rag.ingest                              # data/*.md → vector_store/
python -m uvicorn app.backend:app --reload        # API :8000
streamlit run app/frontend.py                     # UI :8501
adk web agent --port 8001                         # ADK dev UI
python -m pytest -q                               # offline tests (no Docker/network)
```

## Decisions

- **Models (via gateway aliases):** chat `gemini-2.5-flash`; embeddings `gemini-embedding-001` (3072 dims). No local embedding model. Vertex takes **1 input per embedding request**, so the embedder loops one text at a time.
- **Docker:** one `infra/docker-compose.yml` adapted from LiteLLM's quickstart (credited), with top-level `name: ragwithsemanticcache` so it never collides with the ragtechniques gateway. Ports are bound to 127.0.0.1 and overridable (`LITELLM_PORT`, `VALKEY_PORT`). The `config/` folder is mounted, not the single file (single-file mounts go stale). ADC is mounted read-only; Windows sets `GCLOUD_ADC_PATH` with forward slashes.
- **Valkey image is `valkey/valkey-bundle:9`, not 8.1.** The search module in 8.1 (1.0.x) rejects BetterDB's `FT.CREATE` schema (`Unknown argument TEXT`).
- **Auth:** the app uses a scoped LiteLLM **virtual key** (models: the two aliases), never the master key. Root `.env` = app settings; `infra/.env` = gateway secrets. Both are gitignored, and each has a `.env.example` kept in sync by `tests/test_env_examples.py`.
- **Light dependencies:** no langchain except `langchain-text-splitters`. Uses the `openai` SDK (sync for ingest, async for serving) against the gateway, and raw `faiss-cpu` (`IndexFlatIP` over L2-normalised vectors, saved as `index.faiss` + `chunks.json`), so there's no pickle and no `allow_dangerous_deserialization`.
- **Chunking:** `MarkdownHeaderTextSplitter` (h1–h3) then `RecursiveCharacterTextSplitter` (1500/150). Metadata: `doc` (file name), `section` (`h2 > h3` path), `title` (h1). The embedded text is prefixed with `title | section`. Sources are shown as `doc | section`.
- **`rag/service.py` is the single API** (`answer`, `search`, `check_cache`, `cache_stats`, `clear_cache`), all async. It raises; FastAPI maps errors to 400/503/502, and agent tools wrap them as `{"error": ...}` (`_safe`). Dependencies (`get_embedder`, `get_store`, `get_cache`, `chat`) are module functions that tests monkeypatch.
- **Event loops:** the async Valkey client and the `AsyncOpenAI` client are bound to the loop that created them, so both are created lazily **per event loop** (uvicorn and `adk web` each have their own).
- **Cache hits return sources** too: `store()` saves `{"sources": [...]}` as metadata, and on a hit we `HMGET` the matched entry's `prompt` + `metadata`, so the UI shows the cached question it matched.
- **`clear_cache`** = BetterDB `flush()` + `initialize()`, because flush leaves the cache uninitialised. In the agent it is a `FunctionTool(require_confirmation=True)`.
- **BetterDB telemetry** (PostHog, on by default in the library) is **disabled** unless `BETTERDB_TELEMETRY=true`.
- **Cost tracking:** BetterDB's default cost table knows `gemini-2.5-flash` ($0.0003 / $0.0025 per 1k tokens), so `CACHE_COST_MODEL` defaults to the alias itself.

## Cache threshold calibration (gemini-embedding-001, 2026-11-15)

BetterDB returns a cosine **distance** (0 = identical); a hit means `distance ≤ SEMANTIC_CACHE_THRESHOLD`.
The UI and agent also show `similarity = 1 − distance`. The upstream MiniLM value of 0.15 was re-measured:

| Pairs | Distance |
|---|---|
| Paraphrases (e.g. "How long is an access token valid?" ~ "What's the access token lifetime?" 0.1225; "…license cost?" ~ "price of …license?" 0.0047) | 0.005 – 0.128 |
| Near-misses with different answers: cancel *pending* vs *processing* | 0.137 |
| access vs refresh token | 0.154 |
| Narrower rewording of a two-part question (webhook verify + replay vs "validate webhook signatures") | 0.178 |
| Other related / off-topic | 0.29 – 0.56 |
| "variance in mock conversion 1" vs "…3" (different answers!) | 0.030 |

**Chosen default: 0.13.** It catches every measured paraphrase and misses the 0.137 and 0.154 near-misses.
0.15 would have served a wrong answer for pending vs processing. The mock 1 vs 3 false hit can't be fixed
with a threshold; it's documented in the README as the cache's failure mode.

## Environment gotchas

- In Claude Code's sandbox, `pip`/`uv` installs into `.venv` failed with "Operation not permitted"
  until `TMPDIR` pointed at a writable directory (the session scratchpad). uv still failed; pip worked.
- The first `docker compose up` takes about 1–2 min while LiteLLM runs its ~225 Prisma migrations.
- `adk web` defaults to port 8000, the same as uvicorn. Use `--port 8001` to run both.
