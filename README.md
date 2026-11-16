# RagWithSemanticCache

A RAG chatbot over a small set of Markdown docs, with a **semantic cache** in front of the LLM:

- **Retrieval:** Markdown-aware chunks embedded with Vertex AI `gemini-embedding-001` (3072 dims) and stored in FAISS.
- **Answers:** `gemini-2.5-flash`, grounded in the retrieved chunks and citing them as `doc | section`.
- **Semantic cache:** [BetterDB semantic cache](https://pypi.org/project/betterdb-semantic-cache/) on Valkey. A question that means the same as one already answered comes back from the cache in milliseconds, with no LLM call.
- **Three ways to use it:** a FastAPI backend with a Streamlit chat UI, and a Google ADK agent (`adk web`) that explains every cache hit and miss.
- **Gateway:** every model call goes through this repo's own Dockerised **LiteLLM gateway**, using a scoped virtual key. The app never sees Google credentials or the gateway's master key.

```
question ─▶ semantic cache (Valkey) ── hit ──▶ cached answer (+ similarity)
                    │ miss
                    ▼
            embed ─▶ FAISS top-k ─▶ gemini-2.5-flash ─▶ answer ─▶ stored in cache
            (all model calls: app ─▶ LiteLLM gateway ─▶ Vertex AI)
```

The data is three **fictional** documents about "Kestrel", an invented core-banking platform, and "Project Heron", a migration to it (`data/`). They contain a few planted traps: a stale date, a question the docs can't answer, and two questions that are near-identical in wording but have different answers. See [Testing](#testing).

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Google Cloud credentials](#2-google-cloud-credentials)
3. [Gateway settings: `infra/.env`](#3-gateway-settings-infraenv)
4. [Start the stack](#4-start-the-stack)
5. [Mint the app's virtual key](#5-mint-the-apps-virtual-key)
6. [App settings: `.env`](#6-app-settings-env)
7. [Python environment](#7-python-environment)
8. [Ingest the documents](#8-ingest-the-documents)
9. [Run: backend, Streamlit, ADK agent](#9-run-backend-streamlit-adk-agent)
10. [Stop, start, logs](#10-stop-start-logs)
11. [Troubleshooting](#11-troubleshooting)
- [Testing](#testing) · [How the cache decides](#how-the-cache-decides) · [Project layout](#project-layout) · [Configuration](#configuration) · [Tests](#offline-tests) · [License](#license)

Each step gives **macOS** (zsh/bash; Linux is the same) and **Windows PowerShell** commands. Run everything from the repo root unless a step says otherwise.

---

## 1. Prerequisites

| Tool | Notes |
|---|---|
| **Python 3.13** | macOS: `brew install python@3.13`. Windows: the python.org installer (includes the `py` launcher). |
| **Docker Desktop** | Must be running for the gateway and Valkey. |
| **Google Cloud CLI** (`gcloud`) | https://cloud.google.com/sdk/docs/install |
| **A GCP project with Vertex AI enabled** | Billing on. `gemini-2.5-flash` and `gemini-embedding-001` must be available in your region (default `us-east1`). |

Enable the Vertex AI API once (both shells):

```bash
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
```

## 2. Google Cloud credentials

The gateway authenticates to Vertex AI with your **Application Default Credentials** (ADC), mounted read-only into its container.

**macOS**
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
ls ~/.config/gcloud/application_default_credentials.json     # should exist
```

**Windows PowerShell**
```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
Test-Path "$env:APPDATA\gcloud\application_default_credentials.json"   # should be True
```

## 3. Gateway settings: `infra/.env`

`infra/.env` holds the **gateway's** secrets. It is gitignored and is read only by `docker compose`. You need two random keys: the master key (admin only, never given to the app) and the salt key (which encrypts credentials stored in Postgres).

**macOS**
```bash
cp infra/.env.example infra/.env
echo "sk-$(openssl rand -hex 32)"     # run twice: one for LITELLM_MASTER_KEY, one for LITELLM_SALT_KEY
open -e infra/.env                    # or any editor
```

**Windows PowerShell**
```powershell
Copy-Item infra\.env.example infra\.env
# run twice: one for LITELLM_MASTER_KEY, one for LITELLM_SALT_KEY
$b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); "sk-" + -join ($b | ForEach-Object { $_.ToString("x2") })
notepad infra\.env
```

Fill in:

```ini
LITELLM_MASTER_KEY=sk-...            # first generated key
LITELLM_SALT_KEY=sk-...              # second generated key; keep it once the DB exists
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-east1
```

**Windows only:** also set where your ADC file is, with **forward slashes**:

```ini
GCLOUD_ADC_PATH=C:/Users/<you>/AppData/Roaming/gcloud/application_default_credentials.json
```

Optional: `LITELLM_PORT` (default 4000) and `VALKEY_PORT` (default 6379) if those ports are taken. Both are bound to `127.0.0.1` only.

## 4. Start the stack

One compose file, `infra/docker-compose.yml`, runs three containers under its own compose project name, `ragwithsemanticcache`, so it never collides with another LiteLLM stack on the same machine:

| Service | Image | Purpose |
|---|---|---|
| `litellm` | `docker.litellm.ai/berriai/litellm` | OpenAI-compatible gateway → Vertex AI (port 4000) |
| `db` | `postgres:16` | Gateway's store for virtual keys and spend logs (internal only) |
| `valkey` | `valkey/valkey-bundle:9` | Valkey with the **search** module (vector index) for the cache (port 6379) |

**macOS**
```bash
cd infra
docker compose up -d
docker compose ps                                    # all three "running"
curl -s http://localhost:4000/health/liveliness      # → "I'm alive!"
cd ..
```

**Windows PowerShell**
```powershell
cd infra
docker compose up -d
docker compose ps
Invoke-RestMethod http://localhost:4000/health/liveliness
cd ..
```

The **first** start takes a minute or two because the gateway runs its database migrations. Retry the health check until it answers. Admin UI: http://localhost:4000/ui (log in with `LITELLM_MASTER_KEY`).

## 5. Mint the app's virtual key

The app never uses the master key. Create a **virtual key** that can only call the two configured models.

**macOS**
```bash
MK=$(sed -n 's/^LITELLM_MASTER_KEY=//p' infra/.env)
curl -s http://localhost:4000/key/generate \
  -H "Authorization: Bearer $MK" -H 'Content-Type: application/json' \
  -d '{"key_alias":"ragwithsemanticcache-app","models":["gemini-2.5-flash","gemini-embedding-001"]}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["key"])'
```

**Windows PowerShell**
```powershell
$mk = (Select-String -Path infra\.env -Pattern '^LITELLM_MASTER_KEY=(.*)').Matches[0].Groups[1].Value
$body = '{"key_alias":"ragwithsemanticcache-app","models":["gemini-2.5-flash","gemini-embedding-001"]}'
(Invoke-RestMethod -Method Post -Uri http://localhost:4000/key/generate -Headers @{ Authorization = "Bearer $mk" } -ContentType 'application/json' -Body $body).key
```

Copy the printed `sk-...` value for the next step. Keys can be listed, edited or revoked in the admin UI.

## 6. App settings: `.env`

The root `.env` holds the **app's** settings. It is gitignored and is separate from `infra/.env`.

**macOS**
```bash
cp .env.example .env
open -e .env
```

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
notepad .env
```

Set `LITELLM_API_KEY=` to the virtual key from step 5. Everything else has working defaults; see [Configuration](#configuration).

## 7. Python environment

**macOS**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_gateway.py          # embeddings (3072 dims) + chat through the gateway
```

**Windows PowerShell**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
python scripts\test_gateway.py
```

The later steps assume the venv is activated in each new terminal (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`).

## 8. Ingest the documents

```bash
python -m rag.ingest
```
(The command is the same on both.) It splits each `data/*.md` on Markdown headers, caps each section at `CHUNK_SIZE` characters, embeds every chunk through the gateway, and writes `vector_store/index.faiss` and `vector_store/chunks.json`. That's plain FAISS plus JSON, so there's no pickle file. Expect about 44 chunks in roughly 15 seconds: Vertex takes **one input per embedding request** for `gemini-embedding-001`, so chunks are sent one at a time.

Re-run it whenever you change the docs, `CHUNK_SIZE`, or the embedding model.

## 9. Run: backend, Streamlit, ADK agent

Each in its own terminal (venv activated):

```bash
python -m uvicorn app.backend:app --reload      # API on http://localhost:8000 (docs: /docs)
streamlit run app/frontend.py                   # chat UI on http://localhost:8501 (needs the backend)
adk web agent                                   # ADK dev UI on http://localhost:8000, pick "rag_cache_agent"
```

`adk web` also defaults to port 8000. To run it alongside the backend, use `adk web agent --port 8001`.

**Backend endpoints**

| Method | Path | What it does |
|---|---|---|
| POST | `/chat` `{"question": "..."}` | Cache check → retrieve → Gemini → cache store. Returns `answer`, `sources`, `cache_hit`, `cached_question`, `distance`, `similarity`, `threshold` |
| POST | `/search` `{"question": "...", "top_k": 4}` | Retrieval only: the top chunks with cosine scores. No LLM call and no cache |
| POST | `/cache/check` `{"question": "..."}` | Would this question hit? Shows the closest cached question |
| GET | `/stats` | Hits, misses, hit rate, cost saved, entries |
| DELETE | `/cache` | Drop every cached answer and reset the stats |
| GET | `/health` | Valkey + gateway reachability |

**ADK agent tools:** `ask`, `search_chunks`, `check_cache`, `cache_stats`, and `clear_cache`. `clear_cache` asks you to confirm in the UI before it runs.

## 10. Stop, start, logs

Run from `infra/` (same on both shells):

```bash
docker compose ps                    # status
docker compose logs -f litellm       # follow gateway logs (Ctrl+C to exit)
docker compose restart litellm       # apply infra/config/config.yaml changes
docker compose stop                  # pause; containers kept
docker compose up -d                 # start again
docker compose down                  # remove containers; keys, spend logs and cache kept (volumes)
docker compose down -v               # also DELETE the volumes: virtual keys, spend logs, cached answers
```

- After `down -v`, mint a new virtual key (step 5) and update `.env`.
- Quitting Docker Desktop stops everything.
- Stop uvicorn, Streamlit and `adk web` with Ctrl+C.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` on `localhost:4000` or `6379` | Start Docker Desktop, then `docker compose up -d` in `infra/`. On first start, wait for the migrations to finish (`docker compose logs -f litellm`). |
| `set it in infra/.env` from compose | `infra/.env` is missing or a required key is empty (step 3). |
| `LITELLM_API_KEY is not set` | Mint a key (step 5) and put it in the root `.env`. |
| 401/403 from the gateway | The virtual key is wrong or revoked (e.g. after `down -v`). Mint a new one. Don't use the master key. |
| Port 4000 or 6379 already in use | Another gateway or Redis/Valkey is running (e.g. the `ragtechniques` gateway). Stop it, or set `LITELLM_PORT` / `VALKEY_PORT` in `infra/.env` **and** `LITELLM_BASE_URL` / `VALKEY_PORT` in `.env`. |
| `No vector store at 'vector_store'` (HTTP 503) | Run `python -m rag.ingest`. |
| `Query embedding has N dimensions but the index has M` | You changed `EMBEDDING_MODEL`. Re-run ingestion. |
| `FT.CREATE failed: Invalid field type … TEXT` | Your Valkey's search module is too old (e.g. `valkey-bundle:8.1`). Use `valkey/valkey-bundle:9` as in the compose file. |
| `Index dimension mismatch` from the cache | The cache index was created with another embedding model. `DELETE /cache` (or the agent's `clear_cache`), or set a new `SEMANTIC_CACHE_NAME`. |
| Vertex `404 … model was not found` | The model isn't available to your project or region. Try another `VERTEX_LOCATION`. |
| `503 … authentication database is temporarily unreachable` while Postgres is healthy | Usually the gateway can't read its config or ADC file, e.g. after re-running `gcloud auth application-default login`. Recreate the container: `docker compose up -d --force-recreate --no-deps litellm` (keys and DB are kept). |
| Windows: `GOOGLE_APPLICATION_CREDENTIALS` file not found in logs | Set `GCLOUD_ADC_PATH` in `infra/.env` with forward slashes (step 3), then recreate `litellm` as above. |
| `No module named 'mcp'` when running the agent | `python -m pip install -r requirements.txt`. litellm imports `mcp` when tools are passed. |
| Every rewording misses, or wrong answers come back as hits | Tune `SEMANTIC_CACHE_THRESHOLD`; see [How the cache decides](#how-the-cache-decides). |

---

## Testing

Start with an empty cache (`DELETE /cache`, the Streamlit sidebar button, or the agent's `clear_cache`), then ask these questions **in order**. The distances are what `gemini-embedding-001` produced when this repo was built; yours should match to about ±0.01.

**macOS**
```bash
curl -s -X DELETE localhost:8000/cache
ask() { curl -s localhost:8000/chat -H 'Content-Type: application/json' -d "{\"question\": \"$1\"}" | python3 -m json.tool; }
ask "How long is an access token valid?"
```

**Windows PowerShell**
```powershell
Invoke-RestMethod -Method Delete http://localhost:8000/cache
function ask($q) { Invoke-RestMethod -Method Post http://localhost:8000/chat -ContentType 'application/json' -Body (@{ question = $q } | ConvertTo-Json) }
ask "How long is an access token valid?"
```

| # | Question | Expected | Why |
|---|---|---|---|
| 1 | How long is an access token valid? | **miss** (cache empty) → "3600 seconds (one hour)", source `kestrel_api_reference.md \| 2. Authentication` | First time: retrieval + Gemini, then stored |
| 2 | How long is an access token valid? | **hit**, distance 0.0, similarity 1.0, answered in a few ms | Exact repeat |
| 3 | What's the access token lifetime? | **hit**, distance ≈ 0.12, `cached_question` = #1 | Reworded, same meaning |
| 4 | How long is a refresh token valid? | **miss**, distance ≈ 0.15 → "refresh tokens are not issued for the client-credentials flow" | Close wording, different question: just above the threshold |
| 5 | Why did the Fraud Shield pilot start late? | **miss**, distance ≈ 0.46 | Different topic |
| 6 | When does Harbor & Pine go live on Kestrel? | **miss** → **17–18 October 2026**, noting the conflicting September date | **Stale fact:** the overview says "September 2026", but the timeline's decision D-11 moved go-live to 17–18 Oct 2026. The prompt says to prefer the latest decision and flag the conflict |
| 7 | How much does a Kestrel Cloud license cost? | **miss** → a refusal ("can't find it in the documents") | **Unanswerable:** the docs contain no pricing. The model must not make up a number |
| 8 | What is the price of a Kestrel Cloud license? | **hit** (distance ≈ 0.005) → the same refusal | Refusals are cached too |
| 9 | How do I verify the webhook signature and prevent replay attacks? | **miss** → parse `Kestrel-Signature` (`t=`, `v1=`), reject if more than 300 s old, HMAC-SHA256 of `"<t>.<raw body>"` with the signing secret, constant-time compare | Multi-step answer from `5. Webhooks` |
| 10 | How can I validate webhook signatures? | **miss**, distance ≈ 0.18 → a fresh answer | Looks like a rewording of #9, but #9 also asked about replay attacks. A shorter question with less scope is a *different* question, so missing here is correct |
| 11 | What was the variance in mock conversion 1? | **miss** → $41,207.13 | |
| 12 | What was the variance in mock conversion 3? | ⚠️ **hit** (distance ≈ 0.03) → the *wrong* answer, $41,207.13 (correct: $0.00) | **Semantic caching's failure mode:** questions that differ in one number or entity embed almost identically. No threshold can separate them. In production you would add the entity to the cache key, set a category threshold, or put an LLM judge on borderline hits |

Then try `/search` on the same questions to see which chunks were retrieved, and `/cache/check` to see the closest cached question without answering.

### Reading `/stats`

After the 12 questions above (cost varies with answer length):

```json
{"hits": 4, "misses": 8, "total": 12, "hit_rate": "33.3%", "cost_saved_usd": "$0.002581",
 "entries": 8, "cache_name": "ragcache", "threshold": 0.13, "cost_model": "gemini-2.5-flash"}
```

- `hits` / `misses` / `total` / `hit_rate`: every cache lookup, including `/cache/check` and the agent's `check_cache`. A lookup on an empty cache is a miss.
- `entries`: answers stored. One per miss from `/chat` or `ask`.
- `cost_saved_usd`: the LLM cost that hits avoided. Each entry stores the token counts of the call that produced it, and BetterDB prices them with `cost_model` from its built-in cost table, which knows `gemini-2.5-flash` ($0.30 / $2.50 per million input / output tokens). It's small here because the answers are short. Embedding calls aren't counted.
- `threshold`: the maximum cosine distance that counts as a hit.

### The same flow through the ADK agent

Run `adk web agent`, open the UI, and pick **rag_cache_agent**. Then say:

1. "Clear the cache." The agent warns you, then the UI asks you to **confirm** the `clear_cache` tool call.
2. "How long is an access token valid?" → answer plus *"Cache miss: the cache was empty; this answer is now cached."*
3. "What's the access token lifetime?" → *"Cache hit: matched 'How long is an access token valid?' with similarity 0.8775 (distance 0.1225 ≤ threshold 0.13), so no LLM call was made."*
4. "How much does a Kestrel Cloud license cost?" → refusal plus a miss explanation.
5. "Would 'When do access tokens expire?' be a cache hit?" → the agent uses `check_cache`.
6. "Show me the chunks for: webhook replay protection" → `search_chunks`.
7. "Show me the cache stats." → `cache_stats`.

The agent's own reasoning runs on `gemini-2.5-flash` through the same gateway and virtual key (`LiteLlm("openai/gemini-2.5-flash")`).

---

## How the cache decides

BetterDB embeds the incoming question with the same `gemini-embedding-001` alias, runs a KNN search over cached questions in Valkey, and gets back a **cosine distance** (0 = identical, 1 = unrelated). The question is a hit when `distance ≤ SEMANTIC_CACHE_THRESHOLD`. The app also shows `similarity = 1 − distance`.

The default threshold, **0.13**, was calibrated for `gemini-embedding-001` on this corpus:

| Pair type | Measured distances |
|---|---|
| Paraphrases (should hit) | 0.005 – 0.128 |
| Related but different answers (should miss) | 0.137 (cancel *pending* vs *processing*), 0.154 (access vs refresh token), 0.178 (narrower rewording of a two-part question), 0.29 – 0.47 |
| Off-topic | 0.43 – 0.56 |
| Same question, different number (mock 1 vs 3) | 0.03 (impossible to separate) |

The upstream project used 0.15 with MiniLM embeddings. With Gemini embeddings, 0.15 would serve the "pending" answer for the "processing" question. Lower the threshold for safety (fewer, surer hits); raise it for more hits and more risk.

## Project layout

```
infra/
  docker-compose.yml       litellm + db (Postgres) + valkey, project name "ragwithsemanticcache"
  config/config.yaml       gateway aliases: gemini-2.5-flash, gemini-embedding-001
  .env.example             gateway secrets template → infra/.env
rag/
  config.py                app settings from .env
  gateway.py               openai SDK clients (sync + async) for the gateway
  ingest.py                Markdown header split → size cap → embed → FAISS   (python -m rag.ingest)
  store.py                 FAISS IndexFlatIP over normalised vectors + chunks.json
  cache.py                 BetterDB cache wrapper, Valkey client (+ optional ElastiCache IAM auth)
  service.py               the single API: answer, search, check_cache, cache_stats, clear_cache
app/
  backend.py               FastAPI over rag.service
  frontend.py              Streamlit chat UI
agent/rag_cache_agent/     ADK agent + tools (adk web agent)
scripts/test_gateway.py    gateway smoke test
data/                      3 fictional Kestrel docs
tests/                     offline pytest suite
```

## Configuration

All app settings are optional except `LITELLM_API_KEY`. `.env.example` lists every one with its default.

| Setting | Default | Meaning |
|---|---|---|
| `LITELLM_BASE_URL` | `http://localhost:4000` | Gateway URL |
| `LITELLM_API_KEY` | — | Virtual key (step 5) |
| `LLM_MODEL` / `EMBEDDING_MODEL` | `gemini-2.5-flash` / `gemini-embedding-001` | Gateway aliases |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | 1500 / 150 / 4 | Chunking and retrieval |
| `SEMANTIC_CACHE_THRESHOLD` | 0.13 | Max cosine distance for a hit |
| `SEMANTIC_CACHE_NAME` | `ragcache` | Valkey key prefix / index name |
| `CACHE_COST_MODEL` | `gemini-2.5-flash` | Model name used to price saved calls |
| `BETTERDB_TELEMETRY` | `false` | BetterDB sends anonymous usage analytics (PostHog) unless disabled; this app disables it by default |
| `VALKEY_USE_IAM_AUTH` (+ `VALKEY_IAM_USER_ID`, `VALKEY_SERVERLESS_CACHE_NAME`, `AWS_REGION`) | `false` | Use AWS ElastiCache Serverless with IAM auth instead of local Valkey: TLS plus a short-lived SigV4 token, refreshed automatically |

## Offline tests

```bash
python -m pytest -q
```

The tests run with no Docker and no network. They use a hashing fake embedder and an in-memory fake cache. They cover:
- ingestion of all 3 docs with `doc`/`section` metadata;
- golden questions retrieving the expected doc;
- the cache hit/miss flow in the service;
- FastAPI routes and error codes;
- agent tools returning `{"error": ...}` instead of raising;
- a drift test that keeps both `.env.example` files in sync with the code and compose file.

## Credits

- `infra/docker-compose.yml` is adapted from LiteLLM's [quickstart compose file](https://github.com/BerriAI/litellm/blob/main/docker/docker-compose.quickstart.yml) (BerriAI).
- Semantic cache: [betterdb-semantic-cache](https://pypi.org/project/betterdb-semantic-cache/).

## License

MIT, see [LICENSE](LICENSE). The documents in `data/` are fictional.
