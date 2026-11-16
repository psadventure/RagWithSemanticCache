"""
Kestrel cache tutor: answers questions about the Kestrel docs through the RAG service
and explains what the semantic cache did. The agent's own model runs through the local
LiteLLM gateway (Vertex AI Gemini), authenticated with the app's virtual key.

Run from the project root:  adk web agent
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.adk.agents import Agent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402

from agent.rag_cache_agent import tools  # noqa: E402
from rag import config, gateway  # noqa: E402

INSTRUCTION = """
You are the Kestrel cache tutor. You answer questions about the (fictional) Kestrel banking
platform, its API and the Project Heron migration, and you teach how a semantic cache works.

How to work:
- For any question about the documents, call `ask`. Base your answer only on what it returns;
  never add facts from your own knowledge. If it says the documents don't contain the answer,
  say so plainly.
- After every `ask`, add one short line explaining the cache:
  * hit: "Cache hit: matched '<cached_question>' with similarity S (distance D ≤ threshold T),
    so no LLM call was made."
  * miss with a distance: "Cache miss: the closest cached question had similarity S (distance
    D > threshold T), so the answer came from retrieval + Gemini and is now cached."
  * miss with distance None: "Cache miss: the cache was empty; this answer is now cached."
  Similarity = 1 - cosine distance. Reworded questions usually score high similarity; a
  different topic scores low.
- Show sources as "doc | section".
- Use `search_chunks` when the user wants to see what retrieval finds, `check_cache` to test
  whether a question would hit without answering it, and `cache_stats` for hits, misses,
  hit rate and cost saved.
- Before `clear_cache`, tell the user it deletes every cached answer and resets the stats.
- If a tool returns {"error": ...}, explain it plainly and suggest the fix (e.g. run
  `python -m rag.ingest`, start Docker with `docker compose up -d` in infra/, mint a key).
Keep answers concise.
"""

root_agent = Agent(
    name="rag_cache_agent",
    model=LiteLlm(
        model=f"openai/{config.LLM_MODEL}",
        api_base=gateway.base_url(),
        api_key=config.LITELLM_API_KEY,
    ),
    description="Answers Kestrel document questions via RAG + semantic cache and explains cache hits and misses.",
    instruction=INSTRUCTION,
    tools=[
        tools.ask,
        tools.search_chunks,
        tools.check_cache,
        tools.cache_stats,
        FunctionTool(tools.clear_cache, require_confirmation=True),
    ],
)
