"""
Streamlit chat UI for the FastAPI backend.

Run from the project root (with the backend running):
    streamlit run app/frontend.py
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Kestrel RAG + Semantic Cache", page_icon="⚡", layout="centered")


def _api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
    except requests.RequestException as exc:
        return None, f"Backend not reachable at {BACKEND_URL}: {exc}"
    if r.ok:
        return r.json(), None
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return None, f"{r.status_code}: {detail}"


def _badge(m: dict) -> str:
    if m.get("cache_hit"):
        s = f"⚡ Cache hit · similarity {m['similarity']:.4f} (distance {m['distance']:.4f} ≤ {m['threshold']})"
        if m.get("cached_question"):
            s += f" · matched “{m['cached_question']}”"
        return s
    if m.get("distance") is None:
        return "🧠 Cache miss · cache was empty → answered by Gemini and stored"
    return (f"🧠 Cache miss · closest cached question similarity {m['similarity']:.4f} "
            f"(distance {m['distance']:.4f} > {m['threshold']}) → answered by Gemini and stored")


def _render(m: dict) -> None:
    st.caption(_badge(m))
    st.markdown(m["answer"])
    if m.get("sources"):
        with st.expander("📚 Sources"):
            for s in m["sources"]:
                st.caption(s)


# ── Sidebar: cache stats ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Semantic cache")
    stats, err = _api("GET", "/stats")
    if err:
        st.error(err)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Hits", stats["hits"])
        c2.metric("Misses", stats["misses"])
        c1.metric("Hit rate", stats["hit_rate"])
        c2.metric("Entries", stats["entries"] if stats["entries"] is not None else "?")
        st.metric("Cost saved", stats["cost_saved_usd"])
        st.caption(f"Cache `{stats['cache_name']}` · threshold {stats['threshold']} · priced as {stats['cost_model']}")
    if st.button("🗑️ Clear cache"):
        _, err = _api("DELETE", "/cache")
        st.error(err) if err else st.rerun()

# ── Chat ─────────────────────────────────────────────────────────────────────
st.title("⚡ Kestrel RAG + Semantic Cache")
st.markdown("Ask about the (fictional) Kestrel platform, its API, or the Project Heron timeline. "
            "Ask the same thing twice, or reworded, to see the cache hit.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        _render(m) if m["role"] == "assistant" else st.markdown(m["content"])

if question := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result, err = _api("POST", "/chat", json={"question": question})
        if err:
            st.error(err)
        else:
            _render(result)
            st.session_state.messages.append({"role": "assistant", **result})
            st.rerun()  # refresh the sidebar stats
