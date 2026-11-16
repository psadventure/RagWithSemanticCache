import asyncio

import pytest
from google.adk.tools import FunctionTool

from agent.rag_cache_agent import tools
from agent.rag_cache_agent.agent import root_agent
from rag import service

TOOLS = {
    "ask": ("answer", ("q",)),
    "search_chunks": ("search", ("q",)),
    "check_cache": ("check_cache", ("q",)),
    "cache_stats": ("cache_stats", ()),
    "clear_cache": ("clear_cache", ()),
}


@pytest.mark.parametrize("tool_name", TOOLS)
def test_tools_return_error_dict_instead_of_raising(tool_name, monkeypatch):
    service_fn, args = TOOLS[tool_name]

    async def boom(*a, **k):
        raise ConnectionError("Valkey is down")

    monkeypatch.setattr(service, service_fn, boom)
    result = asyncio.run(getattr(tools, tool_name)(*args))
    assert result == {"error": "ConnectionError: Valkey is down"}


def test_ask_passes_through_on_success(offline):
    result = asyncio.run(tools.ask("How long is an access token valid?"))
    assert result["cache_hit"] is False and "answer" in result


def test_agent_wires_all_tools_and_confirms_clear():
    by_name = {getattr(t, "name", getattr(t, "__name__", None)): t for t in root_agent.tools}
    assert set(by_name) == set(TOOLS)
    clear = by_name["clear_cache"]
    assert isinstance(clear, FunctionTool) and clear._require_confirmation is True
