from google.genai import types

from rag import agent


def _resp(parts):
    """Build a fake generate_content response with the given Part list, matching
    the real google-genai response shape (candidates[0].content.parts)."""
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content)

    class FakeResponse:
        candidates = [candidate]

    return FakeResponse()


def test_agentic_answer_stops_immediately_when_no_tool_call(monkeypatch):
    monkeypatch.setattr(agent, "_call_with_retry", lambda client, contents: _resp([types.Part(text="Direct answer, no search needed.")]))

    result = agent.agentic_answer("trivial question")

    assert result["answer"] == "Direct answer, no search needed."
    assert result["tool_calls"] == []
    assert result["hits"] == []
    assert result["n_iterations"] == 1


def test_agentic_answer_executes_tool_call_then_stops(monkeypatch):
    responses = [
        _resp([types.Part(function_call=types.FunctionCall(id="call_1", name="search_filings", args={"query": "AI risk", "sector": "tech"}))]),
        _resp([types.Part(text="Final answer using the search result.")]),
    ]
    monkeypatch.setattr(agent, "_call_with_retry", lambda client, contents: responses.pop(0))
    monkeypatch.setattr(agent, "_search_filings", lambda query, sector=None, ticker=None: (
        [{"id": "chunk-1", "text": "t", "metadata": {"ticker": "GOOGL", "sector": "tech", "filing_date": "2025-01-01"}}]
    ))
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    result = agent.agentic_answer("compare tech vs banks")

    assert result["answer"] == "Final answer using the search result."
    assert result["tool_calls"] == [{"query": "AI risk", "sector": "tech", "ticker": None}]
    assert [h["id"] for h in result["hits"]] == ["chunk-1"]
    assert result["n_iterations"] == 2


def test_agentic_answer_dedupes_hits_across_multiple_tool_calls(monkeypatch):
    responses = [
        _resp([types.Part(function_call=types.FunctionCall(id="call_1", name="search_filings", args={"query": "q1", "sector": "tech"}))]),
        _resp([types.Part(function_call=types.FunctionCall(id="call_2", name="search_filings", args={"query": "q2", "sector": "banks"}))]),
        _resp([types.Part(text="Combined answer.")]),
    ]
    monkeypatch.setattr(agent, "_call_with_retry", lambda client, contents: responses.pop(0))

    fake_meta = {"ticker": "GOOGL", "sector": "tech", "filing_date": "2025-01-01"}

    def fake_search(query, sector=None, ticker=None):
        # "shared" appears from both calls -> should be deduped, not doubled.
        return [{"id": "shared", "text": "t", "metadata": fake_meta}, {"id": f"only-{sector}", "text": "t", "metadata": fake_meta}]

    monkeypatch.setattr(agent, "_search_filings", fake_search)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    result = agent.agentic_answer("compare tech vs banks")

    assert len(result["tool_calls"]) == 2
    hit_ids = sorted(h["id"] for h in result["hits"])
    assert hit_ids == ["only-banks", "only-tech", "shared"]  # 3 unique, not 4


def test_agentic_answer_uses_shared_citation_numbering_across_tool_calls(monkeypatch):
    """Regression test for a real reported bug: each tool call's passages were
    numbered [1]-[N] independently (rag.generate.build_context always restarts at 1),
    so a second call's passages (e.g. the banking side of a sector comparison) were
    shown to the model as [1]-[N] while actually being hits N+1..2N in the final
    deduped `hits` list the UI numbers in its Sources section — the model's citations
    and the on-screen sources silently disagreed by exactly N."""
    responses = [
        _resp([types.Part(function_call=types.FunctionCall(id="call_1", name="search_filings", args={"query": "q1", "sector": "tech"}))]),
        _resp([types.Part(function_call=types.FunctionCall(id="call_2", name="search_filings", args={"query": "q2", "sector": "banks"}))]),
        _resp([types.Part(text="Combined answer.")]),
    ]
    seen_contents = []

    def fake_call(client, contents):
        seen_contents.append(list(contents))
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_with_retry", fake_call)

    meta = {"ticker": "X", "sector": "s", "filing_date": "2025-01-01"}

    def fake_search(query, sector=None, ticker=None):
        return [{"id": f"{sector}-{i}", "text": "t", "metadata": meta} for i in range(2)]

    monkeypatch.setattr(agent, "_search_filings", fake_search)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    result = agent.agentic_answer("compare tech vs banks")

    # seen_contents[2] is what the model was shown right before producing the final
    # answer: [user_query, model_fc_tech, user_response_tech, model_fc_banks, user_response_banks]
    final_call_contents = seen_contents[2]
    tech_response_text = final_call_contents[2].parts[0].function_response.response["result"]
    banks_response_text = final_call_contents[4].parts[0].function_response.response["result"]

    assert "[1]" in tech_response_text and "[2]" in tech_response_text
    assert "[3]" in banks_response_text and "[4]" in banks_response_text
    assert "[1]" not in banks_response_text and "[2]" not in banks_response_text

    # The numbering the model saw must match the order the UI will display in Sources.
    assert [h["id"] for h in result["hits"]] == ["tech-0", "tech-1", "banks-0", "banks-1"]


def test_agentic_answer_respects_max_iterations_cap(monkeypatch):
    """Regression guard for the safety cap the course's own bare example lacks —
    if the model keeps calling tools forever, we must force a stop, not loop
    indefinitely."""
    call_count = {"n": 0}

    def always_calls_tool(client, contents):
        call_count["n"] += 1
        # After the forced-stop message is appended, return a final answer instead
        # of yet another tool call, so the test terminates.
        last_part = contents[-1].parts[0]
        if last_part.text and "maximum number of searches" in last_part.text:
            return _resp([types.Part(text="Forced final answer.")])
        return _resp([types.Part(function_call=types.FunctionCall(id=f"call_{call_count['n']}", name="search_filings", args={"query": "q"}))])

    monkeypatch.setattr(agent, "_call_with_retry", always_calls_tool)
    monkeypatch.setattr(agent, "_search_filings", lambda query, sector=None, ticker=None: [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    result = agent.agentic_answer("question that never converges", max_iterations=3)

    assert result["answer"] == "Forced final answer."
    assert result["n_iterations"] == 3
    assert len(result["tool_calls"]) == 3  # one tool call per iteration up to the cap
