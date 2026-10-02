"""A thinking model that returns an empty answer is asked again with more room, then skipped."""

from __future__ import annotations

import pytest

from morie import llm


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": self._content, "reasoning": "..."}}]}


def test_empty_answer_is_retried_with_a_larger_budget(monkeypatch):
    calls = []

    def fake(base_url, model, messages, *, api_key=None, stream=False, timeout=0, max_tokens=4096):
        calls.append(max_tokens)
        return _Resp("" if len(calls) == 1 else "pong")

    monkeypatch.setattr(llm, "_request_completion", fake)
    assert (
        llm._completion_text("https://gw", "m", [{"role": "user", "content": "hi"}], api_key="k", timeout=5) == "pong"
    )
    assert calls == [4096, 16384]


def test_two_empty_answers_raise_so_the_chain_moves_on(monkeypatch):
    monkeypatch.setattr(llm, "_request_completion", lambda *a, **k: _Resp(None))
    with pytest.raises(llm.EmptyAnswerError):
        llm._completion_text("https://gw", "m", [{"role": "user", "content": "hi"}], api_key="k", timeout=5)
