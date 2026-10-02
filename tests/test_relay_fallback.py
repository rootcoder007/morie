"""The relay never returns "LLM request failed" with status 200: it falls back to the
provider chain and says 503 when nothing can answer."""

from __future__ import annotations

from unittest.mock import patch

from morie.perseus_relay import answer_question


class _Resp:
    def __init__(self, text: str, failed: bool = False) -> None:
        self.text = text
        self.failed = failed
        self.tool_calls_made: list = []
        self.iterations = 1
        self.model = "perseus:test"


class _Agent:
    def __init__(self, resp) -> None:
        self._resp = resp

    def chat(self, question: str):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


HOSTED = {"mode": "hosted", "model": "hosted-model", "output_text": "pong"}
NOTHING = {"mode": "local_fallback", "model": "local", "output_text": "static text"}


def test_agent_answer_passes_through():
    code, data = answer_question(_Agent(_Resp("Moran's I measures spatial autocorrelation.")), "q")
    assert code == 200 and data["backend"] == "agent" and "Moran" in data["text"]


@patch("morie.perseus.ask_percy", return_value=HOSTED)
def test_failed_agent_answer_falls_back_to_provider_chain(mock_ask):
    code, data = answer_question(_Agent(_Resp("LLM request failed: [Errno 111]", failed=True)), "q")
    assert code == 200 and data["text"] == "pong" and data["backend"] == "hosted"
    mock_ask.assert_called_once_with("q")


@patch("morie.perseus.ask_percy", return_value=HOSTED)
def test_agent_exception_falls_back_to_provider_chain(mock_ask):
    code, data = answer_question(_Agent(RuntimeError("backend died")), "q")
    assert code == 200 and data["text"] == "pong"


@patch("morie.perseus.ask_percy", return_value=HOSTED)
def test_no_agent_uses_provider_chain(mock_ask):
    code, data = answer_question(None, "q")
    assert code == 200 and data["model"] == "hosted-model"


@patch("morie.perseus.ask_percy", return_value=NOTHING)
def test_nothing_reachable_is_503(mock_ask):
    code, data = answer_question(None, "q")
    assert code == 503 and "morie login" in data["error"] and data["backend"] == "local_fallback"
