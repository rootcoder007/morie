"""Tests for alreact.alammar_react_agent_loop."""

from morie.fn.alreact import alammar_react_agent_loop


def test_alreact_basic():
    model = lambda ctx: ({"thought": "t", "action": "f",
                          "action_input": 1} if "query" in ctx[-1]
                         else {"thought": "t", "final": ctx[-1]["observation"]})
    out = alammar_react_agent_loop("q", {"f": lambda x: "7"}, model)
    assert out["answer"] == "7"


def test_alreact_edge():
    out = alammar_react_agent_loop("q", {},
        lambda c: {"thought": "t", "action": "nope"}, max_steps=1)
    assert out["exhausted"] is True
