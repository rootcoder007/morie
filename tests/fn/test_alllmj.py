"""Tests for alllmj.alammar_llm_as_judge."""

from morie.fn.alllmj import alammar_llm_as_judge


def test_alllmj_basic():
    out = alammar_llm_as_judge(["a", "b"], "r",
                               lambda r, resp, s: 1.0 if resp == "b" else 0.0)
    assert out["best_response"] == 1


def test_alllmj_edge():
    import pytest
    with pytest.raises(ValueError, match="callable"):
        alammar_llm_as_judge(["a"], "r", "not a function")
