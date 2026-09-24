"""Tests for hmicl.geron_in_context_learning."""

from morie.fn import _array_core as np

from morie.fn.hmicl import geron_in_context_learning


def _scorer(prompt, cand):
    """Simple deterministic scorer: log-prob proportional to count of candidate in prompt."""
    return float(prompt.count(str(cand)))


def test_hmicl_basic():
    """Test basic functionality."""
    examples = [("a", "pos"), ("b", "pos"), ("c", "neg")]
    query = "d"
    result = geron_in_context_learning(_scorer, examples, query)
    assert isinstance(result, dict)
    for key in ("prediction", "prompt", "log_probs", "posterior",
                "n_shot", "candidates", "estimate", "n", "method"):
        assert key in result
    assert result["prediction"] == "pos"
    assert result["n_shot"] == 3
    assert sorted(result["candidates"]) == ["neg", "pos"]
    assert abs(float(np.sum(result["posterior"])) - 1.0) < 1e-9
    assert result["prompt"].splitlines() == [
        "a -> pos", "b -> pos", "c -> neg", "d ->"
    ]


def test_hmicl_edge():
    """Test edge cases (zero-shot with explicit candidates)."""
    result = geron_in_context_learning(
        _scorer, [], "d", candidates=["pos", "neg"]
    )
    assert isinstance(result, dict)
    assert result["n_shot"] == 0
    assert result["prompt"] == "d ->"
    assert sorted(result["candidates"]) == ["neg", "pos"]
    assert abs(float(np.sum(result["posterior"])) - 1.0) < 1e-9


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmicl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
