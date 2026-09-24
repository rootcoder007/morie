"""Tests for kmw2v.kamath_word2vec_skipgram."""

from morie.fn import _array_core as np
import math

from morie.fn.kmw2v import kamath_word2vec_skipgram


def test_kmw2v_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    vocab_size = 10
    embedding_dim = 4
    n_pairs = 20

    # integer centre/context indices in [0, vocab_size), with context != centre
    center_indices = rng.integers(0, vocab_size, n_pairs)
    context_indices = []
    for c in center_indices:
        other = int(rng.integers(0, vocab_size - 1))
        if other >= int(c):
            other += 1
        context_indices.append(other)

    # input / output embedding matrices of shape (vocab_size, embedding_dim)
    V = rng.normal(0, 1, (vocab_size, embedding_dim))
    U = rng.normal(0, 1, (vocab_size, embedding_dim))

    result = kamath_word2vec_skipgram(center_indices, context_indices, V, U)
    assert isinstance(result, dict)
    for key in ("log_likelihood", "mean_log_likelihood", "per_pair",
                "probabilities", "estimate", "vocab_size", "n", "method"):
        assert key in result

    assert math.isfinite(result["log_likelihood"])
    assert math.isfinite(result["mean_log_likelihood"])
    assert result["vocab_size"] == vocab_size
    assert result["n"] == n_pairs
    assert len(result["per_pair"]) == n_pairs
    assert len(result["probabilities"]) == n_pairs
    for p in result["probabilities"]:
        assert 0 <= p <= 1
    assert abs(result["log_likelihood"] - sum(result["per_pair"])) < 1e-10
    assert abs(result["mean_log_likelihood"]
               - result["log_likelihood"] / n_pairs) < 1e-10


def test_kmw2v_edge():
    """Test edge case: single pair with zero embeddings reproduces docstring example."""
    out = kamath_word2vec_skipgram([0], [1], [[0.0], [0.0]], [[0.0], [0.0]])
    assert abs(out["estimate"] + math.log(2)) < 1e-12
    assert out["n"] == 1
    assert out["vocab_size"] == 2
    assert len(out["per_pair"]) == 1
    assert len(out["probabilities"]) == 1
    assert abs(out["probabilities"][0] - 0.5) < 1e-12
    assert math.isfinite(out["log_likelihood"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmw2v as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
