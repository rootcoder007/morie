"""Tests for bigtm.bigram_topic."""

from morie.fn import _array_core as np

from morie.fn.bigtm import bigram_topic


def test_bigtm_basic():
    """Test basic functionality."""
    # docs must be a list of documents, each a list of word indices in [0, V).
    V = 10
    T = 4
    rng = np.random.default_rng(42)
    # Three short documents with word indices in the vocabulary [0, V).
    docs = [
        [int(v) for v in rng.integers(0, V, size=8)],
        [int(v) for v in rng.integers(0, V, size=6)],
        [int(v) for v in rng.integers(0, V, size=10)],
    ]
    result = bigram_topic(docs, T, V)
    # Returned object behaves like a mapping with the documented keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "z" in result
    assert "topic_posterior" in result
    assert "theta" in result
    # Shapes match the document structure.
    assert len(result["estimate"]) == len(docs)
    for d, doc in enumerate(docs):
        assert len(result["estimate"][d]) == len(doc)
    # theta has one row per document and T columns.
    for d in range(len(docs)):
        assert len(result["theta"][d]) == T


def test_bigtm_edge():
    """Test edge cases."""
    V = 10
    T = 4
    rng = np.random.default_rng(42)
    # Two tiny documents; bigram model still runs (first token unassigned).
    docs = [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8],
    ]
    result = bigram_topic(docs, T, V)
    assert isinstance(result, dict)
    assert "estimate" in result
    # Every assigned topic index is within the topic range [0, T).
    for d, row in enumerate(result["estimate"]):
        assert len(row) == len(docs[d])
        for k in row:
            assert 0 <= int(k) < T
