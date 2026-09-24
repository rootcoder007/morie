"""Tests for eslmrf.esl_markov_rf."""

import pytest

from morie.fn import _array_core as np

from morie.fn.eslmrf import esl_markov_rf


def test_eslmrf_basic():
    """Test that exact enumeration is refused past the cap."""
    n = 30
    graph = np.zeros((n, n))
    for i in range(n - 1):
        graph[i, i + 1] = 1
        graph[i + 1, i] = 1
    with pytest.raises(ValueError, match="cap"):
        esl_markov_rf(graph)


def test_eslmrf_edge():
    """Test cap enforcement via list-of-edges input."""
    with pytest.raises(ValueError, match="cap"):
        esl_markov_rf([(i, i + 1) for i in range(30)])
