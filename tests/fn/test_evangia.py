"""Tests for evangia.evt_angular_measure."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.evangia import evt_angular_measure


def test_evangia_basic():
    """Test basic functionality with explicit, verifiable inputs."""
    # Use a deterministic 5x2 matrix of strictly positive observations so that
    # the rank transform produces valid Frechet margins > 0.
    X = [
        [0.5, 1.5],
        [0.7, 0.3],
        [2.0, 1.0],
        [1.2, 0.8],
        [0.4, 2.5],
    ]
    k = 2
    result = evt_angular_measure(X, k)

    assert isinstance(result, dict)

    # Independent reproduction of the documented formula.
    n = len(X)
    c0 = [row[0] for row in X]
    c1 = [row[1] for row in X]

    def avg_ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        return ranks

    r0 = avg_ranks(c0)
    r1 = avg_ranks(c1)
    f0 = [(n + 1.0) / (n + 1.0 - v) for v in r0]
    f1 = [(n + 1.0) / (n + 1.0 - v) for v in r1]
    rad = [f0[i] + f1[i] for i in range(n)]

    # k largest radii (ties broken by lower index, matching the implementation).
    idx = sorted(range(n), key=lambda i: (-rad[i], i))[:k]
    expected_atoms = sorted(f0[i] / rad[i] for i in idx)
    expected_H = 1.0
    expected_mean = sum(expected_atoms) / k

    # Keys returned by the function per its docstring.
    assert "H" in result
    assert "atoms" in result
    assert "weights" in result
    assert "estimate" in result
    assert "n_used" in result
    assert "n" in result

    assert result["n"] == n
    assert result["n_used"] == k
    assert result["H"] == expected_H
    assert result["weights"] == [1.0 / k] * k
    assert result["atoms"] == expected_atoms
    assert result["estimate"] == expected_mean


def test_evangia_edge():
    """Test edge cases (n==k, smallest valid input)."""
    X = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
    ]
    k = 3
    result = evt_angular_measure(X, k)

    assert isinstance(result, dict)

    n = len(X)
    c0 = [row[0] for row in X]
    c1 = [row[1] for row in X]

    def avg_ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        return ranks

    r0 = avg_ranks(c0)
    r1 = avg_ranks(c1)
    f0 = [(n + 1.0) / (n + 1.0 - v) for v in r0]
    f1 = [(n + 1.0) / (n + 1.0 - v) for v in r1]
    rad = [f0[i] + f1[i] for i in range(n)]
    idx = sorted(range(n), key=lambda i: (-rad[i], i))[:k]
    expected_atoms = sorted(f0[i] / rad[i] for i in idx)
    expected_mean = sum(expected_atoms) / k

    assert result["n"] == n
    assert result["n_used"] == k
    assert result["H"] == 1.0
    assert result["atoms"] == expected_atoms
    assert result["estimate"] == expected_mean
    assert result["weights"] == [1.0 / k] * k
