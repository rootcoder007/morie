"""Tests for pcadm.pca_dimension_reduction."""

from morie.fn import _array_core as np
from morie.fn.pcadm import pca_dimension_reduction


def test_pcadm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pca_dimension_reduction(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_pcadm_edge():
    """Test edge cases."""
    result = pca_dimension_reduction(np.array([42.0]))
    assert result["n"] == 1


def test_two_column_pca_recomputed():
    import math

    import pytest

    X = [[1.0, 2.0], [2.0, 3.5], [3.0, 3.0], [4.0, 6.0], [5.0, 5.5]]
    n = 5
    m = [sum(r[j] for r in X) / n for j in range(2)]
    s = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in X) / (n - 1) for b in range(2)] for a in range(2)]
    tr, det = s[0][0] + s[1][1], s[0][0] * s[1][1] - s[0][1] ** 2
    d = math.sqrt(tr * tr / 4 - det)
    ev = [tr / 2 + d, tr / 2 - d]
    r = pca_dimension_reduction(X)
    assert r["explained_variance"] == pytest.approx(ev, rel=1e-10)
    assert r["estimate"] == pytest.approx(ev[0] / sum(ev), rel=1e-10)
    assert r["singular_values"] == pytest.approx([math.sqrt(v * (n - 1)) for v in ev], rel=1e-10)
