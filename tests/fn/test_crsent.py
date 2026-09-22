"""Tests for crsent.cross_entropy."""

from morie.fn import _array_core as np

from morie.fn.crsent import cross_entropy


def test_crsent_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 1.0, 100)
    q = rng.uniform(0.0, 1.0, 100)
    p = p / p.sum()
    q = q / q.sum()
    result = cross_entropy(p, q, 2.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "entropy" in result
    assert "kl" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 100

    expected_ce = 0.0
    for i in range(100):
        expected_ce -= p[i] * (float(np.log(q[i])) / float(np.log(2.0)))
    assert abs(result["estimate"] - expected_ce) < 1e-9


def test_crsent_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 1.0, 50)
    q = rng.uniform(0.0, 1.0, 50)
    p = p / p.sum()
    q = q / q.sum()
    result = cross_entropy(p, q, 2.0)
    assert isinstance(result, dict)
    assert result["n"] == 50

    expected_ce = 0.0
    for i in range(50):
        expected_ce -= p[i] * (float(np.log(q[i])) / float(np.log(2.0)))
    assert abs(result["estimate"] - expected_ce) < 1e-9
    assert abs(result["kl"] - (result["estimate"] - result["entropy"])) < 1e-9
