"""Tests for fzber.fauzi_berry_esseen_quantile."""

from morie.fn import _array_core as np

from morie.fn.fzber import fauzi_berry_esseen_quantile


def test_fzber_basic():
    """Test basic functionality."""
    n = 100
    m = 4
    x = np.linspace(0.0, 2.0, 5)
    result = fauzi_berry_esseen_quantile(x, n, m=m, improved=True)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "bound" in result
    assert "rate" in result
    assert "m" in result
    assert "n" in result
    assert "method" in result

    assert result["m"] == 4
    assert result["n"] == n
    assert result["rate"] == 0.5
    assert result["bound"] == n ** (-0.5)

    # estimate[i] = 2 * Phi(x[i]) - 1 computed independently.
    from math import erf, sqrt
    expected_estimate = [2.0 * 0.5 * (1.0 + erf(float(xi) / sqrt(2.0))) - 1.0
                        for xi in x]
    assert np.allclose(np.asarray(result["estimate"]), np.asarray(expected_estimate))


def test_fzber_edge():
    """Test edge cases."""
    n = 100
    m = 4
    x = np.linspace(0.0, 2.0, 5)
    result = fauzi_berry_esseen_quantile(x, n, m=m, improved=False)

    assert isinstance(result, dict)
    assert result["m"] == 4
    assert result["n"] == n
    assert result["rate"] == 7.0 / 17.0
    assert result["bound"] == n ** (-(7.0 / 17.0))
