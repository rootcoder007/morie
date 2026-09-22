"""Tests for btbca.boot_bca_ci."""

from morie.fn import _array_core as np

from morie.fn.btbca import boot_bca_ci


def _mean(xs):
    s = 0.0
    n = 0
    for v in xs:
        s += float(v)
        n += 1
    return s / n


def test_btbca_basic():
    """Test basic functionality.

    theta_hat must be a scalar (the estimate on the original data),
    theta_b must be a 1-D array of bootstrap replicates, x is the
    original sample (1-D), and stat is a callable taking leave-one-out
    samples and returning a scalar.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    rng_b = np.random.default_rng(7)
    theta_b = rng_b.normal(0.0, 1.0, 100)
    theta_hat = float(_mean(x))
    stat = _mean
    alpha = 0.05

    result = boot_bca_ci(theta_hat, theta_b, x, stat, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "z0" in result
    assert "accel" in result


def test_btbca_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    rng_b = np.random.default_rng(7)
    theta_b = rng_b.normal(0.0, 1.0, 100)
    theta_hat = float(_mean(x))
    stat = _mean
    alpha = 0.05

    result = boot_bca_ci(theta_hat, theta_b, x, stat, alpha)
    assert isinstance(result, dict)
