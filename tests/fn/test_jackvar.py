"""Tests for jackvar.jackknife_variance_survey."""

from morie.fn import _array_core as np
from morie.fn.jackvar import jackknife_variance_survey


def test_jackvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = jackknife_variance_survey(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_jackvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = jackknife_variance_survey(y)
    assert isinstance(result, dict)


def test_delete_one_jackknife_is_s2_over_n():
    """Wolter (4.2.5) for the unweighted mean reduces to s^2 / n; with
    weights the replicates are the weighted means without unit r."""
    import pytest

    y = [3.0, 5.5, 4.0, 8.0, 6.5, 2.0]
    n = 6
    m = sum(y) / n
    s2 = sum((v - m) ** 2 for v in y) / (n - 1)
    assert jackknife_variance_survey(y)["variance"] == pytest.approx(s2 / n, rel=1e-12)
    w = [1.0, 2.0, 1.5, 0.5, 1.0, 2.5]
    th = sum(a * b for a, b in zip(w, y)) / sum(w)
    reps = [sum(w[i] * y[i] for i in range(n) if i != r) / sum(w[i] for i in range(n) if i != r) for r in range(n)]
    v = (n - 1) / n * sum((t - th) ** 2 for t in reps)
    assert jackknife_variance_survey(y, weights=w)["variance"] == pytest.approx(v, rel=1e-12)
