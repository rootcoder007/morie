"""Tests for ksr08.kosorok_multiplier_bootstrap."""

import math

from morie.fn import _array_core as np
from morie.fn.ksr08 import kosorok_multiplier_bootstrap


def test_ksr08_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_multiplier_bootstrap(x)
    assert isinstance(result, dict)
    assert "estimate" in result
    estimate = result["estimate"]
    # Ensure the estimate is a numeric type and finite.
    # If it's an array-like, check each element; otherwise check the scalar.
    if hasattr(estimate, "__iter__") and not isinstance(estimate, str):
        for val in estimate:
            assert math.isfinite(float(val))
    else:
        assert math.isfinite(float(estimate))


def test_ksr08_edge():
    """Test edge cases."""
    result = kosorok_multiplier_bootstrap(np.array([42.0, 43.0]))
    assert isinstance(result, dict)
    assert "estimate" in result
    estimate = result["estimate"]
    if hasattr(estimate, "__iter__") and not isinstance(estimate, str):
        # Check mean and first element
        vals = list(estimate)
        assert math.isfinite(float(np.mean(vals)))
        assert math.isfinite(float(vals[0]))
    else:
        assert math.isfinite(float(estimate))


def test_exponential_multiplier_bootstrap_replayed():
    import pytest

    from morie.fn._tail1core import Lcg

    x = [1.0, 2.5, 3.0, 4.5, 7.0]
    n, B = 5, 10
    g = Lcg(5)
    st = []
    for _ in range(B):
        w = [-math.log(g.unif()) for _ in range(n)]
        wb = sum(w) / n
        st.append(sum(w[i] / wb * x[i] for i in range(n)) / n)
    r = kosorok_multiplier_bootstrap(x, B=B, seed=5)
    assert r["boot_mean"] == pytest.approx(sum(st) / B, rel=1e-13)
    assert r["estimate"] == pytest.approx(sum(x) / n, rel=1e-15)
