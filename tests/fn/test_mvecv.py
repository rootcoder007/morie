"""Tests for mvecv.min_volume_ellipsoid."""

import math

from morie.fn import _array_core as np

from morie.fn.mvecv import min_volume_ellipsoid


def test_mvecv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    q = 3
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, q))
    h = 50  # must exceed p = q + 1 = 4
    result = min_volume_ellipsoid(y, X, h)
    # RichResult may be dict-like or expose the payload via an attribute
    if hasattr(result, "payload"):
        data = result.payload
    else:
        data = result
    assert "estimate" in data
    assert "coef" in data
    assert "intercept" in data
    assert "center" in data
    assert "cov" in data
    assert "subset" in data
    assert "covered" in data
    assert "h" in data
    assert "n" in data
    assert "p" in data
    assert data["n"] == n
    assert data["p"] == q + 1
    assert data["h"] == h
    assert len(data["coef"]) == q
    assert len(data["center"]) == q + 1
    assert math.isfinite(data["estimate"])
    assert math.isfinite(data["intercept"])
    for c in data["coef"]:
        assert math.isfinite(c)


def test_mvecv_edge():
    """Test edge cases with default h."""
    rng = np.random.default_rng(42)
    n = 40
    q = 3
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, q))
    # use the default h
    result = min_volume_ellipsoid(y, X)
    if hasattr(result, "payload"):
        data = result.payload
    else:
        data = result
    assert "estimate" in data
    assert "coef" in data
    assert "intercept" in data
    assert data["n"] == n
    assert data["p"] == q + 1
    assert math.isfinite(data["estimate"])
    assert math.isfinite(data["intercept"])
    assert len(data["coef"]) == q
    assert data["h"] > data["p"]
