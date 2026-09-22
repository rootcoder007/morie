"""Tests for aitcap.compositional_classifyAP."""

from morie.fn import _array_core as np

from morie.fn.aitcap import compositional_classifyAP


def test_aitcap_basic():
    """Test basic functionality on strictly positive compositions."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    X = np.abs(X) + 0.1  # enforce strictly positive parts
    y = rng.integers(0, 3, 100).astype(float)
    x_new = np.abs(rng.normal(0, 1, (3, 5))) + 0.1
    k = 5
    result = compositional_classifyAP(X, y, x_new, k)

    # Keys documented in the implementation's payload
    assert "yhat" in result
    assert "yhat_majority" in result
    assert "dist" in result
    assert "k" in result and result["k"] == 5
    assert "n" in result and result["n"] == 100
    assert "D" in result and result["D"] == 5
    assert "method" in result
    assert "estimate" in result

    # With multiple new compositions, yhat/yhat_majority are arrays of length 3
    assert hasattr(result["yhat"], "__len__") and len(result["yhat"]) == 3
    assert hasattr(result["yhat_majority"], "__len__") and len(result["yhat_majority"]) == 3

    # y values must come from the training labels {0.0, 1.0, 2.0}
    for v in result["yhat"]:
        assert v in (0.0, 1.0, 2.0)

    # dist has length N
    assert hasattr(result["dist"], "__len__") and len(result["dist"]) == 100


def test_aitcap_edge():
    """Test edge cases: single new composition returns scalars, not arrays."""
    rng = np.random.default_rng(42)
    X = np.abs(rng.normal(0, 1, (20, 3))) + 0.1
    y = rng.integers(0, 2, 20).astype(float)
    x_new = np.abs(rng.normal(0, 1, 3)) + 0.1  # 1-D single composition
    k = 3
    result = compositional_classifyAP(X, y, x_new, k)

    assert "yhat" in result and "yhat_majority" in result and "dist" in result
    assert "D" in result and result["D"] == 3
    assert "n" in result and result["n"] == 20
    assert "k" in result and result["k"] == 3

    # Scalar return for a single new composition
    assert not hasattr(result["yhat"], "__len__")
    assert not hasattr(result["yhat_majority"], "__len__")
    assert result["yhat"] in (0.0, 1.0)
