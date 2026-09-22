"""Tests for gh_c9_6.ghosal_wishart_dpm."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_6 import ghosal_wishart_dpm


def test_gh_c9_6_basic():
    """Test basic functionality."""
    # Build query points as a plain Python tuple of floats, matching the
    # documented signature ``x_query=(0.0, 1.0)``.
    x = (0.0, 1.0, 2.0, 3.0, 4.0)
    result = ghosal_wishart_dpm(x)
    assert "estimate" in result
    # The returned "density" is a list (one entry per query point); every
    # entry must be finite (a well-defined Gaussian mixture density).
    density = result["density"]
    assert isinstance(density, list)
    assert len(density) == len(x)
    assert np.all(np.isfinite(np.asarray(density, dtype=float)))
    # "estimate" must equal the first entry of "density" by definition.
    assert result["estimate"] == density[0]


def test_gh_c9_6_edge():
    """Test edge cases."""
    # A single query point still returns a list of length 1 in "density"
    # and a scalar in "estimate"; the function never returns a key named
    # "n".  We verify the documented keys instead.
    result = ghosal_wishart_dpm((42.0,))
    assert isinstance(result["density"], list)
    assert len(result["density"]) == 1
    assert np.isfinite(float(result["estimate"]))
    # The quadrature-based total mass must be a positive finite number
    # (the mixture density is positive and the trapezoidal rule over
    # [-8, 16] with step 16/400 is finite).
    assert np.isfinite(float(result["total_mass"]))
    assert float(result["total_mass"]) > 0.0
    # The "method" key is a short string documenting the literature source.
    assert isinstance(result["method"], str)
