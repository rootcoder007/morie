"""Tests for gh_ap_h1.ghosal_inv_gauss."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_h1 import ghosal_inv_gauss


def test_gh_ap_h1_basic():
    """Test basic functionality."""
    # ghosal_inv_gauss operates on a single scalar x; iterate over values.
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    vals = np.array([ghosal_inv_gauss(x)["estimate"] for x in xs], dtype=float)
    # Independently compute the literature formula on each x.
    import math
    alpha_loc = 1.0
    gamma_sh = 2.0
    expected = np.array(
        [
            math.sqrt(gamma_sh / (2.0 * math.pi * x ** 3))
            * math.exp(-gamma_sh * (x - alpha_loc) ** 2
                        / (2.0 * alpha_loc ** 2 * x))
            for x in xs
        ],
        dtype=float,
    )
    assert "estimate" in ghosal_inv_gauss(1.0)
    assert np.all(np.isfinite(vals))
    assert np.allclose(vals, expected)


def test_gh_ap_h1_edge():
    """Test edge cases."""
    result = ghosal_inv_gauss(42.0)
    # Function returns a scalar estimate for a scalar x; no batch dimension.
    assert "estimate" in result
    assert np.isfinite(float(result["estimate"]))
    # Quadrature-based mass check should confirm unit total mass.
    assert result["normalized"] is True
