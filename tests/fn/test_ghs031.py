"""Tests for ghs031.ghosal_ch3_polya_tree_mixture_post_density."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ghs031 import ghosal_ch3_polya_tree_mixture_post_density


def _normal_cdf(z):
    """Standard normal CDF using math.erf (independent of the function)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _normal_pdf(z):
    """Standard normal PDF."""
    from math import sqrt, exp, pi
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)


def test_ghs031_basic():
    """Test basic functionality against the documented formula."""
    depth = 8
    n = 16
    a_of_level = lambda m: float(m * m)
    a_levels = [a_of_level(j) for j in range(1, depth + 1)]

    # Standard normal parametric family: G_theta = Phi, g_theta = phi at theta=0
    g_theta = lambda v: _normal_pdf(v)
    G_theta = lambda v: _normal_cdf(v)

    data = [float(v) for v in np.random.default_rng(42).normal(0.0, 1.0, n)]
    x = 0.3

    # Expected: g_theta(x0) * prod_j (2 a_j + 2 N*_j) / (2 a_j + N*_{j-1})
    a0 = a_levels[0]
    us = [G_theta(v) for v in data]
    u0 = G_theta(x)
    int_part = 0
    for j in range(depth):
        cell = int((us[0] * (2 ** j))) if False else None  # placeholder, do manual
    # Compute dyadic path counts N*_j at each level on the G_theta scale
    N_star = []
    for j in range(depth + 1):  # levels 0..depth, where level 0 is the root (0)
        if j == 0:
            count = len([u for u in us if 0.0 <= u < 1.0])
        else:
            lo = 0.0
            hi = 1.0
            bin_idx = int(u0 * (2 ** j))
            # cells are [k/2^j, (k+1)/2^j)
            lo = bin_idx / (2 ** j)
            hi = (bin_idx + 1) / (2 ** j)
            count = len([u for u in us if lo <= u < hi])
        N_star.append(count)

    # Level j in the product corresponds to 2 a_j + 2 N*_j over 2 a_j + N*_{j-1}
    # j indexes 1..depth (level 1..depth in the docstring)
    product = 1.0
    for j in range(1, depth + 1):
        jm1 = j - 1
        aj = a_levels[jm1]  # a_of_level(j)
        num = 2.0 * aj + 2.0 * N_star[j]
        den = 2.0 * aj + N_star[jm1]
        product *= num / den
    expected_core = product
    expected_dens = float(g_theta(x)) * expected_core

    result = ghosal_ch3_polya_tree_mixture_post_density(
        x, data, g_theta, G_theta, a_of_level=a_of_level, depth=depth
    )

    # The result should be a RichResult (dict-like) containing documented keys
    assert isinstance(result, dict)
    assert "posterior" in result
    assert "estimate" in result
    assert "uniform_scale_density" in result
    assert "method" in result

    # Posterior density value matches the documented formula
    assert abs(float(result["posterior"]) - expected_dens) < 1e-12
    # estimate equals posterior per the implementation contract
    assert abs(float(result["estimate"]) - float(result["posterior"])) < 1e-15
    # uniform_scale_density equals the product (core) on the G_theta scale
    assert abs(float(result["uniform_scale_density"]) - expected_core) < 1e-12


def test_ghs031_edge():
    """Test edge cases."""
    depth = 6
    n = 32
    a_of_level = lambda m: float(m * m)

    g_theta = lambda v: _normal_pdf(v)
    G_theta = lambda v: _normal_cdf(v)

    data = [float(v) for v in np.random.default_rng(43).normal(0.0, 1.0, n)]
    x = 1.0

    result = ghosal_ch3_polya_tree_mixture_post_density(
        x, data, g_theta, G_theta, a_of_level=a_of_level, depth=depth
    )

    assert isinstance(result, dict)
    assert result.get("method", "").startswith("PT mixture posterior density")
    # Posterior density must be non-negative
    assert float(result["posterior"]) >= 0.0
    assert float(result["estimate"]) >= 0.0
    assert float(result["uniform_scale_density"]) >= 0.0
