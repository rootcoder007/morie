"""Tests for gb47ad.gibbons_anderson_darling."""

import math

from morie.fn import _array_core as np
from morie.fn.gb47ad import gibbons_anderson_darling


def _norm_cdf(v):
    return 0.5 * math.erfc(-v / math.sqrt(2.0))


def test_gb47ad_basic():
    """Test basic functionality: W_n^2 computed from the documented formula."""
    rng = np.random.default_rng(42)
    x = list(rng.normal(0, 1, 100))

    result = gibbons_anderson_darling(x, _norm_cdf)

    # Documented return keys (RichResult behaves dict-like)
    assert isinstance(result, dict)
    for key in ("statistic", "astar", "crit", "reject",
                "z", "n", "case", "alpha", "method"):
        assert key in result

    # Independent computation of W_n^2 from eq. (4.7.1)
    xs_sorted = sorted(float(v) for v in x)
    n = len(xs_sorted)
    z_vals = [float(_norm_cdf(v)) for v in xs_sorted]
    assert all(0.0 < v < 1.0 for v in z_vals)
    s = 0.0
    for j in range(1, n + 1):
        s += (2.0 * j - 1.0) * (
            math.log(z_vals[j - 1]) + math.log(1.0 - z_vals[n - j])
        )
    expected_a2 = -n - s / n

    assert result["n"] == n
    assert result["case"] == "specified"
    assert result["alpha"] == 0.05
    assert result["method"].startswith("Anderson-Darling")
    assert result["statistic"] == float(expected_a2)
    # For "specified" case, astar equals the raw statistic.
    assert result["astar"] == float(expected_a2)
    # reject is 1 when astar > crit, else 0.
    assert result["reject"] == int(result["astar"] > result["crit"])
    assert result["z"] == z_vals


def test_gb47ad_edge():
    """Test edge cases: a small sample with the 'normal-both' case and a custom alpha."""
    rng = np.random.default_rng(42)
    x = list(rng.normal(0, 1, 25))

    result = gibbons_anderson_darling(x, _norm_cdf,
                                      case="normal-both", alpha=0.10)

    assert isinstance(result, dict)
    assert result["case"] == "normal-both"
    assert result["alpha"] == 0.10
    assert result["n"] == 25

    # Independent A* = W^2 * (1 + 0.75/n + 2.25/n^2) for "normal-both".
    n = 25
    expected_astar_factor = 1.0 + 0.75 / n + 2.25 / (n * n)
    assert result["astar"] == result["statistic"] * expected_astar_factor
