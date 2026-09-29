"""Tests for morie.fn.elci — empirical likelihood CI."""

import pytest

from morie.fn import _array_core as np
from morie.fn.elci import elci


def test_ci_contains_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300)
    result = elci(x)
    assert result["ci_lower"] < 0.0 < result["ci_upper"]


def test_ci_reasonable_width():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500) + 2.0
    result = elci(x)
    width = result["ci_upper"] - result["ci_lower"]
    assert 0 < width < 2.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        elci(np.array([]))


def test_el_interval_endpoints_are_where_the_ratio_crosses_chi2():
    """Every grid point inside the interval has -2 log R <= chi2_1(0.95)
    and the neighbours just outside exceed it."""
    import pytest

    from morie.fn import _stats_core as st

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 4.2, 2.5, 3.3, 2.7, 3.9, 2.2]
    r = elci(x, grid_points=120)
    crit = st.chi2.ppf(0.95, df=1)
    grid = [float(v) for v in r["grid"]]
    lr = [float(v) for v in r["log_ratios"]]
    inside = [g for g, v in zip(grid, lr) if v <= crit]
    assert r["ci_lower"] == pytest.approx(min(inside), rel=1e-15)
    assert r["ci_upper"] == pytest.approx(max(inside), rel=1e-15)
    assert r["mean"] == pytest.approx(sum(x) / 12, rel=1e-14)
