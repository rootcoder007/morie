"""Tests for bdmnto.bound_monot_outcome."""

from morie.fn import _array_core as np

from morie.fn.bdmnto import bound_monot_outcome


def test_bdmnto_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_z = np.random.default_rng(42)
    y = rng_y.uniform(0.0, 1.0, 100)
    z = rng_z.uniform(0.0, 1.0, 100)
    d = 0.5
    ymin = 0.0
    ymax = 1.0
    result = bound_monot_outcome(y, z, d, ymin, ymax)

    # Documented keys returned by bound_monot_outcome.
    for key in ("lower", "upper", "width", "nfixed", "n", "d"):
        assert key in result

    n = len(y)
    d_f = float(d)

    # Independent recomputation of L, U, lower, upper, width, nfixed.
    L = [y[i] if z[i] <= d_f else ymin for i in range(n)]
    U = [y[i] if z[i] >= d_f else ymax for i in range(n)]
    lb = sum(L) / n
    ub = sum(U) / n

    assert result["lower"] == lb
    assert result["upper"] == ub
    assert result["width"] == ub - lb
    assert result["n"] == n
    assert result["nfixed"] == sum(1 for i in range(n) if z[i] == d_f)
    assert result["d"] == d_f

    # Lower <= upper by construction.
    assert result["lower"] <= result["upper"]
    # Bounds lie inside the a priori support.
    assert ymin <= result["lower"] <= ymax
    assert ymin <= result["upper"] <= ymax


def test_bdmnto_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_z = np.random.default_rng(42)
    y = rng_y.uniform(0.0, 1.0, 100)
    z = rng_z.uniform(0.0, 1.0, 100)
    d = 0.5
    ymin = 0.0
    ymax = 1.0
    result = bound_monot_outcome(y, z, d, ymin, ymax)

    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "nfixed", "n", "d"):
        assert key in result

    # With z and y drawn i.i.d. uniformly on [0, 1], we generally expect
    # at least some non-fixed units and width bounded by ymax - ymin.
    assert result["nfixed"] <= result["n"]
    assert result["width"] <= ymax - ymin
    assert result["width"] >= 0.0
