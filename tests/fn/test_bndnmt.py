"""Tests for bndnmt.bound_no_monotonicity."""

from morie.fn import _array_core as np

from morie.fn.bndnmt import bound_no_monotonicity


def _binary(rng, n, p):
    """Generate a 0/1 array by thresholding uniform draws."""
    return (rng.uniform(0.0, 1.0, n) < p).astype(float)


def test_bndnmt_basic():
    """Test basic functionality on a constructed 0/1 instrument design."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_Z = np.random.default_rng(43)

    n = 100
    y = rng_y.normal(0.0, 1.0, n)
    D = _binary(rng_D, n, 0.3)
    Z = _binary(rng_Z, n, 0.5)

    result = bound_no_monotonicity(y, D, Z)

    # Returned object should behave like a mapping exposing all documented keys.
    assert isinstance(result, dict)
    for key in (
        "lower", "upper", "width", "estimate", "wald",
        "pi_net", "pi_c_max", "pi_d_max", "itt_y", "n",
    ):
        assert key in result

    # Independently recompute everything from the formula in the docstring.
    yv = np.asarray(y).reshape(-1)
    dv = np.asarray(D).reshape(-1)
    zv = np.asarray(Z).reshape(-1)
    assert yv.shape == (n,) and dv.shape == (n,) and zv.shape == (n,)

    n1 = int((zv == 1.0).sum())
    n0 = n - n1
    sy1 = float(yv[zv == 1.0].mean())
    sy0 = float(yv[zv == 0.0].mean())
    pd1 = float(dv[zv == 1.0].mean())
    pd0 = float(dv[zv == 0.0].mean())
    itt_y = sy1 - sy0
    net = pd1 - pd0
    pc_max = min(pd1, 1.0 - pd0)
    pd_max = pc_max - net
    rng_ = float(yv.max() - yv.min())
    exp_lo = (itt_y - pd_max * rng_) / pc_max
    exp_hi = (itt_y + pd_max * rng_) / pc_max
    exp_width = exp_hi - exp_lo
    exp_estimate = 0.5 * (exp_lo + exp_hi)
    exp_wald = itt_y / net

    assert result["n"] == n
    assert result["itt_y"] == itt_y
    assert result["pi_net"] == net
    assert result["pi_c_max"] == pc_max
    assert result["pi_d_max"] == pd_max
    assert result["lower"] == exp_lo
    assert result["upper"] == exp_hi
    assert result["width"] == exp_width
    assert result["estimate"] == exp_estimate
    assert result["wald"] == exp_wald

    # width = 2 * pd_max * rng / pc_max >= 0, and lower <= upper.
    assert result["lower"] <= result["upper"]
    assert result["width"] >= 0.0


def test_bndnmt_edge():
    """Test edge case: a trivial design still yields a valid bound object."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_Z = np.random.default_rng(43)

    n = 100
    y = rng_y.normal(0.0, 1.0, n)
    D = _binary(rng_D, n, 0.3)
    Z = _binary(rng_Z, n, 0.5)

    result = bound_no_monotonicity(y, D, Z)
    assert isinstance(result, dict)
    assert "lower" in result and "upper" in result
    assert result["lower"] <= result["upper"]
