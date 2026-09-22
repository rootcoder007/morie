"""Tests for bndnpr.bound_nonparam_regr."""

from morie.fn import _array_core as np

from morie.fn.bndnpr import bound_nonparam_regr


def _kernel_regression(y, X, h, x0):
    """Nadaraya-Watson estimator of E[y | X = x0] with Gaussian kernel."""
    u = (X - x0) / h
    k = np.exp(-0.5 * u * u)
    return float(np.sum(k * y) / np.sum(k))


def _nw_p1(X, D, h, x0):
    u = (X - x0) / h
    k = np.exp(-0.5 * u * u)
    w1 = float(np.sum(k * (D == 1.0)))
    w0 = float(np.sum(k * (D == 0.0)))
    return w1 / (w1 + w0)


def test_bndnpr_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(44)
    rng_b = np.random.default_rng(45)

    n = 100
    y = rng_y.normal(0.0, 1.0, n)
    D = rng_d.integers(0, 2, n).astype(float)
    X = rng_x.normal(0.0, 1.0, n)
    bw = float(abs(rng_b.normal(0.0, 1.0))) + 1.0  # strictly positive

    result = bound_nonparam_regr(y, D, X, bw)

    # Independent reference computation from the documented formula.
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    slo = 0.0
    shi = 0.0
    for i in range(n):
        xi = X[i]
        # Nadaraya-Watson conditional means E(y | X=xi, D=t)
        w1_mask = (D == 1.0)
        w0_mask = (D == 0.0)
        u_all = (xi - X) / bw
        k_all = np.exp(-0.5 * u_all * u_all)
        w1 = float(np.sum(k_all * w1_mask))
        w0 = float(np.sum(k_all * w0_mask))
        s1 = float(np.sum(k_all * w1_mask * y))
        s0 = float(np.sum(k_all * w0_mask * y))
        wt = w1 + w0
        p1 = w1 / wt
        p0 = w0 / wt
        m1 = s1 / w1 if w1 > 0.0 else 0.0
        m0 = s0 / w0 if w0 > 0.0 else 0.0
        # Worst-case arms for binary treatment and bounded outcome.
        a1_lo = m1 * p1 + y_min * (1.0 - p1)
        a1_hi = m1 * p1 + y_max * (1.0 - p1)
        a0_lo = m0 * p0 + y_min * (1.0 - p0)
        a0_hi = m0 * p0 + y_max * (1.0 - p0)
        slo += a1_lo - a0_hi
        shi += a1_hi - a0_lo
    lo_ref = slo / n
    hi_ref = shi / n
    width_ref = hi_ref - lo_ref
    estimate_ref = 0.5 * (lo_ref + hi_ref)

    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate", "bw", "n"):
        assert key in result, f"missing key {key!r}"
    assert result["n"] == n
    assert result["bw"] == bw
    assert abs(result["lower"] - lo_ref) < 1e-10
    assert abs(result["upper"] - hi_ref) < 1e-10
    assert abs(result["width"] - width_ref) < 1e-10
    assert abs(result["estimate"] - estimate_ref) < 1e-10
    assert abs(result["width"] - (result["upper"] - result["lower"])) < 1e-10
    assert abs(result["estimate"] - 0.5 * (result["lower"] + result["upper"])) < 1e-10
    assert result["lower"] <= result["upper"]


def test_bndnpr_edge():
    """Test edge cases: very small bandwidth and a larger bandwidth."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(44)

    n = 50
    y = rng_y.normal(0.0, 1.0, n)
    D = rng_d.integers(0, 2, n).astype(float)
    X = rng_x.normal(0.0, 1.0, n)

    # Tiny bandwidth -> p1 in {0,1}, degenerate conditional means.
    bw_small = 1e-3
    r_small = bound_nonparam_regr(y, D, X, bw_small)
    assert isinstance(r_small, dict)
    assert r_small["n"] == n
    assert r_small["bw"] == bw_small
    assert r_small["lower"] <= r_small["upper"]

    # Large bandwidth -> bounds should not be wider than at smaller bandwidth
    # (averaged pointwise bounds are non-increasing in bandwidth width).
    bw_large = 10.0
    r_large = bound_nonparam_regr(y, D, X, bw_large)
    assert isinstance(r_large, dict)
    assert r_large["n"] == n
    assert r_large["bw"] == bw_large
    assert r_large["width"] <= r_small["width"] + 1e-10
