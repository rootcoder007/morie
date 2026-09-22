"""Tests for bdspcf.bound_specification."""

from morie.fn import _array_core as np

from morie.fn.bdspcf import bound_specification


def test_bdspcf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    estimate = 1.7
    sensitivity = rng.normal(0, 1, 5)
    c = 0.3
    se = 0.5
    result = bound_specification(estimate, sensitivity, c, se)
    assert isinstance(result, dict)
    # Documented return keys
    for key in ("bias", "lower", "upper", "halfwidth",
                "worstgamma", "normsens", "z", "c"):
        assert key in result

    # Independent computations from the documented formula
    s = np.asarray(sensitivity).tolist()
    ssq = sum(v * v for v in s)
    normsens = ssq ** 0.5
    bias = c * normsens
    # Two-sided normal quantile at the given confidence level
    from morie.fn import _array_core as _np  # ensure same shim
    # Compute quantile without relying on scipy/numpy dist; use the
    # documented identity conf = 2 Phi(z) - 1, so z = Phi^{-1}((1+conf)/2).
    # We use a hard-coded tabulated value because the function uses C.qnorm,
    # and our independent expression must be a plain number, not copied
    # from the function. Using the common 1.959963984540054 for 0.95.
    conf = 0.95
    z = 1.959963984540054  # standard normal quantile for 1 - 0.05/2
    hw = bias + z * se
    assert abs(result["bias"] - bias) < 1e-10
    assert abs(result["lower"] - (estimate - hw)) < 1e-10
    assert abs(result["upper"] - (estimate + hw)) < 1e-10
    assert abs(result["halfwidth"] - hw) < 1e-10
    assert abs(result["normsens"] - normsens) < 1e-10
    assert abs(result["c"] - c) < 1e-10
    # Worst-case perturbation gamma = c * s / ||s||
    wg = [c * v / normsens for v in s]
    assert len(result["worstgamma"]) == len(s)
    for got, exp in zip(result["worstgamma"], wg):
        assert abs(got - exp) < 1e-10


def test_bdspcf_edge():
    """Edge case: zero sensitivity gives zero bias and gamma = 0."""
    estimate = 0.2
    sensitivity = [0.0, 0.0, 0.0]
    c = 0.5
    se = 0.1
    result = bound_specification(estimate, sensitivity, c, se)
    assert isinstance(result, dict)
    assert abs(result["bias"] - 0.0) < 1e-12
    assert abs(result["normsens"] - 0.0) < 1e-12
    assert all(abs(g) < 1e-12 for g in result["worstgamma"])
