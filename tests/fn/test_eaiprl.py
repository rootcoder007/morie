"""Tests for eaiprl.aipw_efficient_influence."""

from morie.fn import _array_core as np

from morie.fn.eaiprl import aipw_efficient_influence


def test_eaiprl_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_m = np.random.default_rng(44)
    rng_e = np.random.default_rng(45)

    y = rng_y.normal(0, 1, 100)
    D = rng_d.standard_normal(100) > 0  # binary treatment
    # outcome predictions must be a pair (m1, m0) of length n
    m1 = rng_m.normal(0, 1, 100)
    m0 = rng_m.normal(0, 1, 100)
    ml_outcome = (m1, m0)
    # propensities must lie strictly in (0, 1)
    ml_propensity = rng_e.uniform(0.1, 0.9, 100)
    X = rng_d.normal(0, 1, (100, 5))

    result = aipw_efficient_influence(y, D, X, ml_outcome, ml_propensity)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "influence" in result
    assert "ipw" in result
    assert "plugin" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 100
    assert len(result["influence"]) == 100

    # compute the ATE estimate independently from the formula
    dd = [float(v) for v in D]
    ys = [float(v) for v in y]
    m1l = [float(v) for v in m1]
    m0l = [float(v) for v in m0]
    el = [float(v) for v in ml_propensity]
    n = len(ys)
    phi = [
        dd[i] * (ys[i] - m1l[i]) / el[i] + m1l[i]
        - (1.0 - dd[i]) * (ys[i] - m0l[i]) / (1.0 - el[i]) - m0l[i]
        for i in range(n)
    ]
    expected_estimate = sum(phi) / n
    expected_ipw = sum(
        dd[i] * ys[i] / el[i] - (1.0 - dd[i]) * ys[i] / (1.0 - el[i])
        for i in range(n)
    ) / n
    expected_plugin = sum(m1l[i] - m0l[i] for i in range(n)) / n
    expected_se = (sum((v - expected_estimate) ** 2 for v in phi) / (n * n)) ** 0.5

    assert abs(result["estimate"] - expected_estimate) < 1e-12
    assert abs(result["se"] - expected_se) < 1e-12
    assert abs(result["ipw"] - expected_ipw) < 1e-12
    assert abs(result["plugin"] - expected_plugin) < 1e-12
    assert abs(result["ci_lower"] - (expected_estimate - 1.959963984540054 * expected_se)) < 1e-12
    assert abs(result["ci_upper"] - (expected_estimate + 1.959963984540054 * expected_se)) < 1e-12
    assert result["method"] == "augmented IPW ATE (Robins, Rotnitzky & Zhao 1994)"


def test_eaiprl_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_m = np.random.default_rng(44)
    rng_e = np.random.default_rng(45)

    y = rng_y.normal(0, 1, 100)
    D = rng_d.standard_normal(100) > 0
    m1 = rng_m.normal(0, 1, 100)
    m0 = rng_m.normal(0, 1, 100)
    ml_outcome = (m1, m0)
    ml_propensity = rng_e.uniform(0.1, 0.9, 100)
    X = rng_d.normal(0, 1, (100, 5))

    result = aipw_efficient_influence(y, D, X, ml_outcome, ml_propensity)
    assert isinstance(result, dict)
    assert result["n"] == 100
