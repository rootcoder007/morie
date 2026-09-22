"""Tests for fzt42.fauzi_thm4_2_surv2_bias_var."""

from morie.fn import _array_core as np

from morie.fn.fzt42 import fauzi_thm4_2_surv2_bias_var


def test_fzt42_basic():
    """Test basic functionality."""
    t = 2.0
    n = 100
    h = 0.3
    surv = 0.6
    cumsurv = 0.7
    dg = 1.0
    d2g = -0.2
    density = 0.4
    mu2 = 1.0

    result = fauzi_thm4_2_surv2_bias_var(
        t, n, h, surv, cumsurv, dg, d2g, density, mu2=mu2
    )

    # Expected values computed independently from the documented formula:
    # b3 = dg^2 * density - d2g * surv
    # bias = (h^2 / 2) * b3 * mu2
    # variance = (2 * cumsurv - surv^2) / n
    # cov = surv * (1 - surv) / n
    expected_b3 = dg ** 2 * density - d2g * surv
    expected_bias = (h ** 2 / 2.0) * expected_b3 * mu2
    expected_var = (2.0 * cumsurv - surv ** 2) / n
    expected_cov = surv * (1.0 - surv) / n

    # The function returns a RichResult (dict-like) with the documented keys.
    assert hasattr(result, "__getitem__") or isinstance(result, dict)
    for key in ("bias", "variance", "b3", "cov", "h", "n", "method"):
        assert key in result, f"missing key {key!r} in result"

    assert result["b3"] == expected_b3
    assert result["bias"] == expected_bias
    assert result["variance"] == expected_var
    assert result["cov"] == expected_cov
    assert result["h"] == float(h)
    assert result["n"] == int(n)
    assert isinstance(result["method"], str)


def test_fzt42_edge():
    """Test edge cases: scalar inputs, large n, and default mu2."""
    t = 1.5
    n = 1000
    h = 0.5
    surv = 0.3
    cumsurv = 0.4
    dg = 2.0
    d2g = 0.1
    density = 0.8

    # Use default mu2 by omitting it.
    result = fauzi_thm4_2_surv2_bias_var(
        t, n, h, surv, cumsurv, dg, d2g, density
    )

    expected_b3 = dg ** 2 * density - d2g * surv
    expected_bias = (h ** 2 / 2.0) * expected_b3 * 1.0  # mu2 default
    expected_var = (2.0 * cumsurv - surv ** 2) / n
    expected_cov = surv * (1.0 - surv) / n

    assert hasattr(result, "__getitem__") or isinstance(result, dict)
    for key in ("bias", "variance", "b3", "cov", "h", "n", "method"):
        assert key in result, f"missing key {key!r} in result"

    assert result["b3"] == expected_b3
    assert result["bias"] == expected_bias
    assert result["variance"] == expected_var
    assert result["cov"] == expected_cov
