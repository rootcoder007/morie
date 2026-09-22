"""Tests for evgevp.evt_gev_pdf."""

from morie.fn import _array_core as np

from morie.fn.evgevp import evt_gev_pdf


def test_evgevp_basic():
    """Test basic functionality with a scalar xi (xi is a scalar parameter)."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    xi = 0.1  # xi is a scalar shape parameter
    result = evt_gev_pdf(x, mu, sigma, xi)
    assert isinstance(result, dict)
    # The function returns a RichResult-like dict with payload keys.
    assert "f" in result
    assert "mu" in result
    assert "sigma" in result
    assert "xi" in result
    assert "method" in result
    assert result["mu"] == 0.0
    assert result["sigma"] == 1.0
    assert result["xi"] == 0.1

    # Independent numeric check of the GEV density formula.
    # f(x) = (1/sigma) * t(x)^(xi+1) * exp(-t(x))
    # t(x) = [1 + xi*(x-mu)/sigma]^(-1/xi)
    import math
    arr = np.asarray(x) if hasattr(np, "asarray") else x
    # Convert to plain Python list for element-wise access.
    xs = list(arr)
    expected = []
    for v in xs:
        arg = 1.0 + 0.1 * (v - 0.0) / 1.0
        if arg <= 0:
            expected.append(float("nan"))
        else:
            t = arg ** (-1.0 / 0.1)
            val = (1.0 / 1.0) * (t ** (0.1 + 1.0)) * math.exp(-t)
            expected.append(val)
    got = list(result["f"])
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if e != e:  # nan
            import math as _m
            assert _m.isnan(g)
        else:
            assert abs(g - e) < 1e-10


def test_evgevp_edge():
    """Test edge cases with xi=0 (Gumbel limit)."""
    x = np.array([0.0])  # single-element input -> scalar output
    mu = 0.0
    sigma = 1.0
    xi = 0.0
    result = evt_gev_pdf(x, mu, sigma, xi)
    assert isinstance(result, dict)
    assert "f" in result
    # In the Gumbel limit f(x) = exp(-(x-mu)) * exp(-exp(-(x-mu))).
    val = result["f"]
    # Should be a scalar (single element).
    import math
    expected = math.exp(-(0.0 - 0.0)) * math.exp(-math.exp(-(0.0 - 0.0)))
    assert abs(float(val) - expected) < 1e-12
