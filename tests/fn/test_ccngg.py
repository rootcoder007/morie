"""Tests for ccngg.nakagawa_conditional_r2."""

from morie.fn import _array_core as np
from morie.fn.ccngg import nakagawa_conditional_r2


def test_ccngg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ccngg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y)
    assert isinstance(result, dict)


def test_nakagawa_r2_by_moments_recomputed():
    """OLS on X, one-way ANOVA moments of the residuals by cluster."""
    import pytest

    y = [3.1, 2.8, 4.0, 5.2, 4.9, 6.1, 2.2, 2.9, 3.4]
    x = [1.0, 0.5, 1.5, 2.0, 2.5, 3.0, 0.2, 0.9, 1.1]
    cl = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    n = 9
    mx, my = sum(x) / n, sum(y) / n
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    fit = [my + b * (a - mx) for a in x]
    var_f = sum((f - my) ** 2 for f in fit) / (n - 1)
    res = [c - f for c, f in zip(y, fit)]
    groups = [[res[i] for i in range(n) if cl[i] == k] for k in ("a", "b", "c")]
    gm = sum(res) / n
    ssb = sum(3 * (sum(g) / 3 - gm) ** 2 for g in groups)
    ssw = sum((v - sum(g) / 3) ** 2 for g in groups for v in g)
    msb, msw = ssb / 2, ssw / 6
    m0 = (n - 27 / n) / 2
    var_r = max((msb - msw) / m0, 0.0)
    tot = var_f + var_r + msw
    r = nakagawa_conditional_r2(y, [[a] for a in x], cluster=cl)
    assert r["r2_marginal"] == pytest.approx(var_f / tot, rel=1e-10)
    assert r["r2_conditional"] == pytest.approx((var_f + var_r) / tot, rel=1e-10)
    assert r["var_resid"] == pytest.approx(msw, rel=1e-10)
