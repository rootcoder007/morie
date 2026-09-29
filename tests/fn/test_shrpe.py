"""Tests for morie.fn.shrpe."""

from morie.fn import _array_core as np
from morie.fn.shrpe import shrpe


def test_shrpe_smoke():
    rng = np.random.default_rng(42)
    result = shrpe(returns=rng.uniform(10, 100, size=50))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.shrpe import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_sharpe_ratio_recomputed():
    import math

    import pytest

    r = [0.01, -0.02, 0.03, 0.015, 0.0, 0.02]
    ex = [v - 0.005 for v in r]
    m = sum(ex) / 6
    sd = math.sqrt(sum((v - m) ** 2 for v in ex) / 5)
    res = shrpe(r, risk_free=0.005)
    assert res.value == pytest.approx(m / sd, rel=1e-12)
