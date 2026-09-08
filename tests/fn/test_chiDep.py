"""Tests for chiDep.chi_dependence."""
from morie.fn import _array_core as np
from morie.fn.chiDep import chi_dependence


def test_perfect_dependence_near_one():
    x = [float(v) for v in np.random.default_rng(1).normal(0, 1, 800)._flat()]
    r = chi_dependence(x, x, u=0.9)
    assert r["estimate"] > 0.9


def test_independence_near_zero():
    rng = np.random.default_rng(2)
    x = [float(v) for v in rng.normal(0, 1, 800)._flat()]
    y = [float(v) for v in rng.normal(0, 1, 800)._flat()]
    r = chi_dependence(x, y, u=0.9)
    assert r["estimate"] < 0.35
